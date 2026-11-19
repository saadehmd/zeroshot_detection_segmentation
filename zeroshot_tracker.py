"""Detect and segment video objects; press Q or Escape to stop.

Press Right Arrow to skip the next 10 frames and refresh detection/tracking.
Skipped frames are not processed or included in the saved output video.
Usage: python zeroshot_detector_segmenter.py input.mp4 classes.txt --output-vid annotated.mp4
The required UTF-8 classes file contains one label per line. Blank lines are
ignored; labels are lowercased and deduplicated in file order.
Omit --output-vid to display results without saving them.
OpenVocabSegmentationApp.detect(frame) uses the instance's classes and accepts a BGR frame
and returns detections
and inference time in milliseconds. The constructor initializes both models
and processors; pass fp16=True to select reduced precision.
initDetector(model_id) explicitly loads or replaces the cached detector using
Auto factories. detect() requires compatible grounded-detection processing.
Detection assigns each box the configured label with the highest token score,
excluding special/separator tokens, before thresholding and class-aware NMS.
No text threshold or free-form label decoding is used.
segment(frame, detections) returns one binary uint8 mask per box, shaped
(number_of_boxes, height, width). draw() returns an annotated frame copy.
Detection runs on frames 0, 10, 20, ...; SAM2 propagates box-prompted masks
between refreshes. Each refresh replaces the tracking session (IDs are local
to that interval). Boxes are derived from visible masks; labels and confidence
scores are retained from the latest detection. Empty/invisible masks are hidden.
With include_invisible=True, an empty mask retains its most recent valid box
(or its detector box until a valid tracked mask is available).
The displayed inference time includes detection (when run) and tracking.
"""

import argparse
from colorsys import hsv_to_rgb
import math
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2VideoProcessor, Sam2VideoModel, )
from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession
from utils import mask_to_bbox, get_supported_fp16, class_aware_nms, get_detection_labels_with_max_confidence


class DetectionResult(TypedDict):
    boxes: torch.Tensor
    scores: torch.Tensor
    text_labels: list[str]


class OpenVocabSegmentationApp:

    def __init__(self, classes_file: Path | str, fp16: bool = False) -> None:
        labels = Path(classes_file).read_text(encoding="utf-8").splitlines()
        self.classes = list(dict.fromkeys(" ".join(label.lower().split()) for label in labels if label.strip()))
        if not self.classes:
            raise ValueError("The classes file must contain at least one non-empty label")
        self.label_colors = self.initColorPalette(self.classes)
        self._fp16 = fp16
        self.initDetector()
        self.initSegmenter()
        self._tracking_session: Sam2VideoInferenceSession | None = None
        self._tracking_labels: list[str] = []
        self._tracking_scores = torch.empty(0)
        self._tracking_boxes = torch.empty((0, 4))
        self._tracking_frame_idx = 0

    @staticmethod
    def initColorPalette(classes: list[str]) -> dict[str, tuple[int, int, int]]:
        """Assign each unique label an evenly spaced hue, in OpenCV BGR order."""
        labels = list(dict.fromkeys(classes))
        colors = {}
        for index, label in enumerate(labels):
            red, green, blue = hsv_to_rgb(index / len(labels), 0.75, 1.0)
            colors[label] = (round(blue * 255), round(green * 255), round(red * 255))
        return colors

    def initDetector(self, model_id: str = "IDEA-Research/grounding-dino-tiny") -> None:
        """Load a zero-shot detector and processor, replacing the cache on success."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} for inference")

        dtype = get_supported_fp16(device) if self._fp16 else torch.float32
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self._det_processor, self._det_model = processor, model

    def initSegmenter(self, model_id: str = "facebook/sam2-hiera-tiny") -> None:
        """Load a segmenter and processor, replacing the cache on success."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} for inference")
        dtype = get_supported_fp16(device) if self._fp16 else torch.float32
        processor = Sam2VideoProcessor.from_pretrained(model_id)
        model = Sam2VideoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self._sam_processor, self._sam_model = processor, model
        self._tracking_session = None

    def detect(self, frame: NDArray[np.uint8]) -> tuple[DetectionResult, float]:
        """Return one configured label per box, class-aware NMS (IoU 0.5), and ms."""
        text_prompt = " . ".join(self.classes) + " ."

        device = self._det_model.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._det_processor(images=rgb_frame,
                                     text=text_prompt,
                                     return_tensors="pt",
                                     return_offsets_mapping=True).to(device, self._det_model.dtype)
        offsets = inputs.pop("offset_mapping")[0]
        with torch.inference_mode():
            outputs = self._det_model(**inputs)
            boxes, scores, labels = get_detection_labels_with_max_confidence(
                outputs.logits[0], outputs.pred_boxes[0], offsets,
                self.classes, frame.shape[:2], threshold=0.35,
            )
            results = {"boxes": boxes, "scores": scores, "text_labels": labels}
        keep = class_aware_nms(results["boxes"], results["scores"], results["text_labels"])
        results = {
            "boxes": results["boxes"][keep],
            "scores": results["scores"][keep.to(results["scores"].device)],
            "text_labels": [results["text_labels"][index] for index in keep.tolist()],
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_ms = (perf_counter() - start) * 1000
        return results, inference_ms

    def __start_tracking_session(self, detections: DetectionResult, height: int, width: int):
        boxes = detections["boxes"]
        self._tracking_session = None
        self._tracking_frame_idx = 0
        self._tracking_labels = list(detections["text_labels"])
        self._tracking_scores = detections["scores"].detach().cpu().clone()
        self._tracking_boxes = boxes.detach().cpu().to(dtype=torch.float32).clone()
        if len(boxes):
            self._tracking_session = self._sam_processor.init_video_session(inference_device=self._sam_model.device,
                                                                            dtype=self._sam_model.dtype)

            self._sam_processor.add_inputs_to_inference_session(inference_session=self._tracking_session,
                                                                frame_idx=0,
                                                                obj_ids=list(range(len(boxes))),
                                                                input_boxes=[boxes.detach().cpu().tolist()],
                                                                original_size=(height, width))

    def __get_tracked_boxes_from_masks(self, object_ids: list[int], masks: NDArray[np.uint8],
                                       include_invisible: bool) -> tuple[list[list[float]], list[int]]:
        tracked_boxes = []
        selected = []
        for index, mask in enumerate(masks):
            obj_id = object_ids[index]
            box = mask_to_bbox(mask)
            if box is not None:
                self._tracking_boxes[obj_id] = torch.tensor(box, dtype=torch.float32)
                tracked_boxes.append(box)
                selected.append(index)
            elif include_invisible:
                tracked_boxes.append(self._tracking_boxes[obj_id].tolist())
                selected.append(index)
        return tracked_boxes, selected

    def track(self,
              frame: NDArray[np.uint8],
              detections: DetectionResult | None = None,
              *,
              include_invisible: bool = False,
              ) -> tuple[DetectionResult, NDArray[np.uint8]]:
        """Track masks; included invisible objects retain their last valid boxes."""
        height, width = frame.shape[:2]
        if detections is not None:
            self.__start_tracking_session(detections, height, width)

        if self._tracking_session is None:
            empty_detections = {"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "text_labels": []}
            empty_masks = np.empty((0, height, width), dtype=np.uint8)
            return (empty_detections, empty_masks)

        inputs = self._sam_processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                     return_tensors="pt").to(self._sam_model.device, self._sam_model.dtype)
        with torch.inference_mode():
            outputs = self._sam_model(inference_session=self._tracking_session,
                                      frame_idx=self._tracking_frame_idx,
                                      frame=inputs["pixel_values"][0])

            masks = (self._sam_processor.post_process_masks([outputs.pred_masks.float().cpu()],
                                                            original_sizes=[(height, width)],
                                                            binarize=True,
                                                            )[0][:, 0].cpu().numpy().astype(np.uint8))
        self._tracking_frame_idx += 1
        if outputs.object_score_logits is not None:
            absent = (outputs.object_score_logits.detach().reshape(-1) <= 0).cpu().numpy()
            masks[absent] = 0

        tracked_boxes, selected = self.__get_tracked_boxes_from_masks(outputs.object_ids, masks, include_invisible)
        object_ids = [outputs.object_ids[index] for index in selected]
        return {
            "boxes": torch.tensor(tracked_boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": self._tracking_scores[object_ids],
            "text_labels": [self._tracking_labels[obj_id] for obj_id in object_ids],
        }, masks[selected]

    def draw(self, frame: NDArray[np.uint8], detections: DetectionResult, masks: NDArray[np.uint8] | None,
             ) -> NDArray[np.uint8]:
        """Return a frame copy with mask overlays, boxes, labels, and scores."""
        boxes = detections["boxes"].detach().cpu().tolist()
        scores = detections["scores"].detach().cpu().tolist()
        labels = detections["text_labels"]
        if not len(boxes) == len(scores) == len(labels):
            raise ValueError("Detection boxes, scores, and labels must have matching counts")
        if masks is not None and masks.shape != (len(boxes), *frame.shape[:2]):
            raise ValueError("Masks must have shape (N, frame height, frame width)")
        unknown_labels = set(labels) - self.label_colors.keys()
        if unknown_labels:
            raise ValueError(f"Detected labels are not in the classes file: {sorted(unknown_labels)}")
        colors = [self.label_colors[label] for label in labels]
        frame = frame.copy()
        if masks is not None:
            overlay = np.zeros_like(frame)
            foreground = np.zeros(frame.shape[:2], dtype=bool)
            # Apply higher-confidence colors last without repeatedly blending overlaps.
            for index in sorted(range(len(boxes)), key=lambda i: scores[i]):
                mask = masks[index] != 0
                overlay[mask] = colors[index]
                foreground |= mask
            frame[foreground] = (frame[foreground] * 0.5 + overlay[foreground] * 0.5).astype(np.uint8)
        for box, label, score, color in zip(boxes, labels, scores, colors):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{label}: {score:.2f}", (max(0, x1), max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        )
        return frame

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse and validate CLI arguments before loading models."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("video", type=Path, help="Path to the input video")
        parser.add_argument("classes_file", type=Path, help="Required UTF-8 text file, one class label per line")
        parser.add_argument("--output-vid", type=Path, help="Optional output video path (.mp4 or .avi)")
        parser.add_argument("--fp16", action="store_true", help="Use FP16 precision for inference")
        args = parser.parse_args()
        if not args.video.is_file():
            parser.error(f"Input video does not exist: {args.video}")
        if not args.classes_file.is_file():
            parser.error(f"Classes file does not exist: {args.classes_file}")
        if args.output_vid:
            if args.output_vid.resolve() == args.video.resolve():
                parser.error("Input and output video paths must be different")
            if args.output_vid.suffix.lower() not in {".mp4", ".avi"}:
                parser.error("--output-vid must have an .mp4 or .avi extension")
        return args

    def main(self, args: argparse.Namespace) -> None:
        capture = cv2.VideoCapture(str(args.video))
        writer = None
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {args.video}")
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not read the first frame: {args.video}")

            if args.output_vid:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    raise RuntimeError(f"Invalid source video frame rate: {fps}")
                height, width = frame.shape[:2]
                codec = "mp4v" if args.output_vid.suffix.lower() == ".mp4" else "MJPG"
                writer = cv2.VideoWriter(str(args.output_vid), cv2.VideoWriter_fourcc(*codec), fps, (width, height), )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {args.output_vid}")

            window_name = "Grounding DINO - Right: skip 10 frames | Q / Esc: quit"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            self._tracking_session = None
            frame_idx = 0
            refresh_detection = False
            while success:
                device = self._sam_model.device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = perf_counter()
                if refresh_detection or frame_idx % 20 == 0:
                    # Perform detection every 10 frames only.
                    detections, _ = self.detect(frame)
                    results, masks = self.track(frame, detections)
                    refresh_detection = False
                else:
                    # Use SAM2's mask tracking capability for subsequent frames.
                    results, masks = self.track(frame)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_ms = (perf_counter() - start) * 1000
                frame = self.draw(frame, results, masks)

                timing_text = f"Inference: {inference_ms:.1f} ms"
                for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 2)]:
                    cv2.putText(frame, timing_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, )
                if writer is not None:
                    writer.write(frame)
                cv2.imshow(window_name, frame)
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (65363, 2555904, 16777236):
                    for _ in range(10):
                        if not capture.grab():
                            break
                        frame_idx += 1
                    self._tracking_session = None
                    refresh_detection = True
                success, frame = capture.read()
                frame_idx += 1
        finally:
            self._tracking_session = None
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = OpenVocabSegmentationApp.parse_args()
    OpenVocabSegmentationApp(args.classes_file, fp16=args.fp16).main(args)
