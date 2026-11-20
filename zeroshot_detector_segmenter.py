"""Detect and segment video objects; press Q or Escape to stop.

Use Up/Down to adjust the skip step (initially 10), and Right/Left to seek.
Skipped frames are not processed or saved.
Usage: python zeroshot_detector_segmenter.py input.mp4 classes.txt --output-vid annotated.mp4
The required UTF-8 classes file contains one label per line. Blank lines are
ignored; labels are lowercased and deduplicated in file order.
Omit --output-vid to display results without saving them.
ZeroShotDetectorSegmenter.detect(frame) uses GroundingDino and returns detections
and inference time in milliseconds. Pass fp16=True to select reduced precision.
Detection assigns each box the configured label with the highest token score,
excluding special/separator tokens, before thresholding and class-aware NMS.
No text threshold or free-form label decoding is used.
segment(frame, detections) returns one binary uint8 mask per box, shaped
(number_of_boxes, height, width). draw() returns an annotated frame copy.
The displayed inference time includes detection and segmentation.
"""

import argparse
from colorsys import hsv_to_rgb
from functools import partial
import math
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
from transformers import Sam2Processor, Sam2Model
from detector_gui import DetectorGui
from grounding_dino import GroundingDino
import utils


class ZeroShotDetectorSegmenter:

    def __init__(self, classes_file: Path | str, fp16: bool = False) -> None:
        labels = Path(classes_file).read_text(encoding="utf-8").splitlines()
        self.classes = list(dict.fromkeys(" ".join(label.lower().split()) for label in labels if label.strip()))
        if not self.classes:
            raise ValueError("The classes file must contain at least one non-empty label")
        self.label_colors = self.initColorPalette(self.classes)
        self._fp16 = fp16
        self.initDetector()
        self.initSegmenter()

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
        """Create the shared detector with this app's configuration."""
        self._detector = GroundingDino(self.classes,
                                       det_threshold=0.2,
                                       nms_iou_threshold=0.5,
                                       use_fp16=self._fp16,
                                       model_id=model_id)

    def initSegmenter(self, model_id: str = "facebook/sam2-hiera-tiny") -> None:
        """Load a segmenter and processor, replacing the cache on success."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} for inference")
        dtype = (utils.get_supported_fp16(device) if self._fp16 else torch.float32)
        processor = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self._sam_processor, self._sam_model = processor, model

    @utils.timed
    def segment(self, frame: NDArray[np.uint8], detections: utils.DetectionResult) -> NDArray[np.uint8]:
        """Return the highest-IoU binary mask per box at the original frame size."""

        if len(detections.boxes) == 0:
            return np.empty((0, *frame.shape[:2]), dtype=np.uint8)
        device = self._sam_model.device
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._sam_processor(images=rgb_frame,
                                     input_boxes=[detections.boxes.detach().cpu().tolist()],
                                     return_tensors="pt",
                                     ).to(device, self._sam_model.dtype)
        with torch.inference_mode():
            outputs = self._sam_model(**inputs)
            best_indices = outputs.iou_scores[0].argmax(dim=-1)
            box_indices = torch.arange(len(detections.boxes), device=best_indices.device)
            best_masks = outputs.pred_masks[0][box_indices, best_indices]
            masks = self._sam_processor.post_process_masks(best_masks.unsqueeze(0).unsqueeze(2).float().cpu(),
                                                           inputs["original_sizes"].cpu(),
                                                           binarize=True,
                                                           )[0][:, 0]
        return masks.detach().cpu().numpy().astype(np.uint8)

    def draw(self,
             frame: NDArray[np.uint8],
             detections: utils.DetectionResult,
             masks: NDArray[np.uint8] | None,
             timing_text: str | None = None,
             ) -> NDArray[np.uint8]:
        """Return a frame copy with masks, boxes, labels, scores, and optional timing text."""
        boxes = detections.boxes.detach().cpu().tolist()
        scores = detections.scores.detach().cpu().tolist()
        labels = detections.labels
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
        if timing_text is not None:
            for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 2)]:
                cv2.putText(frame, timing_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
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

    def _process_frame(self, frame: NDArray[np.uint8], *, output_path: Path = None) -> None:
        """Annotate a frame in place and optionally write it to the output video."""
        if output_path is not None and self._writer is None:
            height, width = frame.shape[:2]
            codec = "mp4v" if output_path.suffix.lower() == ".mp4" else "MJPG"
            self._writer = cv2.VideoWriter(str(output_path),
                                           cv2.VideoWriter_fourcc(*codec),
                                           self._gui.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")

        results, det_ms = self._detector.detect(frame)
        masks, seg_ms = self.segment(frame, results)

        timing_text = f"Inference: {det_ms + seg_ms:.1f} ms"
        frame[:] = self.draw(frame, results, masks, timing_text)

        if self._writer is not None:
            self._writer.write(frame)

    def main(self, args: argparse.Namespace) -> None:
        self._gui = DetectorGui(str(args.video), skip_step=10)
        self._writer: cv2.VideoWriter | None = None
        try:
            self._gui.run(partial(self._process_frame, output_path=args.output_vid))
            if self._gui.frame_index < 0:
                raise RuntimeError(f"Could not read the first frame: {args.video}")
        finally:
            if self._writer is not None:
                self._writer.release()


if __name__ == "__main__":
    args = ZeroShotDetectorSegmenter.parse_args()
    ZeroShotDetectorSegmenter(args.classes_file, fp16=args.fp16).main(args)
