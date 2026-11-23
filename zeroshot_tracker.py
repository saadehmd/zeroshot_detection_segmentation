"""Detect and track video objects; press Q or Escape to stop.

Use Up/Down to adjust the skip step (initially 10), and Right/Left to seek.
Seeking refreshes detection/tracking. Skipped frames are not processed or saved.
Usage: python zeroshot_tracker.py input.mp4 classes.txt --output-vid annotated.mp4
The required UTF-8 classes file contains one label per line. Blank lines are
ignored; labels are lowercased and deduplicated in file order.
Omit --output-vid to display results without saving them.
Detection uses GroundingDino. Pass fp16=True to select reduced precision.
Detection assigns each box the configured label with the highest token score,
excluding special/separator tokens, before thresholding and class-aware NMS.
No text threshold or free-form label decoding is used.
draw() returns an annotated frame copy.
Detection runs on frames 0, 20, 40, ...; SAM2 propagates box-prompted masks
between refreshes. Each refresh replaces the tracking session (IDs are local
to that interval). Boxes are derived from visible masks; labels and confidence
scores are retained from the latest detection. Empty/invisible masks are hidden.
With include_invisible=True, an empty mask retains its most recent valid box
(or its detector box until a valid tracked mask is available).
The displayed inference time includes detection (when run) and tracking.
"""

import argparse
from pathlib import Path
from typing import override

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
from transformers import Sam2VideoProcessor, Sam2VideoModel
from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession
from zeroshot_detector_segmenter import ZeroShotDetectorSegmenter
import utils


class ZeroShotTracker(ZeroShotDetectorSegmenter):
    _detection_threshold = 0.35
    _cli_description = __doc__

    @override
    def __init__(self, classes_file: Path | str, fp16: bool = False) -> None:
        super().__init__(classes_file, fp16=fp16)
        self._tracking_session: Sam2VideoInferenceSession | None = None
        self._tracking_labels: list[str] = []
        self._tracking_scores = torch.empty(0)
        self._tracking_boxes = torch.empty((0, 4))
        self._tracking_frame_idx = 0
        self._previous_frame_index: int | None = None

    @override
    def initSegmenter(self, model_id: str = "facebook/sam2-hiera-tiny") -> None:
        """Load a segmenter and processor, replacing the cache on success."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} for inference")
        dtype = utils.get_supported_fp16(device) if self._fp16 else torch.float32
        processor = Sam2VideoProcessor.from_pretrained(model_id)
        model = Sam2VideoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self._sam_processor, self._sam_model = processor, model
        self._tracking_session = None

    def __start_tracking_session(self, detections: utils.DetectionResult, height: int, width: int):
        boxes = detections.boxes
        self._tracking_session = None
        self._tracking_frame_idx = 0
        self._tracking_labels = list(detections.labels)
        self._tracking_scores = detections.scores.detach().cpu().clone()
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
            box = utils.mask_to_bbox(mask)
            if box is not None:
                self._tracking_boxes[obj_id] = torch.tensor(box, dtype=torch.float32)
                tracked_boxes.append(box)
                selected.append(index)
            elif include_invisible:
                tracked_boxes.append(self._tracking_boxes[obj_id].tolist())
                selected.append(index)
        return tracked_boxes, selected

    @utils.timed
    def _track(self,
               frame: NDArray[np.uint8],
               detections: utils.DetectionResult | None = None,
               *,
               include_invisible: bool = False,
               ) -> tuple[utils.DetectionResult, NDArray[np.uint8]]:
        """Track masks; included invisible objects retain their last valid boxes."""
        height, width = frame.shape[:2]
        if detections is not None:
            self.__start_tracking_session(detections, height, width)

        if self._tracking_session is None:
            empty_detections = utils.DetectionResult(torch.empty((0, 4)), torch.empty(0), [])
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
        return utils.DetectionResult(
            boxes=torch.tensor(tracked_boxes, dtype=torch.float32).reshape(-1, 4),
            scores=self._tracking_scores[object_ids],
            labels=[self._tracking_labels[obj_id] for obj_id in object_ids],
        ), masks[selected]

    @override
    def _infer_frame(self, frame: NDArray[np.uint8], ) -> tuple[utils.DetectionResult, NDArray[np.uint8], float]:
        frame_index = self._gui.frame_index
        first_frame = self._previous_frame_index is None
        detections = None
        det_ms = 0.0
        frames_skipped_by_user = not first_frame and frame_index != self._previous_frame_index + 1
        if first_frame or frames_skipped_by_user or frame_index % 20 == 0:
            detections, det_ms = self._detector.detect(frame)
        (results, masks), track_ms = self._track(frame, detections)
        self._previous_frame_index = frame_index

        return results, masks, det_ms + track_ms

    @override
    def main(self, args: argparse.Namespace) -> None:
        self._previous_frame_index = None
        self._tracking_session = None
        try:
            super().main(args)
        finally:
            self._tracking_session = None


if __name__ == "__main__":
    args = ZeroShotTracker.parse_args()
    ZeroShotTracker(args.classes_file, fp16=args.fp16).main(args)
