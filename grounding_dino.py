from dataclasses import dataclass
from xml.parsers.expat import model

import cv2
from open_vocab_detector import OpenVocabDetector
from numpy.typing import NDArray
import numpy as np
import torch
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoObjectDetectionOutput
from transformers.models.grounding_dino.image_processing_grounding_dino import GroundingDinoImageProcessor
import utils


@dataclass
class GroundingDinoInput:
    inputs: dict[str, torch.Tensor]
    offsets: torch.Tensor
    labels: list[str]
    image_size: tuple[int, int]


class GroundingDino(OpenVocabDetector[GroundingDinoInput, GroundingDinoObjectDetectionOutput]):
    """Grounding DINO with one configured class per query and class-aware NMS.

    Frames are OpenCV BGR images. Returned scores are the winning class's
    maximum token probability. No free-form text decoding is performed.
    """

    def promptify(self, classes: list[str]) -> str:
        return " . ".join(classes) + " ."

    def preprocess(self, frame: NDArray[np.uint8]) -> GroundingDinoInput:
        """Normalize queries, convert BGR to RGB, and prepare device-ready tensors."""
        if not utils.is_valid_image(frame):
            raise ValueError("Frame must be a non-empty BGR image with shape (H, W, 3)")

        inputs = self.__processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                  text=self.__prompt,
                                  return_tensors="pt",
                                  return_offsets_mapping=True,
                                  ).to(self.__device, self.__dtype)

        offsets = inputs.pop("offset_mapping")[0]
        return GroundingDinoInput(inputs=dict(inputs),
                                  offsets=offsets,
                                  labels=self.__classes,
                                  image_size=(frame.shape[0], frame.shape[1]),
                                  )

    def postprocess(self, output: GroundingDinoObjectDetectionOutput,
                    prepared: GroundingDinoInput) -> utils.DetectionResult:
        """Select canonical labels, threshold, resize boxes, and apply class-aware NMS."""
        boxes, scores, labels = utils.get_detection_labels_with_max_confidence(
            output.logits[0], output.pred_boxes[0], prepared.offsets,
            prepared.labels, prepared.image_size, self.__det_threshold,
        )
        detections = utils.DetectionResult(boxes, scores, labels)
        keep = utils.class_aware_nms(detections, self.__nms_iou_threshold)
        return utils.DetectionResult(
            boxes=boxes[keep],
            scores=scores[keep],
            text_labels=[labels[index] for index in keep.tolist()],
        )
