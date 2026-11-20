from dataclasses import dataclass
from time import time

import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.ops import batched_nms
import time
from functools import wraps


def torch_sync_if_cuda_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        torch_sync_if_cuda_available()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        torch_sync_if_cuda_available()
        end = time.perf_counter()
        duration = (end - start) * 1000
        return result, duration

    return wrapper


@dataclass
class DetectionResult:
    """Detection tensors and labels with shapes and counts checked at creation."""

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: list[str]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check shapes and matching counts, including for zero detections."""
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
            raise ValueError("DetectionResult boxes must have shape (N, 4)")
        if self.scores.ndim != 1 or not len(self.boxes) == len(self.scores) == len(self.labels):
            raise ValueError("DetectionResult boxes, scores, and labels must have matching counts")


def is_valid_image(frame: NDArray[np.uint8]) -> bool:
    """Check if the frame is a non-empty BGR image with shape (H, W, 3)."""
    return frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0


def get_detection_labels_with_max_confidence(logits: torch.Tensor,
                                             pred_boxes: torch.Tensor,
                                             offsets: torch.Tensor,
                                             labels: list[str],
                                             image_size: tuple[int, int],
                                             threshold: float,
                                             ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Assign each query its highest-scoring class, then threshold and resize XYXY boxes.

    Offsets must come from tokenizing ' . '.join(labels) + ' .'. A class score
    is its maximum token probability; separator and special tokens are excluded.
    """
    token_offsets = offsets.detach().cpu().tolist()
    class_scores = []
    start = 0
    for label in labels:
        end = start + len(label)
        indices = [
            index for index, (left, right) in enumerate(token_offsets)
            if right > left and left >= start and right <= end and index < logits.shape[-1]
        ]
        if not indices:
            raise ValueError(f"No prompt tokens found for class {label!r}; check prompt truncation")
        class_scores.append(logits[:, indices].sigmoid().amax(dim=-1))
        start = end + 3
    scores, class_ids = torch.stack(class_scores, dim=-1).max(dim=-1)
    keep = scores > threshold
    centers, sizes = pred_boxes[keep].chunk(2, dim=-1)
    boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
    height, width = image_size
    boxes = boxes * boxes.new_tensor([width, height, width, height])
    return boxes, scores[keep], [labels[index] for index in class_ids[keep].tolist()]


def class_aware_nms(detections: DetectionResult, iou_threshold: float = 0.5) -> torch.Tensor:
    """Return surviving XYXY box indices in descending score order.

    Suppress lower-scoring boxes only when they share the same label and their
    IoU with a kept box exceeds the threshold.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be between 0 and 1")

    label_ids = {label: index for index, label in enumerate(dict.fromkeys(detections.labels))}

    classes = torch.tensor([label_ids[label] for label in detections.labels],
                           dtype=torch.int64,
                           device=detections.boxes.device,
                           )
    return batched_nms(detections.boxes.float(),
                       detections.scores.to(device=detections.boxes.device, dtype=torch.float32),
                       classes,
                       iou_threshold,
                       )


def get_supported_fp16(device: str) -> torch.dtype:

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU is not available. Can't use FP16. Re-initialize without the --fp16 arg.")

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def mask_to_bbox(mask: NDArray[np.uint8]) -> list[int] | None:
    """Return [x1, y1, x2, y2] with exclusive maxima, or None for an empty mask."""
    if mask.ndim != 2:
        raise ValueError("Mask must have shape (height, width)")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
