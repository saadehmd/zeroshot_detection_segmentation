import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.ops import batched_nms


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: list[str],
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Return kept XYXY box indices, sorted by score, suppressing within each label."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Boxes must have shape (N, 4)")
    if scores.ndim != 1 or not len(boxes) == len(scores) == len(labels):
        raise ValueError("Boxes, scores, and labels must have matching counts")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be between 0 and 1")
    normalized_labels = [" ".join(label.lower().split()) for label in labels]
    label_ids = {label: index for index, label in enumerate(dict.fromkeys(normalized_labels))}
    classes = torch.tensor(
        [label_ids[label] for label in normalized_labels],
        dtype=torch.int64,
        device=boxes.device,
    )
    return batched_nms(
        boxes.float(), scores.to(device=boxes.device, dtype=torch.float32),
        classes, iou_threshold,
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
