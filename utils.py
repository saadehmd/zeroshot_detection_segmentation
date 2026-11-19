import numpy as np
from numpy.typing import NDArray
import torch
from torchvision.ops import batched_nms


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


def class_aware_nms(boxes: torch.Tensor,
                    scores: torch.Tensor,
                    labels: list[str],
                    iou_threshold: float = 0.5,
                    ) -> torch.Tensor:
    """Return kept XYXY indices sorted by score; labels must already be canonical."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Boxes must have shape (N, 4)")
    if scores.ndim != 1 or not len(boxes) == len(scores) == len(labels):
        raise ValueError("Boxes, scores, and labels must have matching counts")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be between 0 and 1")
    label_ids = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    classes = torch.tensor([label_ids[label] for label in labels], dtype=torch.int64, device=boxes.device, )
    return batched_nms(boxes.float(), scores.to(device=boxes.device, dtype=torch.float32), classes, iou_threshold, )


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
