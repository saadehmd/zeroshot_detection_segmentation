import numpy as np
from numpy.typing import NDArray
import torch


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
