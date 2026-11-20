from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TypedDict
from numpy.typing import NDArray
import numpy as np
import utils
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PreparedInput = TypeVar("PreparedInput")
ModelOutput = TypeVar("ModelOutput")


class OpenVocabDetector(ABC, Generic[PreparedInput, ModelOutput]):
    """Detector contract; subclasses initialize after setting their configuration.

    Preprocessing returns per-call context needed by postprocessing, rather than
    storing frame sizes, queries, or token offsets on the detector instance.
    """

    def __init__(self,
                 classes: list[str],
                 det_threshold: float,
                 nms_iou_threshold: float,
                 use_fp16: bool,
                 model_id: str) -> None:

        if not 0 <= det_threshold <= 1 or not 0 <= nms_iou_threshold <= 1:
            raise ValueError("Detection and NMS thresholds must be between 0 and 1")

        self.__classes = self.normalize_labels(classes)
        self.__prompt = self.promptify(self.__classes)
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__dtype = utils.get_supported_fp16(self.__device.type) if use_fp16 else torch.float32
        self.__det_threshold = det_threshold
        self.__nms_iou_threshold = nms_iou_threshold
        self.init_model(model_id)

    def init_model(self, model_id: str) -> None:
        """Load the processor and model."""
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id,
                                                                    torch_dtype=self.__dtype).to(self.__device).eval()
        self.__processor, self.__model = processor, model

    def detect(self, frame: NDArray[np.uint8]) -> utils.DetectionResult:
        """Run the complete pipeline and return absolute XYXY boxes and labels."""
        prepared = self.preprocess(frame)
        with torch.inference_mode():
            output = self.__model(**prepared.inputs)
            return self.postprocess(output, prepared)

    @staticmethod
    def normalize_labels(text_queries: list[str]) -> list[str]:
        if not text_queries or any(not query.strip() for query in text_queries):
            raise ValueError("Text queries must contain non-empty class labels")
        return list(dict.fromkeys(" ".join(query.lower().split()) for query in text_queries))

    @abstractmethod
    def promptify(self, classes: list[str]) -> str:
        """Converts a list of class names into a model-specific prompt format."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, frame: NDArray[np.uint8]) -> PreparedInput:
        """Prepare model inputs and per-frame postprocessing context."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, output: ModelOutput, prepared: PreparedInput) -> utils.DetectionResult:
        """Convert model output into filtered, frame-sized detections."""
        raise NotImplementedError
