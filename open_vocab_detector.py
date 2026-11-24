from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Generic, TypeVar
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

    _MAX_WORDS_PER_LABEL = 2

    def __init__(self, prompt: str, det_threshold: float, nms_iou_threshold: float, use_fp16: bool,
                 model_id: str) -> None:

        if not 0 <= det_threshold <= 1 or not 0 <= nms_iou_threshold <= 1:
            raise ValueError("Detection and NMS thresholds must be between 0 and 1")

        self._classes = self.extract_user_prompt(prompt)
        self._prompt = self.create_model_prompt(self._classes)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = utils.get_supported_fp16(self._device.type) if use_fp16 else torch.float32
        self._det_threshold = det_threshold
        self._nms_iou_threshold = nms_iou_threshold
        self.init_model(model_id)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def classes(self) -> list[str]:
        return self._classes

    def init_model(self, model_id: str) -> None:
        """Load the processor and model."""
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id,
                                                                    torch_dtype=self._dtype).to(self.device).eval()
        self._processor, self.__model = processor, model

    @utils.timed
    def detect(self, frame: NDArray[np.uint8]) -> utils.DetectionResult:
        """Run the complete pipeline and return absolute XYXY boxes and labels."""
        prepared = self.preprocess(frame)
        with torch.inference_mode():
            output = self.__model(**prepared.inputs)
            return self.postprocess(output, prepared)

    @staticmethod
    def extract_user_prompt(prompt: str) -> list[str]:
        """Extract and normalize labels from the given prompt."""
        # if str is a path to a file, read its contents
        if Path(prompt).is_file():
            prompt = Path(prompt).read_text(encoding="utf-8")

        #If prompt has commas split by commas, otherwise split by newlines
        if "," in prompt:
            labels = [label.strip() for label in prompt.split(",") if label.strip()]
        else:
            labels = [label.strip() for label in prompt.splitlines() if label.strip()]
        return OpenVocabDetector.normalize_labels(labels)

    @staticmethod
    def normalize_labels(text_queries: list[str]) -> list[str]:
        """Lowercase labels and collapse whitespace, hyphens, and underscores into singlespaces."""
        if not text_queries or any(not query.strip() for query in text_queries):
            raise ValueError("Text queries must contain non-empty class labels")

        normalized_labels = []
        for query in text_queries:
            cleaned_query = re.sub(r"[-_\s]+", " ", query.lower()).strip()
            normalized_labels.append(cleaned_query)
            has_qualifier = len(cleaned_query.split()) == 2
            if has_qualifier:
                # Append the noun (last word) as a separate un-qualified label for multi-word fuzzy matching
                normalized_labels.append(cleaned_query.split()[-1])

        if any(len(label.split()) > OpenVocabDetector._MAX_WORDS_PER_LABEL for label in normalized_labels):
            raise ValueError(f"Each class label must contain at most {OpenVocabDetector._MAX_WORDS_PER_LABEL} words")

        print("Queried Labels: ", normalized_labels)

        return normalized_labels

    def prune_detections_with_spurious_labels(self,
                                              detections: utils.DetectionResult,
                                              apply_fuzzy_matching: bool = True) -> utils.DetectionResult:
        """ Since the queried classes can be a mix of single word and multi-word classes, sometimes the multi-word 
        labels inferred by the model might not be present in the queried class list as a combination but might individually
        be present. This ends up creating spurious combinations of words. For example and original query set of
        [luxury car, old furniture, boat] could possibly have predicted labels like "luxury furniture" or "old boat" 
        which don't exist in the queried classes but might make sense given a scene and it's semantics. This function removes 
        detections that don't match the queried classes. Boxes, scores, and labels are
        filtered together in-place, preserving their order and alignment.

        Args:
            detections (utils.DetectionResult): The detection results to be pruned.
            apply_fuzzy_matching (bool): Before throwing away a spurious label, this will try to match it against the 
            queried class list using fuzzy matching i.e.; It will try to match the noun of the predicted label with all
            the nouns in the queried class list.
        Returns:
            utils.DetectionResult: The pruned detection results.
        """
        detections.validate()

        # Split all class names into individual words for fuzzy matching.
        queried_nouns = set(label.split()[-1]
                            for label in self._classes)  # Extract the last word (noun) from each class label
        keep = []
        for idx, predicted_label in enumerate(detections.labels):
            if predicted_label in self._classes:
                matches = True
            elif apply_fuzzy_matching and predicted_label.split():
                # If a predicted label's noun matches any of the queried nouns, consider it a valid match
                predicted_noun = predicted_label.split()[-1]
                matches = predicted_noun in queried_nouns and predicted_noun in self._classes

                # The predicted label is reduced to its unqualified noun label in order to have a label that is in the original queried class list
                predicted_label = predicted_noun.lower()
            else:
                matches = predicted_label in self._classes  # looks for full-word exact match

            if matches:
                detections.labels[idx] = predicted_label
                keep.append(idx)

        indices = torch.tensor(keep, dtype=torch.int64, device=detections.boxes.device)
        detections.boxes = detections.boxes[indices]
        detections.scores = detections.scores[indices.to(detections.scores.device)]
        detections.labels = [detections.labels[idx] for idx in keep]
        return detections

    @abstractmethod
    def create_model_prompt(self, classes: list[str]) -> str:
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
