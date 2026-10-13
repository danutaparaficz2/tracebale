"""Classifiers label a cropped image of one component."""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


class ReferenceClassifier:
    """Nearest-reference classifier on mean colour; needs no ML dependencies.

    references: label -> list of example RGB crops (arrays) or RGB tuples.
    """

    def __init__(self, references: Dict[str, Sequence]):
        self._refs = []
        for label, examples in references.items():
            for ex in examples:
                self._refs.append((label, self._feature(ex)))
        if not self._refs:
            raise ValueError("at least one reference is required")

    @staticmethod
    def _feature(x) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(-1, 3).mean(axis=0)

    def classify(self, crop: np.ndarray, mask: np.ndarray) -> Tuple[str, float]:
        feat = self._feature(crop[mask]) if mask.any() else self._feature(crop)
        dists = [(np.linalg.norm(feat - f), label) for label, f in self._refs]
        d, label = min(dists)
        return label, float(1.0 / (1.0 + d / 255.0))


class ClipClassifier:
    """Zero-shot labelling with CLIP (needs the `clip` extra)."""

    def __init__(self, labels: List[str], model: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.labels = labels
        self._model = CLIPModel.from_pretrained(model)
        self._proc = CLIPProcessor.from_pretrained(model)

    def classify(self, crop: np.ndarray, mask: np.ndarray) -> Tuple[str, float]:
        import torch

        masked = np.where(mask[..., None], crop, 255).astype(np.uint8)
        inputs = self._proc(text=[f"a photo of a {l}" for l in self.labels],
                            images=Image.fromarray(masked), return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = self._model(**inputs).logits_per_image.softmax(dim=1)[0]
        i = int(probs.argmax())
        return self.labels[i], float(probs[i])
