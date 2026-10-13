from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class Component:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    area: int
    mask: np.ndarray


class ComponentRecognizer:
    """Segment an image, then classify each segment to identify its component."""

    def __init__(self, segmenter, classifier, min_area: int = 1):
        self.segmenter = segmenter
        self.classifier = classifier
        self.min_area = min_area

    def recognize(self, image) -> List[Component]:
        if not isinstance(image, np.ndarray):
            image = np.asarray(Image.open(image).convert("RGB"))
        components = []
        for mask in self.segmenter.segment(image):
            area = int(mask.sum())
            if area < self.min_area:
                continue
            ys, xs = np.where(mask)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            label, score = self.classifier.classify(image[y0:y1, x0:x1], mask[y0:y1, x0:x1])
            components.append(Component(label, score, (x0, y0, x1, y1), area, mask))
        return components
