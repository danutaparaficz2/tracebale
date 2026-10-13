"""Segment an image with SAM, then classify each segment with CLIP."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence

import numpy as np
from PIL import Image


@dataclass
class Detection:
    label: str
    score: float
    box: tuple  # (x0, y0, x1, y1) in pixels
    area: int
    mask: np.ndarray = None  # bool HxW

    def to_dict(self):
        d = asdict(self)
        d.pop("mask")
        return d


def mask_to_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def crop_segment(image: Image.Image, mask: np.ndarray, background=(255, 255, 255)):
    """Crop the segment's bounding box, blanking pixels outside the mask."""
    box = mask_to_box(mask)
    if box is None:
        return None
    arr = np.array(image.convert("RGB"))
    out = np.empty_like(arr)
    out[:] = background
    out[mask] = arr[mask]
    x0, y0, x1, y1 = box
    return Image.fromarray(out[y0:y1, x0:x1])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ComponentRecognizer:
    def __init__(self, labels: Sequence[str], sam_model="facebook/sam-vit-base",
                 clip_model="openai/clip-vit-base-patch32", device=None,
                 min_area=500, min_score=0.0, template="a photo of a {}"):
        import torch
        from transformers import CLIPModel, CLIPProcessor, pipeline

        self.labels = list(labels)
        self.min_area, self.min_score, self.template = min_area, min_score, template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.sam = pipeline("mask-generation", model=sam_model, device=self.device)
        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(clip_model)

    def _classify(self, crops: List[Image.Image]):
        texts = [self.template.format(l) for l in self.labels]
        inputs = self.proc(text=texts, images=crops, return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            logits = self.clip(**inputs).logits_per_image.cpu().numpy()
        return [softmax(row) for row in logits]

    def recognize(self, image: Image.Image) -> List[Detection]:
        image = image.convert("RGB")
        masks = self.sam(image, points_per_batch=64)["masks"]
        segs, crops = [], []
        for m in masks:
            m = np.asarray(m, dtype=bool)
            w, h = image.size
            if m.shape == (w, h) and w != h:  # some transformers versions return (W, H)
                m = m.T
            if m.sum() < self.min_area:
                continue
            crop = crop_segment(image, m)
            if crop is not None:
                segs.append(m)
                crops.append(crop)
        if not crops:
            return []
        dets = []
        for m, probs in zip(segs, self._classify(crops)):
            i = int(probs.argmax())
            if probs[i] >= self.min_score:
                dets.append(Detection(self.labels[i], float(probs[i]), mask_to_box(m), int(m.sum()), m))
        return sorted(dets, key=lambda d: -d.score)


def annotate(image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    from PIL import ImageDraw

    out = image.convert("RGB").copy()
    overlay = np.array(out)
    rng = np.random.default_rng(0)
    for d in detections:
        overlay[d.mask] = (0.5 * overlay[d.mask] + 0.5 * rng.integers(0, 255, 3)).astype(np.uint8)
    out = Image.fromarray(overlay)
    draw = ImageDraw.Draw(out)
    for d in detections:
        draw.rectangle(d.box, outline=(255, 0, 0), width=2)
        draw.text((d.box[0] + 3, d.box[1] + 3), f"{d.label} {d.score:.2f}", fill=(255, 255, 255))
    return out
