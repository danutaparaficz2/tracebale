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


def crop_segment(image: Image.Image, mask: np.ndarray, background=(255, 255, 255),
                 margin=0.0, keep_context=False):
    """Crop the segment's bounding box (plus a relative margin).

    By default pixels outside the mask are blanked; with keep_context=True the
    surrounding image is kept so the classifier can see frames and neighbours.
    """
    box = mask_to_box(mask)
    if box is None:
        return None
    arr = np.array(image.convert("RGB"))
    if keep_context:
        out = arr
    else:
        out = np.empty_like(arr)
        out[:] = background
        out[mask] = arr[mask]
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(arr.shape[1], x1 + mx), min(arr.shape[0], y1 + my)
    return Image.fromarray(out[y0:y1, x0:x1])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ComponentRecognizer:
    def __init__(self, labels: Sequence[str], sam_model="facebook/sam-vit-base",
                 clip_model="openai/clip-vit-base-patch32", device=None,
                 min_area=500, min_score=0.0, template="a photo of a {}",
                 min_area_fraction=0.0, targets: Sequence[str] = None,
                 crop_margin=0.0, keep_context=False):
        import torch
        from transformers import CLIPModel, CLIPProcessor, pipeline

        self.labels = list(labels)
        self.min_area, self.min_score, self.template = min_area, min_score, template
        self.min_area_fraction = min_area_fraction
        # labels outside `targets` act as negative classes and are discarded
        self.targets = set(targets) if targets is not None else set(self.labels)
        self.crop_margin, self.keep_context = crop_margin, keep_context
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
        w, h = image.size
        min_pixels = max(self.min_area, self.min_area_fraction * w * h)
        for m in masks:
            m = np.asarray(m, dtype=bool)
            if m.shape == (w, h) and w != h:  # some transformers versions return (W, H)
                m = m.T
            if m.sum() < min_pixels:
                continue
            crop = crop_segment(image, m, margin=self.crop_margin, keep_context=self.keep_context)
            if crop is not None:
                segs.append(m)
                crops.append(crop)
        if not crops:
            return []
        dets = []
        for m, probs in zip(segs, self._classify(crops)):
            i = int(probs.argmax())
            if self.labels[i] in self.targets and probs[i] >= self.min_score:
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


class GroundedSegmenter:
    """Detect first (Grounding DINO, text prompts), then segment each box with SAM."""

    def __init__(self, labels: Sequence[str], detector_model="IDEA-Research/grounding-dino-tiny",
                 sam_model="facebook/sam-vit-base", device=None,
                 box_threshold=0.35, text_threshold=0.25, min_area=500, min_area_fraction=0.0):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  SamModel, SamProcessor)

        self.labels = [l.lower() for l in labels]
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.min_area, self.min_area_fraction = min_area, min_area_fraction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.det_proc = AutoProcessor.from_pretrained(detector_model)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(detector_model).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_model)
        self.sam = SamModel.from_pretrained(sam_model).to(self.device).eval()

    def _match_label(self, text: str):
        for l in self.labels:
            if l in text.lower():
                return l
        return None

    def _detect(self, image: Image.Image):
        prompt = ". ".join(self.labels) + "."
        inputs = self.det_proc(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.det(**inputs)
        res = self.det_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[image.size[::-1]],
        )[0]
        found = []
        labels = res.get("text_labels", res.get("labels"))
        for box, score, text in zip(res["boxes"], res["scores"], labels):
            label = self._match_label(str(text))
            if label:
                found.append((label, float(score), [float(v) for v in box]))
        return found

    def _segment(self, image: Image.Image, box):
        inputs = self.sam_proc(image, input_boxes=[[box]], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.sam(**inputs, multimask_output=True)
        masks = self.sam_proc.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]  # (num_masks, H, W)
        best = int(outputs.iou_scores.cpu()[0, 0].argmax())
        return np.asarray(masks[best], dtype=bool)

    def recognize(self, image: Image.Image) -> List[Detection]:
        image = image.convert("RGB")
        w, h = image.size
        min_pixels = max(self.min_area, self.min_area_fraction * w * h)
        dets = []
        for label, score, box in self._detect(image):
            mask = self._segment(image, box)
            area = int(mask.sum())
            if area < min_pixels:
                continue
            dets.append(Detection(label, score, mask_to_box(mask), area, mask))
        return sorted(dets, key=lambda d: -d.score)
