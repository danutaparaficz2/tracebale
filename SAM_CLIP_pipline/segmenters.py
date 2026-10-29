"""Segmenters turn an RGB image (H x W x 3 uint8 array) into boolean masks."""
from collections import deque
from typing import List

import numpy as np


class ColorRegionSegmenter:
    """Dependency-light fallback: connected regions of similar (quantised) colour."""

    def __init__(self, levels: int = 8, min_area: int = 50):
        self.levels = levels
        self.min_area = min_area

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        step = 256 // self.levels
        q = (image // step).astype(np.int32)
        key = (q[..., 0] * self.levels + q[..., 1]) * self.levels + q[..., 2]
        h, w = key.shape
        seen = np.zeros((h, w), bool)
        masks = []
        for sy in range(h):
            for sx in range(w):
                if seen[sy, sx]:
                    continue
                k = key[sy, sx]
                seen[sy, sx] = True
                queue, pixels = deque([(sy, sx)]), [(sy, sx)]
                while queue:
                    y, x = queue.popleft()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and key[ny, nx] == k:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                            pixels.append((ny, nx))
                if len(pixels) >= self.min_area:
                    mask = np.zeros((h, w), bool)
                    ys, xs = zip(*pixels)
                    mask[list(ys), list(xs)] = True
                    masks.append(mask)
        return masks


class SamSegmenter:
    """Segment Anything automatic mask generator (needs the `sam` extra)."""

    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cpu"):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self._generator = SamAutomaticMaskGenerator(sam)

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        results = sorted(self._generator.generate(image), key=lambda r: -r["area"])
        return [r["segmentation"].astype(bool) for r in results]
