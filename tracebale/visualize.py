import numpy as np
from PIL import Image, ImageDraw


def overlay(image, components, alpha: float = 0.5) -> Image.Image:
    """Return the image with each component's mask tinted and its bbox + label drawn."""
    if not isinstance(image, np.ndarray):
        image = np.asarray(Image.open(image).convert("RGB"))
    rng = np.random.default_rng(0)
    out = image.astype(np.float32)
    colors = []
    for c in components:
        col = rng.integers(40, 255, 3)
        colors.append(tuple(int(v) for v in col))
        out[c.mask] = (1 - alpha) * out[c.mask] + alpha * col
    img = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for c, col in zip(components, colors):
        draw.rectangle(c.bbox, outline=col, width=2)
        draw.text((c.bbox[0] + 3, c.bbox[1] + 3), f"{c.label} {c.score:.2f}", fill=(255, 255, 255))
    return img
