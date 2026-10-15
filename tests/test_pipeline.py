from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from tracebale import ComponentRecognizer
from tracebale.pipeline import annotate, crop_segment, mask_to_box, softmax


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
LABELS = [
    "door handle",
    "lock plate",
    "label sticker",
    "graffiti",
    "door",
    "floor",
    "wall",
    "ceiling",
    "window",
    "door frame",
]
VISUALIZATION_OUTPUT = PROJECT_ROOT / "output" / "test_pipeline.png"


def test_pipeline_on_first_image():
    images = sorted(
        path for path in (PROJECT_ROOT / "Images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    assert images, "Images directory must contain at least one supported image"

    with Image.open(images[0]) as image:
        image.thumbnail((1024, 1024))
        image_size = image.size
        detections = ComponentRecognizer(
            LABELS, min_area=2000, min_score=0.3
        ).recognize(image)
        annotated = annotate(image, detections)
        VISUALIZATION_OUTPUT.parent.mkdir(exist_ok=True)
        annotated.save(VISUALIZATION_OUTPUT)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(VISUALIZATION_OUTPUT)])
        else:
            annotated.show()

    assert annotated.size == image_size
    for detection in detections:
        assert detection.label in LABELS
        assert 0.0 <= detection.score <= 1.0
        assert detection.area > 0


def test_mask_to_box():
    m = np.zeros((10, 10), bool)
    m[2:5, 3:8] = True
    assert mask_to_box(m) == (3, 2, 8, 5)
    assert mask_to_box(np.zeros((4, 4), bool)) is None


def test_crop_segment_blanks_outside():
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    m = np.zeros((10, 10), bool)
    m[2:4, 2:4] = True
    m[3, 3] = False
    c = crop_segment(img, m)
    assert c.size == (2, 2)
    assert c.getpixel((1, 1)) == (255, 255, 255)
    assert c.getpixel((0, 0)) == (0, 0, 0)


def test_softmax():
    assert abs(softmax(np.array([1.0, 2.0, 3.0])).sum() - 1) < 1e-9
