import numpy as np
from PIL import Image

from tracebale.pipeline import crop_segment, mask_to_box, softmax


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
