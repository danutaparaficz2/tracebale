import numpy as np

from tracebale import ColorRegionSegmenter, ComponentRecognizer, ReferenceClassifier


def test_recognizes_colored_components():
    img = np.full((40, 60, 3), 255, np.uint8)
    img[5:15, 5:20] = (255, 0, 0)
    img[20:35, 30:50] = (0, 0, 255)
    clf = ReferenceClassifier({"resistor": [(250, 0, 0)], "capacitor": [(0, 0, 250)],
                               "board": [(255, 255, 255)]})
    comps = ComponentRecognizer(ColorRegionSegmenter(), clf).recognize(img)
    found = {c.label: c.bbox for c in comps}
    assert found["resistor"] == (5, 5, 20, 15)
    assert found["capacitor"] == (30, 20, 50, 35)
    assert "board" in found
