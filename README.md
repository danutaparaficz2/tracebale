# tracebale
Analyses images with Segment Anything (or a lightweight fallback) to recognise which components appear in a picture.

Pipeline: **segment** the image into masks, then **classify** each cropped mask.

- Segmenters: `SamSegmenter` (extra `sam`, needs a SAM checkpoint), `ColorRegionSegmenter` (no ML deps).
- Classifiers: `ClipClassifier` (extra `clip`, zero-shot from your label list), `ReferenceClassifier` (mean-colour nearest reference).

```python
from tracebale import SamSegmenter, ClipClassifier, ComponentRecognizer
rec = ComponentRecognizer(SamSegmenter("sam_vit_b.pth"), ClipClassifier(["resistor", "capacitor", "chip"]), min_area=100)
for c in rec.recognize("board.jpg"):
    print(c.label, c.score, c.bbox)
```

CLI: `tracebale board.jpg --labels resistor,capacitor,chip --sam-checkpoint sam_vit_b.pth`

Tests: `pip install -e .[test] && pytest`
