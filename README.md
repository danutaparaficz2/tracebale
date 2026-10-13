# tracebale
Analyses an image and recognises which component each region is.

Pipeline: **Segment Anything (SAM)** proposes masks → each masked crop is classified
zero-shot with **CLIP** against a list of component names you provide.

```
pip install -e .
tracebale board.jpg --labels "resistor,capacitor,microchip,connector" --out annotated.png
```

Output is JSON (label, score, box, area per segment); `--out` saves an annotated image.
Models download from Hugging Face on first run (`--sam-model`, `--clip-model` to change).

Python:
```python
from PIL import Image
from tracebale import ComponentRecognizer
rec = ComponentRecognizer(["resistor", "capacitor"])
dets = rec.recognize(Image.open("board.jpg"))
```

Tests: `pip install pytest numpy pillow && pytest` (pure-logic tests need no models).
