# tracebale
Analyses an image and recognises which component each region is. The current
public pipeline does not train or fine-tune a model on the supplied image or
component labels.

## What the code does

The default `ComponentRecognizer` pipeline is:

1. It converts the input image to RGB.
2. **Segment Anything (SAM)** generates candidate masks automatically. The
   default model is `facebook/sam-vit-base`, which is the **SAM ViT-B
   (Vision Transformer Base)** model. The Transformers mask-generation
   pipeline prompts SAM over an image-point grid; no user-drawn boxes or
   points are required.
3. Masks smaller than `min_area` are discarded (the default is 500 pixels).
   For every remaining mask, the code crops its bounding box and replaces
   pixels outside the mask with white.
4. **CLIP** compares each masked crop with text prompts made from the labels:
   `a photo of a {label}`. The default model is
   `openai/clip-vit-base-patch32`, a **CLIP ViT-B/32** image-and-text model.
   The label with the highest image-text similarity is selected. The
   similarities are converted to a softmax score; the CLI keeps detections
   with a score of at least 0.3 by default.
5. The detections are sorted by descending score and returned as a label,
   score, bounding box, area, and mask. The JSON CLI output omits the mask;
   `--out` writes the original image with coloured masks, red boxes, and
   labels overlaid.

This is zero-shot recognition: the component names supplied by the user are
candidate text labels, not training examples. The code does not claim that a
detected component is present with calibrated probability, and it does not
fine-tune either model for circuit boards or electronic components.

## Models and pretrained weights

Both default models are loaded with `from_pretrained` through Hugging Face
Transformers. Their weights and processors are downloaded and cached on the
first run; they are not trained by this project.

### SAM ViT-B

- Model ID: `facebook/sam-vit-base`
- Architecture: SAM with a ViT-B image encoder, plus SAM's prompt encoder and
  mask decoder.
- Pretraining: the Segment Anything model was trained using the **SA-1B
  dataset**, consisting of approximately **11 million licensed, privacy-
  respecting images and 1.1 billion masks**.
- In this project it is used for automatic, zero-shot mask generation, not for
  component classification.

### CLIP ViT-B/32

- Model ID: `openai/clip-vit-base-patch32`
- Architecture: a ViT-B/32 image encoder and a Transformer text encoder.
- Pretraining: OpenAI trained CLIP with contrastive learning on publicly
  available image-caption data collected from the internet and existing
  datasets, including YFCC100M. The original CLIP release describes the
  training set as approximately **400 million image-text pairs** (often called
  WebImageText); the complete dataset was not released.
- In this project it is used without fine-tuning to rank the supplied English
  component names.

The repository also contains `models/sam_vit_b.pth`, but the current
`ComponentRecognizer` does **not** load that file. It uses the Hugging Face
model ID above. The lower-level `SamSegmenter` class can load a checkpoint
path explicitly, but that class is not used by the default CLI or the public
`ComponentRecognizer` pipeline.

```
pip install -e .
tracebale board.jpg --labels "resistor,capacitor,microchip,connector" --out annotated.png
```

Output is JSON (label, score, box, area per segment); `--out` saves an annotated image.
Use `--sam-model` and `--clip-model` to select different compatible Hugging Face
models.

Python:
```python
from PIL import Image
from tracebale import ComponentRecognizer
rec = ComponentRecognizer(["resistor", "capacitor"])
dets = rec.recognize(Image.open("board.jpg"))
```

Tests: `pip install pytest numpy pillow && pytest` (pure-logic tests need no models).
