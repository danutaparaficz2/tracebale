# sam_clip_pipeline: Grounding DINO + SAM + CLIP

Local, zero-shot detection and segmentation of building components. No vLLM
server is needed. This was an experimental alternative to the VLM audit in
`../vllm_pipeline/`.

| File | Role |
| --- | --- |
| `pipeline.py` | `GroundedSegmenter` (DINO + SAM), `ComponentRecognizer` (SAM + CLIP), `annotate`, helpers |
| `main.py` | Batch runner using `GroundedSegmenter` for the labels `door`, `window` |
| `cli.py` | Command line for `ComponentRecognizer` (SAM + CLIP) |
| `segmenters.py`, `classifiers.py` | Lower-level SAM segmenter / CLIP classifier building blocks |

## Two variants

| Variant | How it works | Models (Hugging Face, downloaded on first use) |
| --- | --- | --- |
| **DINO + SAM** (`GroundedSegmenter`, used by `main.py`) | Grounding DINO finds boxes for the text labels; SAM turns each box into a mask | `IDEA-Research/grounding-dino-tiny`, `facebook/sam-vit-base` |
| **SAM + CLIP** (`ComponentRecognizer`, used by `cli.py`) | SAM proposes masks over the whole image; CLIP picks the best label for each masked crop ("a photo of a {label}") | `facebook/sam-vit-base`, `openai/clip-vit-base-patch32` |

Nothing is trained or fine-tuned. Results are zero-shot: SAM + CLIP has no
"neither" option, so unrelated regions can be labelled as a door or window, and
all detections need human review.

## Setup

From the repository root:

```bash
python -m pip install torch torchvision transformers pillow pillow-heif numpy
```

## Run

**DINO + SAM, batch** - every image in `01_Data/Example of Inventory/Images/` is processed (resized to 1024 px)
and saved to `output/<name>.png`:

```bash
python -m sam_clip_pipeline.main                   # all images
python -m sam_clip_pipeline.main "01_Data/Example of Inventory/Images/photo.jpg"  # one image
```

Labels and thresholds are constants at the top of `main.py` (`LABELS`,
`BOX_THRESHOLD=0.35`, `TEXT_THRESHOLD=0.25`, `MIN_AREA`, `MIN_AREA_FRACTION`).

**SAM + CLIP, single image** - prints JSON (label, score, box, area):

```bash
python -m sam_clip_pipeline.cli "01_Data/Example of Inventory/Images/photo.jpg" \
  --labels "door,window,radiator" --out annotated.png \
  --min-score 0.3 --min-area 500
```

Use `--sam-model` / `--clip-model` for other compatible Hugging Face models.

**Python:**

```python
from PIL import Image
from sam_clip_pipeline.pipeline import GroundedSegmenter, ComponentRecognizer, annotate

img = Image.open("photo.jpg")
dets = GroundedSegmenter(["door", "window"]).recognize(img)   # DINO + SAM
dets = ComponentRecognizer(["door", "window"]).recognize(img)  # SAM + CLIP
annotate(img, dets).save("annotated.png")
```

Each `Detection` has `label`, `score`, `box`, `area` and `mask`.

## Tests

```bash
python -m pip install pytest
python -m pytest tests
```

Example outputs of this pipeline on the sample photos are in
`../reports/output_of_SAM_doors_windows/`; `scripts/make_detection_report.py`
builds a Word report from them.
