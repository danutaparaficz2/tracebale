# vllm_pipeline: VLM audit, SUPSI rules and metric depth

Turns survey photographs into the TRACEABLE inventory workbook.

```text
Images/*.jpg --vLLM.py--> vllm_output/*.json --supsi_kb_2.py--> Excel inventory
                                                   ^
                              estimate_metric.py (optional, Depth Anything V2)
```

| File | Role |
| --- | --- |
| `vLLM.py` | Sends each image to a Qwen3-VL server (OpenAI-compatible API) with a Swiss eBKP-H audit prompt; saves one JSON per image |
| `supsi_kb_2.py` | Builds/uses `supsi_kb.sqlite`, matches components to WBS codes and anomalies, computes residual service life and reuse tier, exports Excel |
| `estimate_metric.py` | Estimates real-world width/height/thickness of a bounding box from a single image (Depth Anything V2 metric + Open3D) |
| `supsi_kb.sqlite` | Local SUPSI catalog (service lives, anomalies); rebuilt with `--rebuild-db` |
| `vllm_output/` | VLM JSON reports (one per image) |

Run all commands from the repository root with the virtual environment active.

## 1. Generate VLM reports (`vLLM.py`)

Start a vLLM server for `Qwen/Qwen3-VL-32B-Instruct` first (for example
`vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8000`), then:

```bash
python -m pip install openai requests pillow pillow-heif
python vllm_pipeline/vLLM.py
```

Images are read from `Images/` and reports written to `vllm_pipeline/vllm_output/`.
Valid existing reports are skipped. Settings are environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint |
| `OPENAI_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | model name |
| `IMAGE_DIR` / `VLLM_OUTPUT_DIR` | `Images/` / `vllm_output/` | input / output folders |
| `MAX_IMAGE_SIDE` | `1568` | long side images are downscaled to |
| `OPENAI_MAX_TOKENS` | `4000` | response limit |
| `OPENAI_REQUEST_TIMEOUT` | `300` | seconds per request |
| `OPENAI_SKIP_EXISTING` | `1` | `0` re-runs images that already have a report |
| `OPENAI_START_INDEX` | `1` | one-based image index to start from |
| `OPENAI_SKIP_ENDPOINT_CHECK` | `0` | `1` skips the server health check |

Example: `OPENAI_START_INDEX=9 OPENAI_SKIP_EXISTING=0 python vllm_pipeline/vLLM.py`.

Each component in the JSON has the eBKP-H classification, name, geometry
estimate, aesthetics/functionality/pollutant condition, connection type,
disassembly difficulty, anomalies and `bounding_box_2d`. The prompt asks for
`[ymin, xmin, ymax, xmax]`, but Qwen3-VL actually returns `[x1, y1, x2, y2]` on a
0-1000 scale; see `scripts/plot_bboxes.py`.

## 2. Export the inventory (`supsi_kb_2.py`)

```bash
python -m pip install pandas openpyxl
```

**Without depth** (geometry = the VLM's text estimate):

```bash
python vllm_pipeline/supsi_kb_2.py --export-excel TRACEABLE_Inventory_new.xlsx
```

**With Depth Anything V2** (metric size written to `Geometry`, suffixed
`measured via Metric Depth`):

```bash
python -m pip install open3d torch numpy
export PYTHONPATH="$PWD/../Depth-Anything-V2/metric_depth:$PYTHONPATH"

python vllm_pipeline/supsi_kb_2.py \
  --image-dir Images \
  --depth-weights /path/to/depth_anything_v2_metric_hypersim_vitl.pth \
  --export-excel TRACEABLE_Inventory_MetricDepth.xlsx
```

Options: `--vllm-dir` (default `vllm_pipeline/vllm_output`), `--image-dir`,
`--depth-weights`, `--export-excel`, `--rebuild-db`. Depth is used only when both
`--image-dir` and an existing `--depth-weights` file are given; any failure falls
back to the VLM geometry for that component. Metric values are single-image
estimates (65 degree field-of-view assumed without EXIF focal length), not survey-grade.

Measure one object directly:

```bash
python vllm_pipeline/estimate_metric.py --image Images/<photo>.jpg \
  --bbox YMIN XMIN YMAX XMAX --weights /path/to/depth_anything_v2_metric_hypersim_vitl.pth
```

(`--bbox` is normalized `[ymin xmin ymax xmax]`, 0-1000.)

## Visualising and reporting

Use `scripts/plot_bboxes.py` to draw the boxes of an inventory workbook on the
images, and `scripts/make_report.py` to build a Word report of images with their
VLM descriptions. Generated files go to `reports/`.
