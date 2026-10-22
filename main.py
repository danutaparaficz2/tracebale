"""Run the whole pipeline: python main.py [image]"""
import sys
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from tracebale import ComponentRecognizer
from tracebale.pipeline import annotate

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "Images"
IMAGE_SUFFIXES = {
    ".bmp", ".gif", ".heic", ".heif", ".jpeg", ".jpg", ".mpo", ".png",
    ".tif", ".tiff", ".webp",
}
LABELS = ["door", "window"]
MIN_AREA = 2000
MIN_SCORE = 0.3
OUTPUT_DIR = ROOT / "output"


def _find_images(directory=IMAGE_DIR):
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {directory}")
    return paths


def _output_path(image_path):
    return OUTPUT_DIR / f"{image_path.stem}.png"


def main():
    register_heif_opener()
    single_image = len(sys.argv) > 1
    paths = [Path(sys.argv[1])] if single_image else _find_images()

    recognizer = ComponentRecognizer(LABELS, min_area=MIN_AREA, min_score=MIN_SCORE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        print(f"Processing {path}", flush=True)
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((1024, 1024))  # keeps CPU runtime reasonable
        detections = recognizer.recognize(image)
        for d in detections:
            print(f"{d.label}\t{d.score:.2f}\tbox={d.box}\tarea={d.area}")
        output = _output_path(path)
        annotate(image, detections).save(output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
