"""Run the whole pipeline with default parameters: python main.py [image]"""
import sys
from pathlib import Path

from PIL import Image

from tracebale import ComponentRecognizer
from tracebale.pipeline import annotate

ROOT = Path(__file__).parent
LABELS = ["door handle", "lock plate", "label sticker", "graffiti", "door", "floor", "wall",
          "ceiling", "window", "door frame"]
MIN_AREA = 2000
MIN_SCORE = 0.3
OUTPUT = ROOT / "output" / "result.png"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "Images").glob("*.jpg"))[0]
    image = Image.open(path)
    image.thumbnail((1024, 1024))  # keeps CPU runtime reasonable
    detections = ComponentRecognizer(LABELS, min_area=MIN_AREA, min_score=MIN_SCORE).recognize(image)
    for d in detections:
        print(f"{d.label}\t{d.score:.2f}\tbox={d.box}\tarea={d.area}")
    OUTPUT.parent.mkdir(exist_ok=True)
    annotate(image, detections).save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
