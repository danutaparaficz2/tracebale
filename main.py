"""Run the whole pipeline with default parameters: python main.py [image]"""
import sys
from pathlib import Path

from tracebale import ClipClassifier, ComponentRecognizer, SamSegmenter
from tracebale.visualize import overlay

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "models" / "sam_vit_b.pth"
LABELS = ["door handle", "lock plate", "label sticker", "graffiti", "door", "floor", "wall",
          "ceiling", "window", "door frame"]
MIN_AREA = 2000
OUTPUT = ROOT / "output" / "result.png"


def main():
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "Images").glob("*.jpg"))[0]
    recognizer = ComponentRecognizer(SamSegmenter(str(CHECKPOINT)), ClipClassifier(LABELS), MIN_AREA)
    components = recognizer.recognize(image)
    for c in components:
        print(f"{c.label}\t{c.score:.2f}\tbbox={c.bbox}\tarea={c.area}")
    OUTPUT.parent.mkdir(exist_ok=True)
    overlay(image, components).save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
