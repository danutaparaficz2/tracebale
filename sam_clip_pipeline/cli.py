import argparse
import json

from PIL import Image


def main(argv=None):
    p = argparse.ArgumentParser(description="Identify components in an image.")
    p.add_argument("image")
    p.add_argument("--labels", required=True, help="Comma-separated component names, e.g. 'resistor,capacitor,chip'")
    p.add_argument("--out", help="Path for annotated image")
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--min-area", type=int, default=500)
    p.add_argument("--sam-model", default="facebook/sam-vit-base")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    a = p.parse_args(argv)

    from .pipeline import ComponentRecognizer, annotate

    labels = [l.strip() for l in a.labels.split(",") if l.strip()]
    rec = ComponentRecognizer(labels, a.sam_model, a.clip_model, min_area=a.min_area, min_score=a.min_score)
    img = Image.open(a.image)
    dets = rec.recognize(img)
    print(json.dumps([d.to_dict() for d in dets], indent=2))
    if a.out:
        annotate(img, dets).save(a.out)


if __name__ == "__main__":
    main()
