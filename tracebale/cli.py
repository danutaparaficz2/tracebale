import argparse

from .classifiers import ClipClassifier
from .pipeline import ComponentRecognizer
from .segmenters import ColorRegionSegmenter, SamSegmenter


def main(argv=None):
    p = argparse.ArgumentParser(description="Recognise components in an image")
    p.add_argument("image")
    p.add_argument("--labels", required=True, help="comma-separated candidate component names")
    p.add_argument("--sam-checkpoint", help="SAM checkpoint; falls back to colour regions if omitted")
    p.add_argument("--sam-model-type", default="vit_b")
    p.add_argument("--min-area", type=int, default=100)
    args = p.parse_args(argv)

    segmenter = (SamSegmenter(args.sam_checkpoint, args.sam_model_type)
                 if args.sam_checkpoint else ColorRegionSegmenter())
    recognizer = ComponentRecognizer(segmenter, ClipClassifier(args.labels.split(",")), args.min_area)
    for c in recognizer.recognize(args.image):
        print(f"{c.label}\t{c.score:.2f}\tbbox={c.bbox}\tarea={c.area}")


if __name__ == "__main__":
    main()
