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
    p.add_argument("--output", help="save the image with masks, boxes and labels drawn on it")
    args = p.parse_args(argv)

    segmenter = (SamSegmenter(args.sam_checkpoint, args.sam_model_type)
                 if args.sam_checkpoint else ColorRegionSegmenter())
    recognizer = ComponentRecognizer(segmenter, ClipClassifier(args.labels.split(",")), args.min_area)
    components = recognizer.recognize(args.image)
    if args.output:
        from .visualize import overlay

        overlay(args.image, components).save(args.output)
    for c in components:
        print(f"{c.label}\t{c.score:.2f}\tbbox={c.bbox}\tarea={c.area}")


if __name__ == "__main__":
    main()
