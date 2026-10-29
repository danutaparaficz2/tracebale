"""Plot Qwen3 bounding boxes from the inventory Excel onto their images and display them."""
import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[1]
XLSX = BASE / "reports" / "TRACEABLE_Inventory_Qwen3.xlsx"
IMAGE_DIR = BASE / "01_Data" / "Example of Inventory" / "Images"
EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def find_image(code, image_dir):
    for ext in EXTS:
        p = image_dir / f"{code}{ext}"
        if p.exists():
            return p
    return None


def parse_box(value):
    try:
        box = ast.literal_eval(str(value))
        if len(box) == 4:
            return [float(v) for v in box]
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=XLSX)
    ap.add_argument("--images", type=Path, default=IMAGE_DIR)
    ap.add_argument("--save-dir", type=Path, default=None, help="Also save annotated images here")
    ap.add_argument("--absolute", action="store_true",
                    help="Boxes are in pixels instead of Qwen3's 0-1000 normalised coordinates")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for code, group in df.groupby("code images", sort=False):
        path = find_image(str(code), args.images)
        if path is None:
            print(f"Image not found for {code}")
            continue
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        w, h = img.size

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        cmap = plt.get_cmap("tab10")
        for i, (_, row) in enumerate(group.iterrows()):
            box = parse_box(row.get("Bounding Box"))
            if box is None:
                continue
            x1, y1, x2, y2 = box  # Qwen3 order: [x1, y1, x2, y2]
            if not args.absolute:
                x1, x2 = x1 / 1000 * w, x2 / 1000 * w
                y1, y2 = y1 / 1000 * h, y2 / 1000 * h
            color = cmap(i % 10)
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor=color, linewidth=3))
            label = str(row.get("Third level", "") or row.get("Component name", ""))
            ax.text(x1, max(y1 - 5, 0), f"{i + 1}. {label}", color="white", fontsize=9,
                    bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))
        ax.set_title(f"{code} ({len(group)} components)")
        ax.axis("off")
        plt.tight_layout()
        if args.save_dir:
            fig.savefig(args.save_dir / f"{code}_bbox.png", dpi=120)
        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
