"""
estimate_metric.py - Metric Dimension Estimation for TRACEABLE
Extracts true physical Height, Width, Depth (m) from RGB images + VLM Bounding Boxes.
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image, ExifTags

# Import model definition from Depth-Anything-V2 repo
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    raise ImportError("Please ensure Depth-Anything-V2 is in your PYTHONPATH or local folder.")


def load_metric_depth_model(
    checkpoint_path: str = "depth_anything_v2_metric_hypersim_vitl.pth",
    encoder: str = "vitl",
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
):
    """Loads the Depth Anything V2 Metric model weights."""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    # max_depth=20.0 meters for indoor scenes
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 20.0})
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device).eval()
    return model


def get_focal_length_px(image: Image.Image) -> float:
    """Extracts focal length from EXIF metadata, or falls back to standard 65 deg FOV."""
    w, h = image.size
    exif = image.getexif()
    
    # Try EXIF 35mm equivalent focal length (Tag 41989) or standard FocalLength (Tag 37386)
    if exif:
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == "FocalLengthIn35mmFilm" and value:
                # 35mm film sensor width is 36mm
                return float(value) * (w / 36.0)
    
    # Robust fallback: standard smartphone ~65 degree horizontal field of view
    return 0.5 * w / np.tan(np.radians(65.0 / 2.0))


def estimate_component_size(
    image_path: str,
    bbox_1000: list[int],  # [ymin, xmin, ymax, xmax] in 0-1000 normalized space
    depth_model
) -> dict:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    focal_length = get_focal_length_px(image)
    cx, cy = w / 2.0, h / 2.0

    # 1. Convert 0-1000 normalized box to pixel coordinates
    ymin, xmin, ymax, xmax = bbox_1000
    x0, y0 = int(xmin * w / 1000), int(ymin * h / 1000)
    x1, y1 = int(xmax * w / 1000), int(ymax * h / 1000)

    # 2. Predict metric depth map (meters)
    with torch.no_grad():
        depth_map = depth_model.infer_image(np.array(image))

    # 3. Create component region mask (substitute with SAM mask if loaded)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True

    # 4. Filter depth values inside mask
    ys, xs = np.where(mask)
    zs = depth_map[ys, xs]

    # Outlier rejection: discard bottom/top 5% to remove wall bleed / foreground clutter
    z_min, z_max = np.percentile(zs, 5), np.percentile(zs, 95)
    keep = (zs >= z_min) & (zs <= z_max) & (zs > 0.3)
    xs, ys, zs = xs[keep], ys[keep], zs[keep]

    # 5. Pinhole Back-projection to 3D point cloud
    X = (xs - cx) * zs / focal_length
    Y = (ys - cy) * zs / focal_length
    Z = zs
    points_3d = np.vstack([X, Y, Z]).T

    # 6. Fit 3D Oriented Bounding Box (OBB) using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Statistical Outlier Removal
    clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    obb = clean_pcd.get_oriented_bounding_box()

    # Sort extents so: [thickness, width, height]
    extents = np.sort(obb.extent)
    thickness, width, height = extents[0], extents[1], extents[2]

    return {
        "width_m": round(float(width), 3),
        "height_m": round(float(height), 3),
        "thickness_m": round(float(thickness), 3),
        "area_m2": round(float(width * height), 3),
        "median_distance_m": round(float(np.median(zs)), 2)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate real-world dimensions from image and bounding box.")
    parser.add_argument("--image", required=True, help="Path to survey photo")
    parser.add_argument("--bbox", nargs=4, type=int, default=[312, 6, 825, 968],
                        help="Normalized bbox [ymin xmin ymax xmax] (0-1000 scale)")
    parser.add_argument("--weights", default="depth_anything_v2_metric_hypersim_vitl.pth",
                        help="Path to Depth Anything V2 Metric weights")
    args = parser.parse_args()

    print(f"Loading metric depth model from {args.weights}...")
    model = load_metric_depth_model(checkpoint_path=args.weights)

    print(f"Estimating dimensions for: {args.image}")
    results = estimate_component_size(args.image, args.bbox, model)

    print("\n--- MEASURED METRIC DIMENSIONS ---")
    print(f"Width:           {results['width_m']} m")
    print(f"Height:          {results['height_m']} m")
    print(f"Thickness:       {results['thickness_m']} m")
    print(f"Frontal Area:    {results['area_m2']} m²")
    print(f"Distance to Cam: {results['median_distance_m']} m")