"""
Extracts a full 4-band real Sentinel-2 L2A scene (10m) from the dataset
and saves it as data/real_s2/sentinel2_l2a_scene_10m.tif for inference demonstration.
"""

import os
import sys
import json
import rasterio as rio
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import read_geotiff, write_geotiff


def prepare_sample_scene(output_path: str = "data/real_s2/sentinel2_l2a_scene_10m.tif"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Use sample from test set
    with open("data/processed/splits.json", 'r') as f:
        splits = json.load(f)

    test_samples = splits["test"]
    sample_dir = test_samples[0]
    lr_file = os.path.join(sample_dir, "lr.tif")

    raw_data, meta = read_geotiff(lr_file)
    print(f"[Prep Scene] Extracted real Sentinel-2 L2A tile from {sample_dir}")
    print(f"  Shape: {raw_data.shape}, CRS: {meta.get('crs')}, Res: {meta.get('res')}")

    # Write out as dedicated real Sentinel-2 scene
    write_geotiff(output_path, raw_data[0:4], meta=meta, upscale_factor=1.0)
    print(f"[Prep Scene] Saved real Sentinel-2 scene to: {output_path}")


if __name__ == "__main__":
    prepare_sample_scene()
