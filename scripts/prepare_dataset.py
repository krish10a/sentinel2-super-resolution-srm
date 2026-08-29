"""
Fast Multi-Threaded Dataset Pre-fetcher for SEN2NAIPv2.
Fetches real 10m LR and 2.5m HR 4-band pairs directly from Hugging Face
using HTTP Range Requests with concurrent worker threads.
Zero rasterio / GDAL network roundtrip overhead.
"""

import os
import sys
import time
import io
import json
import argparse
import requests
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio as rio
import tacoreader.v1 as tacoreader
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import write_geotiff

BASE_URL = "https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2/resolve/main/sen2naipv2-unet.0000.part.taco"


def decode_tortilla_bytes(sample_bytes: bytes):
    """Parses raw Tortilla container bytes into LR and HR numpy arrays."""
    FO = int.from_bytes(sample_bytes[2:10], "little")
    FL = int.from_bytes(sample_bytes[10:18], "little")

    footer_bytes = sample_bytes[FO : FO + FL]
    table = pq.read_table(pa.BufferReader(footer_bytes)).to_pandas()

    sub_off_lr = int(table.iloc[0]["tortilla:offset"])
    sub_len_lr = int(table.iloc[0]["tortilla:length"])
    sub_off_hr = int(table.iloc[1]["tortilla:offset"])
    sub_len_hr = int(table.iloc[1]["tortilla:length"])

    lr_raw_bytes = sample_bytes[sub_off_lr : sub_off_lr + sub_len_lr]
    hr_raw_bytes = sample_bytes[sub_off_hr : sub_off_hr + sub_len_hr]

    with rio.open(io.BytesIO(lr_raw_bytes)) as s_lr, rio.open(io.BytesIO(hr_raw_bytes)) as s_hr:
        lr_arr = s_lr.read().astype(np.float32)  # [4, 130, 130]
        hr_arr = s_hr.read().astype(np.float32)  # [4, 520, 520]
        meta_lr = {'crs': str(s_lr.crs), 'transform': s_lr.transform}
        meta_hr = {'crs': str(s_hr.crs), 'transform': s_hr.transform}

    return lr_arr, hr_arr, meta_lr, meta_hr


def fetch_single_sample(item_tuple, session):
    """Worker task to fetch one sample byte range."""
    idx, offset, length, tortilla_id, roi_id, sample_dir = item_tuple
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    
    resp = session.get(BASE_URL, headers=headers, timeout=30)
    if resp.status_code not in (200, 206):
        raise ValueError(f"HTTP error {resp.status_code} for sample {idx}")

    lr_arr, hr_arr, meta_lr, meta_hr = decode_tortilla_bytes(resp.content)

    os.makedirs(sample_dir, exist_ok=True)
    
    # Save local GeoTIFFs
    lr_path = os.path.join(sample_dir, "lr.tif")
    hr_path = os.path.join(sample_dir, "hr.tif")
    write_geotiff(lr_path, lr_arr, meta=meta_lr)
    write_geotiff(hr_path, hr_arr, meta=meta_hr)

    # Metadata
    meta_info = {
        "sample_id": f"sample_{idx + 1:05d}",
        "original_id": tortilla_id,
        "roi_id": roi_id,
        "crs": str(meta_hr['crs']),
        "lr_shape": list(lr_arr.shape),
        "hr_shape": list(hr_arr.shape),
        "scale_factor": 4,
        "bands": ["B04", "B03", "B02", "B08"]
    }
    with open(os.path.join(sample_dir, "meta.json"), 'w') as f:
        json.dump(meta_info, f, indent=2)

    return idx


def prepare_real_dataset(output_dir: str = "data/processed",
                         target_samples: int = 300,
                         max_workers: int = 16):
    """
    Downloads target real samples concurrently using thread pool.
    """
    os.makedirs(output_dir, exist_ok=True)
    index_cache_file = "data/index_sen2naipv2_unet.parquet"

    print("[Dataset] Loading SEN2NAIPv2 dataset index...", flush=True)
    if os.path.exists(index_cache_file):
        import pandas as pd
        dataset = pd.read_parquet(index_cache_file)
        print(f"[Dataset] Loaded cached index from {index_cache_file} ({len(dataset):,} samples)", flush=True)
    else:
        dataset = tacoreader.load("tacofoundation:sen2naipv2-unet")
        # Cache index for instant loading next time
        os.makedirs(os.path.dirname(index_cache_file), exist_ok=True)
        dataset.to_parquet(index_cache_file)
        print(f"[Dataset] Cached dataset index to {index_cache_file}", flush=True)

    tasks = []
    for idx in range(min(target_samples, len(dataset))):
        row = dataset.iloc[idx]
        offset = int(row["tortilla:offset"])
        length = int(row["tortilla:length"])
        tortilla_id = str(row.get("tortilla:id", f"sample_{idx}"))
        
        if "__" in tortilla_id:
            roi_id = tortilla_id.split("__")[0]
        else:
            roi_id = f"ROI_{idx // 50:03d}"

        sample_dir = os.path.join(output_dir, f"sample_{idx + 1:05d}")
        tasks.append((idx, offset, length, tortilla_id, roi_id, sample_dir))

    print(f"\n[Dataset] Downloading {len(tasks)} real samples using {max_workers} concurrent worker threads...", flush=True)
    t0 = time.time()

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
    session.mount("https://", adapter)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_sample, task, session): task[0] for task in tasks}
        pbar = tqdm(total=len(tasks), desc="Fetching Real SEN2NAIP Samples")
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            pbar.update(1)
        pbar.close()

    elapsed = time.time() - t0
    print("\n" + "=" * 60, flush=True)
    print(f"[Dataset] SUCCESS: {completed} real samples downloaded in {elapsed:.2f}s ({completed / elapsed:.1f} samples/sec)!", flush=True)
    print("=" * 60 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fast parallel real SEN2NAIPv2 dataset fetcher")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--samples", type=int, default=300, help="Number of real samples (e.g. 200 train + 50 val + 50 test = 300)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent download threads")
    args = parser.parse_args()

    prepare_real_dataset(output_dir=args.output_dir, target_samples=args.samples, max_workers=args.workers)


if __name__ == "__main__":
    main()
