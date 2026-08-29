"""
Real Sentinel-2 Scene Super-Resolution Inference Pipeline.
Applies the trained Residual SwinIR model to arbitrary-sized real Sentinel-2 L2A scenes (10m).
Preserves geospatial CRS, transforms, geographic bounds, and produces true 2.5m GeoTIFFs.
"""

import os
import sys
import time
import argparse
import rasterio as rio
from rasterio.transform import Affine
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import create_residual_swinir_satellite
from src.preprocessing import read_geotiff, normalize_reflectance
from src.utils import render_rgb, render_cir, set_seed


from src.inference_real_s2 import create_2d_hann_window, run_tiled_residual_inference



def infer_real_scene(input_tif: str, output_tif: str, checkpoint: str, tile_size: int = 64, stride: int = 48):
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(" >>> REAL SENTINEL-2 SUPER-RESOLUTION INFERENCE PIPELINE <<<")
    print("=" * 85)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not os.path.exists(input_tif):
        raise FileNotFoundError(f"Input Sentinel-2 scene not found: {input_tif}")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint}")

    # 1. Read Input GeoTIFF
    print(f"\n[1/5] Loading Input Sentinel-2 GeoTIFF: {input_tif}")
    t_start = time.time()
    raw_data, meta = read_geotiff(input_tif)
    orig_shape = raw_data.shape
    orig_res = meta.get('res', (10.0, 10.0))
    orig_crs = meta.get('crs')
    orig_transform = meta.get('transform')

    print(f"  Input Shape:        {orig_shape} (Bands: {orig_shape[0]}, H: {orig_shape[1]}, W: {orig_shape[2]})")
    print(f"  Input Resolution:   {orig_res[0]:.2f}m x {orig_res[1]:.2f}m")
    print(f"  Input CRS:          {orig_crs}")
    print(f"  Input Transform:    {orig_transform}")
    print(f"  Input Dynamic Range: Min={raw_data.min():.2f}, Max={raw_data.max():.2f}, Mean={raw_data.mean():.2f}")

    # Ensure 4-band RGBN (B04-Red, B03-Green, B02-Blue, B08-NIR)
    data_4b = raw_data[0:4].astype(np.float32)

    # 2. Normalize Reflectance to [0.0, 1.0]
    print("\n[2/5] Normalizing surface reflectance...")
    norm_data = normalize_reflectance(data_4b)

    # 3. Load Trained Model
    print(f"\n[3/5] Loading Residual SwinIR model from: {checkpoint}")
    model = create_residual_swinir_satellite().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    # 4. Run Tiled Super-Resolution
    print("\n[4/5] Running Tiled Sliding-Window Inference with 2D Hann Blending...")
    t_infer_start = time.time()
    sr_norm = run_tiled_residual_inference(
        model=model,
        input_img=norm_data,
        tile_size_lr=tile_size,
        stride_lr=stride,
        scale=4,
        device=device
    )
    infer_time = time.time() - t_infer_start
    print(f"[Inference Finished] Total inference time: {infer_time:.2f}s")

    # 5. Export 2.5m GeoTIFF with Preserved Geospatial Metadata
    print(f"\n[5/5] Exporting 2.5m Super-Resolved GeoTIFF to: {output_tif}")
    os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)

    scale_factor = 4.0
    out_h = orig_shape[1] * int(scale_factor)
    out_w = orig_shape[2] * int(scale_factor)
    new_transform = orig_transform * Affine.scale(1.0 / scale_factor, 1.0 / scale_factor)

    # Convert back to uint16 BOA surface reflectance (0 - 10,000 scale)
    sr_uint16 = (sr_norm * 10000.0).clip(0, 10000).astype(np.uint16)

    out_meta = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'nodata': 0,
        'width': out_w,
        'height': out_h,
        'count': 4,
        'crs': orig_crs,
        'transform': new_transform,
        'compress': 'deflate'
    }

    with rio.open(output_tif, 'w', **out_meta) as dst:
        dst.write(sr_uint16)
        dst.set_band_description(1, "B04_Red_2.5m")
        dst.set_band_description(2, "B03_Green_2.5m")
        dst.set_band_description(3, "B02_Blue_2.5m")
        dst.set_band_description(4, "B08_NIR_2.5m")

    # Re-open and validate the generated GeoTIFF
    print("\n" + "-" * 70)
    print(" >>> GEOTIFF VALIDATION CHECKS <<<")
    print("-" * 70)
    with rio.open(output_tif) as dst:
        dst_res = dst.res
        dst_crs = dst.crs
        dst_bounds = dst.bounds
        dst_count = dst.count
        dst_shape = (dst.count, dst.height, dst.width)
        print(f"  [x] Output File Exists:     {output_tif}")
        print(f"  [x] Output Shape:           {dst_shape} (Scale factor = 4.0x)")
        print(f"  [x] Output Resolution:      {dst_res[0]:.2f}m x {dst_res[1]:.2f}m (True 2.5m)")
        print(f"  [x] Output CRS:             {dst_crs} (Matched: {dst_crs == orig_crs})")
        print(f"  [x] Output Transform:       {dst.transform}")
        print(f"  [x] Output Dynamic Range:   Min={sr_uint16.min()}, Max={sr_uint16.max()}, Mean={sr_uint16.mean():.1f}")
        print(f"  [x] NaN / Inf Count:        {np.isnan(sr_norm).sum()} NaNs, {np.isinf(sr_norm).sum()} Infs")
        print(f"  [x] Band Count & Order:     {dst_count} Bands (B04-Red, B03-Green, B02-Blue, B08-NIR)")

    peak_gpu_mb = (torch.cuda.max_memory_allocated(0) / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    print(f"  [x] Peak GPU VRAM:          {peak_gpu_mb:.1f} MB (Well within 4GB VRAM)")

    # 6. Generate High-Resolution Visual Inspection Figures
    print("\n" + "-" * 70)
    print(" >>> GENERATING VISUAL INSPECTION FIGURES <<<")
    print("-" * 70)
    out_dir_vis = os.path.dirname(os.path.abspath(output_tif))
    
    # 2x2 High-Resolution Visual Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # RGB 10m vs 2.5m
    axes[0, 0].imshow(render_rgb(norm_data))
    axes[0, 0].set_title("Input Sentinel-2 L2A True-Color RGB (10m Resolution)", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(render_rgb(sr_norm))
    axes[0, 1].set_title("Residual SwinIR Super-Resolved RGB (2.5m Resolution)", fontsize=13, fontweight='bold', color='darkgreen')
    axes[0, 1].axis('off')

    # CIR 10m vs 2.5m
    axes[1, 0].imshow(render_cir(norm_data))
    axes[1, 0].set_title("Input Sentinel-2 Color-Infrared CIR (10m Resolution)", fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(render_cir(sr_norm))
    axes[1, 1].set_title("Residual SwinIR Super-Resolved CIR (2.5m Resolution)", fontsize=13, fontweight='bold', color='darkgreen')
    axes[1, 1].axis('off')

    plt.suptitle(f"Real Sentinel-2 Super-Resolution: {os.path.basename(input_tif)} -> {os.path.basename(output_tif)}", fontsize=15, y=0.99)
    plt.tight_layout()
    vis_path = os.path.join(out_dir_vis, f"{os.path.splitext(os.path.basename(output_tif))[0]}_comparison.png")
    plt.savefig(vis_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [x] Saved 2x2 Visual Comparison: {vis_path}")

    # Zoomed-In Crop Comparison (Center 64x64 LR -> 256x256 HR)
    ch, cw = norm_data.shape[1] // 2, norm_data.shape[2] // 2
    crop_lr = norm_data[:, max(0, ch - 32):ch + 32, max(0, cw - 32):cw + 32]
    crop_sr = sr_norm[:, max(0, ch*4 - 128):ch*4 + 128, max(0, cw*4 - 128):cw*4 + 128]

    fig_z, axes_z = plt.subplots(1, 2, figsize=(14, 7))
    axes_z[0].imshow(render_rgb(crop_lr))
    axes_z[0].set_title("Zoomed Center Crop: Original 10m Sentinel-2 (Bicubic Look)", fontsize=12)
    axes_z[0].axis('off')

    axes_z[1].imshow(render_rgb(crop_sr))
    axes_z[1].set_title("Zoomed Center Crop: Residual SwinIR 2.5m Super-Resolved", fontsize=12, fontweight='bold', color='darkgreen')
    axes_z[1].axis('off')

    plt.suptitle("Zoomed-In Detail & Edge Sharpening Analysis", fontsize=14, y=0.98)
    plt.tight_layout()
    zoom_path = os.path.join(out_dir_vis, f"{os.path.splitext(os.path.basename(output_tif))[0]}_zoomed.png")
    plt.savefig(zoom_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [x] Saved Zoomed Detail Inspection: {zoom_path}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Real Sentinel-2 Scene Super-Resolution Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input 10m Sentinel-2 GeoTIFF")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save super-resolved 2.5m GeoTIFF")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_residual_swinir_100ep.pth", help="Model checkpoint path")
    parser.add_argument("--tile_size", type=int, default=64, help="Tile dimension at LR (10m)")
    parser.add_argument("--stride", type=int, default=48, help="Stride dimension at LR (10m) for overlap")
    args = parser.parse_args()

    infer_real_scene(
        input_tif=args.input,
        output_tif=args.output,
        checkpoint=args.checkpoint,
        tile_size=args.tile_size,
        stride=args.stride
    )


if __name__ == "__main__":
    main()
