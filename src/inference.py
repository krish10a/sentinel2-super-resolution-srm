"""
Real Sentinel-2 Scene Inference Engine.
Handles:
- Arbitrary sized 4-band Sentinel-2 L2A GeoTIFFs (B04-Red, B03-Green, B02-Blue, B08-NIR)
- Tiled sliding-window inference with overlapping Hann/cosine blending (eliminates edge artifacts)
- GeoTIFF export preserving CRS, geographic coordinates, and updating Affine transform to 2.5m resolution
- Visual inspection figures comparing 10m input vs 2.5m output in True Color (RGB) and CIR (NIR)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import create_swinir_satellite
from src.preprocessing import read_geotiff, write_geotiff, normalize_reflectance, denormalize_reflectance
from src.utils import get_device, save_real_scene_comparison


def create_2d_window(patch_size: int) -> np.ndarray:
    """Creates a 2D Hann window for smooth seam blending during tiled stitching."""
    w1d = np.hanning(patch_size + 2)[1:-1]
    w2d = np.outer(w1d, w1d)
    w2d = np.clip(w2d, 1e-4, 1.0)
    return w2d.astype(np.float32)


def run_tiled_inference(model: torch.nn.Module,
                        input_img: np.ndarray,
                        tile_size_lr: int = 64,
                        stride_lr: int = 48,
                        device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    Performs memory-safe tiled inference over large Sentinel-2 images with seamless blend.
    Args:
        model: SwinIR PyTorch model
        input_img: [4, H_lr, W_lr] normalized reflectance in [0.0, 1.0]
        tile_size_lr: LR tile dimension (default 64)
        stride_lr: LR stride for overlap (default 48 -> 25% overlap)
        device: torch device
    Returns:
        output_hr: [4, 4*H_lr, 4*W_lr] super-resolved image
    """
    scale = 4
    C, H_lr, W_lr = input_img.shape
    tile_size_hr = tile_size_lr * scale
    stride_hr = stride_lr * scale

    H_hr = H_lr * scale
    W_hr = W_lr * scale

    output_accum = np.zeros((C, H_hr, W_hr), dtype=np.float32)
    weight_accum = np.zeros((1, H_hr, W_hr), dtype=np.float32)
    blend_window = create_2d_window(tile_size_hr)[np.newaxis, :, :]  # [1, 256, 256]

    # Generate tile coordinates
    y_starts = list(range(0, max(1, H_lr - tile_size_lr + 1), stride_lr))
    if y_starts[-1] + tile_size_lr < H_lr:
        y_starts.append(H_lr - tile_size_lr)
    
    x_starts = list(range(0, max(1, W_lr - tile_size_lr + 1), stride_lr))
    if x_starts[-1] + tile_size_lr < W_lr:
        x_starts.append(W_lr - tile_size_lr)

    # Pad image if smaller than tile_size_lr
    pad_h = max(0, tile_size_lr - H_lr)
    pad_w = max(0, tile_size_lr - W_lr)
    if pad_h > 0 or pad_w > 0:
        input_img = np.pad(input_img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        y_starts = [0]
        x_starts = [0]

    tiles = []
    for y in y_starts:
        for x in x_starts:
            tiles.append((y, x))

    print(f"[Inference] Running tiled processing: {len(tiles)} tiles of {tile_size_lr}x{tile_size_lr} -> {tile_size_hr}x{tile_size_hr}...")

    model.eval()
    with torch.no_grad():
        for y_lr, x_lr in tqdm(tiles, desc="Tiling Progress"):
            tile_lr = input_img[:, y_lr:y_lr + tile_size_lr, x_lr:x_lr + tile_size_lr]
            t_lr = torch.from_numpy(tile_lr).unsqueeze(0).to(device)

            t_sr = model(t_lr)
            t_sr = torch.clamp(t_sr, 0.0, 1.0)
            sr_patch = t_sr.squeeze(0).cpu().numpy()

            y_hr = y_lr * scale
            x_hr = x_lr * scale

            output_accum[:, y_hr:y_hr + tile_size_hr, x_hr:x_hr + tile_size_hr] += sr_patch * blend_window
            weight_accum[:, y_hr:y_hr + tile_size_hr, x_hr:x_hr + tile_size_hr] += blend_window

    # Normalize by accumulated weights
    weight_accum = np.maximum(weight_accum, 1e-6)
    final_sr = output_accum / weight_accum

    # Unpad if padded
    final_sr = final_sr[:, :H_hr, :W_hr]
    return np.clip(final_sr, 0.0, 1.0)


def infer_scene(args):
    device = get_device()

    # 1. Read Input GeoTIFF
    print(f"[Inference] Reading Sentinel-2 scene: {args.input_tif}")
    raw_data, meta = read_geotiff(args.input_tif)
    print(f"[Inference] Loaded image shape: {raw_data.shape} (Channels: {raw_data.shape[0]}, H: {raw_data.shape[1]}, W: {raw_data.shape[2]})")

    # If band order needs adjustment: Default expects [B04-Red, B03-Green, B02-Blue, B08-NIR]
    if raw_data.shape[0] < 4:
        raise ValueError(f"Sentinel-2 input must contain at least 4 bands (got {raw_data.shape[0]})")
    
    data_4b = raw_data[0:4, :, :]

    # 2. Normalize
    norm_data = normalize_reflectance(data_4b, max_val=args.reflectance_max)

    # 3. Load SwinIR Model
    model = create_swinir_satellite(
        embed_dim=args.embed_dim,
        window_size=args.window_size
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Inference] Loaded model weights from: {args.checkpoint}")

    # 4. Run Tiled Inference
    sr_norm = run_tiled_inference(
        model=model,
        input_img=norm_data,
        tile_size_lr=args.tile_size,
        stride_lr=args.stride,
        device=device
    )

    # 5. Denormalize to target format
    sr_output = denormalize_reflectance(sr_norm, max_val=args.reflectance_max, target_range=args.output_format)
    print(f"[Inference] Super-resolved output shape: {sr_output.shape} (2.5m resolution)")

    # 6. Save GeoTIFF
    os.makedirs(os.path.dirname(os.path.abspath(args.output_tif)), exist_ok=True)
    write_geotiff(
        filepath=args.output_tif,
        data=sr_output,
        meta=meta,
        upscale_factor=4.0,
        nodata_val=-9999.0
    )
    print(f"[Inference] [SAVED] Successfully wrote 2.5m GeoTIFF to: {args.output_tif}")

    # 7. Generate comparison visualization
    if args.save_vis:
        vis_path = os.path.join(args.output_vis_dir, "real_s2_comparison.png")
        save_real_scene_comparison(
            lr=norm_data,
            sr=sr_norm,
            save_path=vis_path,
            title="Real Sentinel-2 Super-Resolution (10m -> 2.5m)"
        )
        print(f"[Inference] [SAVED] Saved visual comparison figure to: {vis_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Real Sentinel-2 Scene Super-Resolution Inference")
    parser.add_argument("--input_tif", type=str, required=True, help="Path to input 4-band Sentinel-2 GeoTIFF (10m)")
    parser.add_argument("--output_tif", type=str, default="outputs/predictions/s2_super_resolved_2_5m.tif", help="Path to output GeoTIFF (2.5m)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--output_vis_dir", type=str, default="outputs/comparisons", help="Directory for visual figures")
    parser.add_argument("--tile_size", type=int, default=64, help="Tile size for LR sliding window (default 64)")
    parser.add_argument("--stride", type=int, default=48, help="Stride for LR sliding window (default 48)")
    parser.add_argument("--embed_dim", type=int, default=60, help="SwinIR embedding dimension")
    parser.add_argument("--window_size", type=int, default=8, help="SwinIR window size")
    parser.add_argument("--reflectance_max", type=float, default=10000.0, help="Sentinel-2 L2A max reflectance scale")
    parser.add_argument("--output_format", type=str, default="original", choices=["original", "unit", "uint8"], help="Output value range")
    parser.add_argument("--save_vis", action="store_true", default=True, help="Save visual comparison image")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer_scene(args)
