"""
Core Tiled Inference Engine for Real Sentinel-2 Super-Resolution.
Handles memory-safe overlapping patch inference and smooth blending using a 2D Hann window.
"""

import numpy as np
import torch
from tqdm import tqdm


def create_2d_hann_window(patch_size: int) -> np.ndarray:
    """Creates a 2D Hann window for smooth seam blending across overlapping tiles."""
    w1d = np.hanning(patch_size + 2)[1:-1]
    w2d = np.outer(w1d, w1d)
    w2d = np.clip(w2d, 1e-4, 1.0)
    return w2d.astype(np.float32)


def run_tiled_residual_inference(
    model: torch.nn.Module,
    input_img: np.ndarray,
    tile_size_lr: int = 64,
    stride_lr: int = 48,
    scale: int = 4,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """
    Memory-safe sliding-window inference with 2D Hann blending for seamless stitching.
    Args:
        model: Trained PyTorch super-resolution model
        input_img: [4, H_lr, W_lr] normalized input array in [0.0, 1.0]
        tile_size_lr: Tile size in LR coordinates (default 64)
        stride_lr: Stride in LR coordinates (default 48 for overlapping windows)
        scale: Upscale factor (default 4)
        device: PyTorch device
    Returns:
        sr_final: [4, 4*H_lr, 4*W_lr] super-resolved image array in [0.0, 1.0]
    """
    C, H_lr, W_lr = input_img.shape
    tile_size_hr = tile_size_lr * scale
    stride_hr = stride_lr * scale
    H_hr = H_lr * scale
    W_hr = W_lr * scale

    output_accum = np.zeros((C, H_hr, W_hr), dtype=np.float32)
    weight_accum = np.zeros((1, H_hr, W_hr), dtype=np.float32)
    blend_window = create_2d_hann_window(tile_size_hr)[np.newaxis, :, :]

    # Calculate tile coordinates
    y_starts = list(range(0, max(1, H_lr - tile_size_lr + 1), stride_lr))
    if len(y_starts) == 0 or y_starts[-1] + tile_size_lr < H_lr:
        y_starts.append(max(0, H_lr - tile_size_lr))
    
    x_starts = list(range(0, max(1, W_lr - tile_size_lr + 1), stride_lr))
    if len(x_starts) == 0 or x_starts[-1] + tile_size_lr < W_lr:
        x_starts.append(max(0, W_lr - tile_size_lr))

    # Remove duplicates
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    tiles = [(y, x) for y in y_starts for x in x_starts]
    print(f"[Inference] Processing {len(tiles)} overlapping tiles ({tile_size_lr}x{tile_size_lr} -> {tile_size_hr}x{tile_size_hr})...")

    model.eval()
    with torch.no_grad():
        for y_lr, x_lr in tqdm(tiles, desc="Tiled Inference"):
            patch_lr = input_img[:, y_lr:y_lr + tile_size_lr, x_lr:x_lr + tile_size_lr]
            
            # Handle edge padding if scene is smaller than tile
            pad_h = max(0, tile_size_lr - patch_lr.shape[1])
            pad_w = max(0, tile_size_lr - patch_lr.shape[2])
            if pad_h > 0 or pad_w > 0:
                patch_lr = np.pad(patch_lr, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

            t_lr = torch.from_numpy(patch_lr).unsqueeze(0).float().to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                t_sr = model(t_lr)
                t_sr = torch.clamp(t_sr, 0.0, 1.0)
            
            patch_sr = t_sr.squeeze(0).cpu().float().numpy()

            if pad_h > 0 or pad_w > 0:
                patch_sr = patch_sr[:, :patch_sr.shape[1] - pad_h * scale, :patch_sr.shape[2] - pad_w * scale]
                w_window = blend_window[:, :patch_sr.shape[1], :patch_sr.shape[2]]
            else:
                w_window = blend_window

            y_hr = y_lr * scale
            x_hr = x_lr * scale
            h_p = patch_sr.shape[1]
            w_p = patch_sr.shape[2]

            output_accum[:, y_hr:y_hr + h_p, x_hr:x_hr + w_p] += patch_sr * w_window
            weight_accum[:, y_hr:y_hr + h_p, x_hr:x_hr + w_p] += w_window

    weight_accum = np.maximum(weight_accum, 1e-6)
    sr_final = output_accum / weight_accum
    return np.clip(sr_final, 0.0, 1.0)
