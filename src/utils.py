"""
Utility Functions: Logging, Visualization, Checkpoint Management, and Reproducibility.
"""

import os
import random
import json
import csv
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Returns CUDA device if available (e.g. RTX 3050/2050 4GB), otherwise CPU."""
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] Using GPU: {dev_name} ({vram_gb:.2f} GB VRAM)")
        return torch.device("cuda:0")
    else:
        print("[Device] CUDA not available, using CPU")
        return torch.device("cpu")


def percentile_stretch(img: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """
    Applies 2%-98% percentile linear contrast stretching for realistic satellite visualization.
    img: [H, W, C] in range [0, 1] or arbitrary reflectance.
    """
    img_stretched = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        band = img[:, :, c]
        p_low = np.percentile(band, lower_pct)
        p_high = np.percentile(band, upper_pct)
        if p_high > p_low:
            stretched = (band - p_low) / (p_high - p_low)
        else:
            stretched = band
        img_stretched[:, :, c] = np.clip(stretched, 0.0, 1.0)
    return img_stretched


def render_rgb(data: np.ndarray) -> np.ndarray:
    """
    Extracts RGB (B04-Red, B03-Green, B02-Blue = Bands 0, 1, 2) and applies contrast stretch.
    Input: [C, H, W] where C >= 3.
    Returns: [H, W, 3] in [0, 1] for plt.imshow.
    """
    rgb = data[0:3, :, :].transpose(1, 2, 0)  # [H, W, 3]
    return percentile_stretch(rgb)


def render_cir(data: np.ndarray) -> np.ndarray:
    """
    Extracts Color-Infrared (CIR: B08-NIR, B04-Red, B03-Green = Bands 3, 0, 1) for vegetation analysis.
    Input: [C, H, W] where C >= 4.
    Returns: [H, W, 3] in [0, 1] for plt.imshow.
    """
    cir = np.stack([data[3], data[0], data[1]], axis=-1)  # [H, W, 3]
    return percentile_stretch(cir)


def save_visual_comparison(bicubic: np.ndarray,
                           sr: np.ndarray,
                           ground_truth: np.ndarray,
                           save_path: str,
                           title_prefix: str = "Super-Resolution Comparison (x4)",
                           metrics: Optional[Dict[str, float]] = None):
    """
    Generates and saves the required 2x2 comparison figure:
    ┌────────────┬────────────┐
    │ Bicubic    │ GroundTruth│
    ├────────────┼────────────┤
    │ SwinIR SR  │ Difference │
    └────────────┴────────────┘
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Top-Left: Bicubic RGB
    bicubic_rgb = render_rgb(bicubic)
    axes[0, 0].imshow(bicubic_rgb)
    axes[0, 0].set_title("Bicubic Baseline (2.5m)", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # Top-Right: Ground Truth HR RGB
    gt_rgb = render_rgb(ground_truth)
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Ground Truth HR (2.5m)", fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    # Bottom-Left: SwinIR SR RGB
    sr_rgb = render_rgb(sr)
    axes[1, 0].imshow(sr_rgb)
    sr_title = "SwinIR SR (2.5m)"
    if metrics:
        sr_title += f"\nPSNR: {metrics.get('psnr_overall', 0):.2f} dB | SSIM: {metrics.get('ssim_overall', 0):.4f} | SAM: {metrics.get('sam_overall_deg', 0):.2f}°"
    axes[1, 0].set_title(sr_title, fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')

    # Bottom-Right: Error Difference Map (Mean absolute error across bands)
    diff = np.mean(np.abs(sr - ground_truth), axis=0)  # [H, W]
    im_diff = axes[1, 1].imshow(diff, cmap='inferno', vmin=0.0, vmax=0.2)
    axes[1, 1].set_title("Residual Error Map (|SR - GT|)", fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(title_prefix, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_real_scene_comparison(lr: np.ndarray,
                               sr: np.ndarray,
                               save_path: str,
                               title: str = "Real Sentinel-2 Inference (10m -> 2.5m)"):
    """
    Generates comparison figure for real scenes where ground truth is unavailable:
    Side-by-side RGB and False Color CIR views.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Top-Left: Original 10m S2 RGB
    axes[0, 0].imshow(render_rgb(lr))
    axes[0, 0].set_title("Original Sentinel-2 (10m RGB)", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # Top-Right: SwinIR SR 2.5m RGB
    axes[0, 1].imshow(render_rgb(sr))
    axes[0, 1].set_title("SwinIR Super-Resolved (2.5m RGB)", fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    # Bottom-Left: Original 10m S2 CIR (False Color NIR)
    axes[1, 0].imshow(render_cir(lr))
    axes[1, 0].set_title("Original Sentinel-2 (10m Color-Infrared NIR)", fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')

    # Bottom-Right: SwinIR SR 2.5m CIR
    axes[1, 1].imshow(render_cir(sr))
    axes[1, 1].set_title("SwinIR Super-Resolved (2.5m Color-Infrared NIR)", fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_checkpoint(state: Dict[str, Any], filepath: str):
    """Saves model checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None) -> int:
    """Loads checkpoint weights and state."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0)
    print(f"[Checkpoint] Successfully loaded checkpoint from {filepath} (Epoch {start_epoch})")
    return start_epoch


def save_metrics_summary(results: list, json_path: str, csv_path: str):
    """Saves per-image and aggregate metrics to JSON and CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    if len(results) > 0:
        keys = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
