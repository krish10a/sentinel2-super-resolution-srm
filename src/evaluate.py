"""
Evaluation and Benchmarking Script for Sentinel-2 Super-Resolution (x4).
Evaluates both Bicubic baseline and SwinIR against Ground Truth HR.
Computes:
- PSNR (Overall, RGB, NIR)
- SSIM (Overall, RGB, NIR)
- RMSE (Overall, RGB, NIR)
- SAM (Overall, RGB in degrees)
Saves comparison figures and exports metrics to CSV and JSON.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def format_table(data, headers):
    """Fallback ASCII table formatting if tabulate is not installed."""
    if HAS_TABULATE:
        return tabulate(data, headers=headers, tablefmt="grid")
    
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    rows_str = "\n".join(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)) for row in data)
    return f"{header_str}\n{sep_str}\n{rows_str}"

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import create_swinir_satellite
from src.dataset import SEN2NAIPDataset
from src.metrics import compute_all_metrics, bicubic_baseline
from src.utils import get_device, save_visual_comparison, save_metrics_summary


def evaluate(args):
    device = get_device()

    os.makedirs(os.path.join(args.output_dir, "comparisons"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)

    # 1. Load test dataset
    if not os.path.exists(args.splits_json):
        raise FileNotFoundError(f"Splits file not found: {args.splits_json}")

    with open(args.splits_json, 'r') as f:
        splits = json.load(f)

    test_dirs = splits.get("test", [])
    if len(test_dirs) == 0:
        raise ValueError("No test samples found in splits.json!")

    test_ds = SEN2NAIPDataset(test_dirs, is_train=False, augment=False, max_samples=args.test_limit)
    print(f"[Evaluate] Loaded {len(test_ds)} test pairs from unseen geographic regions.")

    # 2. Load Model
    model = create_swinir_satellite(
        embed_dim=args.embed_dim,
        window_size=args.window_size
    ).to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Evaluate] Successfully loaded checkpoint weights from {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    model.eval()

    results_swinir = []
    results_bicubic = []
    combined_rows = []

    print("\n" + "=" * 70)
    print(" >>> RUNNING EVALUATION: Bicubic vs SwinIR SR against Ground Truth <<<")
    print("=" * 70)

    for i in tqdm(range(len(test_ds)), desc="Evaluating Test Set"):
        sample = test_ds[i]
        lr_tensor = sample["lr"].unsqueeze(0).to(device)  # [1, 4, 64, 64]
        hr_tensor = sample["hr"].unsqueeze(0).to(device)  # [1, 4, 256, 256]

        lr_np = sample["lr"].numpy()  # [4, 64, 64]
        hr_np = sample["hr"].numpy()  # [4, 256, 256]

        # 1. Bicubic Baseline
        bicubic_np = bicubic_baseline(lr_np, scale=4)
        metrics_bicubic = compute_all_metrics(bicubic_np, hr_np)
        metrics_bicubic["method"] = "Bicubic"
        metrics_bicubic["sample_idx"] = i
        results_bicubic.append(metrics_bicubic)

        # 2. SwinIR Prediction
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
            sr_np = sr_tensor.squeeze(0).cpu().float().numpy()

        metrics_swinir = compute_all_metrics(sr_np, hr_np)
        metrics_swinir["method"] = "SwinIR"
        metrics_swinir["sample_idx"] = i
        results_swinir.append(metrics_swinir)

        # Combined per-image row for CSV
        combined_rows.append({
            "sample_idx": i,
            "sample_dir": sample["sample_dir"],
            "bicubic_psnr": metrics_bicubic["psnr_overall"],
            "swinir_psnr": metrics_swinir["psnr_overall"],
            "psnr_gain": metrics_swinir["psnr_overall"] - metrics_bicubic["psnr_overall"],
            "bicubic_ssim": metrics_bicubic["ssim_overall"],
            "swinir_ssim": metrics_swinir["ssim_overall"],
            "ssim_gain": metrics_swinir["ssim_overall"] - metrics_bicubic["ssim_overall"],
            "bicubic_sam_deg": metrics_bicubic["sam_overall_deg"],
            "swinir_sam_deg": metrics_swinir["sam_overall_deg"],
            "bicubic_rmse": metrics_bicubic["rmse_overall"],
            "swinir_rmse": metrics_swinir["rmse_overall"],
            "swinir_psnr_rgb": metrics_swinir["psnr_rgb"],
            "swinir_psnr_nir": metrics_swinir["psnr_nir"]
        })

        # Save visual comparison for first N samples
        if i < args.num_visualizations:
            img_save_path = os.path.join(args.output_dir, "comparisons", f"test_sample_{i:04d}.png")
            save_visual_comparison(
                bicubic=bicubic_np,
                sr=sr_np,
                ground_truth=hr_np,
                save_path=img_save_path,
                title_prefix=f"Test Sample #{i:03d} (Unseen Location)",
                metrics=metrics_swinir
            )

    # 3. Compute Aggregate Statistics
    def aggregate(res_list):
        agg = {}
        for key in res_list[0].keys():
            if key in ["method", "sample_idx"]:
                continue
            vals = [r[key] for r in res_list]
            agg[key + "_mean"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        return agg

    agg_bicubic = aggregate(results_bicubic)
    agg_swinir = aggregate(results_swinir)

    # Save to JSON & CSV
    json_path = os.path.join(args.output_dir, "metrics", "test_metrics.json")
    csv_path = os.path.join(args.output_dir, "metrics", "test_metrics.csv")
    save_metrics_summary(combined_rows, json_path, csv_path)

    summary_json_path = os.path.join(args.output_dir, "metrics", "summary_benchmark.json")
    with open(summary_json_path, 'w') as f:
        json.dump({"bicubic": agg_bicubic, "swinir": agg_swinir}, f, indent=2)

    # 4. Print Beautiful Markdown / CLI Table
    table_data = [
        ["PSNR Overall (dB) [Higher Better]", f"{agg_bicubic['psnr_overall_mean']:.2f} +/- {agg_bicubic['psnr_overall_std']:.2f}", f"{agg_swinir['psnr_overall_mean']:.2f} +/- {agg_swinir['psnr_overall_std']:.2f}", f"{agg_swinir['psnr_overall_mean'] - agg_bicubic['psnr_overall_mean']:+.2f} dB"],
        ["SSIM Overall [Higher Better]", f"{agg_bicubic['ssim_overall_mean']:.4f} +/- {agg_bicubic['ssim_overall_std']:.4f}", f"{agg_swinir['ssim_overall_mean']:.4f} +/- {agg_swinir['ssim_overall_std']:.4f}", f"{agg_swinir['ssim_overall_mean'] - agg_bicubic['ssim_overall_mean']:+.4f}"],
        ["RMSE Overall [Lower Better]", f"{agg_bicubic['rmse_overall_mean']:.4f} +/- {agg_bicubic['rmse_overall_std']:.4f}", f"{agg_swinir['rmse_overall_mean']:.4f} +/- {agg_swinir['rmse_overall_std']:.4f}", f"{agg_swinir['rmse_overall_mean'] - agg_bicubic['rmse_overall_mean']:+.4f}"],
        ["SAM Overall (deg) [Lower Better]", f"{agg_bicubic['sam_overall_deg_mean']:.2f} deg +/- {agg_bicubic['sam_overall_deg_std']:.2f}", f"{agg_swinir['sam_overall_deg_mean']:.2f} deg +/- {agg_swinir['sam_overall_deg_std']:.2f}", f"{agg_swinir['sam_overall_deg_mean'] - agg_bicubic['sam_overall_deg_mean']:+.2f} deg"],
        ["PSNR RGB (dB) [Higher Better]", f"{agg_bicubic['psnr_rgb_mean']:.2f}", f"{agg_swinir['psnr_rgb_mean']:.2f}", f"{agg_swinir['psnr_rgb_mean'] - agg_bicubic['psnr_rgb_mean']:+.2f} dB"],
        ["PSNR NIR (dB) [Higher Better]", f"{agg_bicubic['psnr_nir_mean']:.2f}", f"{agg_swinir['psnr_nir_mean']:.2f}", f"{agg_swinir['psnr_nir_mean'] - agg_bicubic['psnr_nir_mean']:+.2f} dB"],
        ["SSIM NIR [Higher Better]", f"{agg_bicubic['ssim_nir_mean']:.4f}", f"{agg_swinir['ssim_nir_mean']:.4f}", f"{agg_swinir['ssim_nir_mean'] - agg_bicubic['ssim_nir_mean']:+.4f}"]
    ]

    print("\n" + "=" * 70)
    print(" >>> FINAL PHASE 1 BENCHMARK RESULTS <<<")
    print("=" * 70)
    print(format_table(table_data, headers=["Metric", "Bicubic (2.5m)", "SwinIR SR (2.5m)", "Gain / Difference"]))
    print(f"\n[Saved] Detailed per-image metrics saved to: {csv_path}")
    print(f"[Saved] Summary benchmark JSON saved to: {summary_json_path}")
    print(f"[Saved] Visual comparison plots saved in: {os.path.join(args.output_dir, 'comparisons')}")
    print("=" * 70 + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate Sentinel-2 Super-Resolution")
    parser.add_argument("--splits_json", type=str, default="data/processed/splits.json", help="Path to splits.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--test_limit", type=int, default=200, help="Max test samples to evaluate")
    parser.add_argument("--num_visualizations", type=int, default=20, help="Number of comparison figures to save")
    parser.add_argument("--embed_dim", type=int, default=60, help="SwinIR embedding dimension")
    parser.add_argument("--window_size", type=int, default=8, help="SwinIR window size")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)
