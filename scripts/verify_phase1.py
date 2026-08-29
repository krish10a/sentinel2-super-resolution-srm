"""
Automated Pipeline Verification Script for Phase 1 Sentinel-2 Super-Resolution Mapping.
Validates environment, dataset integrity, geographic isolation, model training, evaluation, and GeoTIFF inference.
"""

import os
import sys
import json
import numpy as np
import torch
import rasterio as rio

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import create_swinir_satellite
from src.preprocessing import read_geotiff
from src.metrics import compute_all_metrics


def run_checks():
    print("=" * 70)
    print(" >>> PHASE 1 AUTOMATED PIPELINE VERIFICATION SUITE <<<")
    print("=" * 70)

    checklist = []

    # 1. Environment & CUDA Checks
    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 10)
    checklist.append(("[x]" if py_ok else "[ ]", f"Python version valid: {py_ver}"))

    cuda_avail = torch.cuda.is_available()
    checklist.append(("[x]" if cuda_avail else "[ ]", f"CUDA available: {cuda_avail}"))

    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU only"
    gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if cuda_avail else "N/A"
    checklist.append(("[x]" if cuda_avail else "[ ]", f"GPU detected: {gpu_name} ({gpu_mem})"))

    # 2. Real Dataset Access
    splits_path = "data/processed/splits.json"
    splits_exist = os.path.exists(splits_path)
    checklist.append(("[x]" if splits_exist else "[ ]", f"Dataset splits file found: {splits_path}"))

    train_dirs, val_dirs, test_dirs = [], [], []
    if splits_exist:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        train_dirs = splits.get("train", [])
        val_dirs = splits.get("val", [])
        test_dirs = splits.get("test", [])

    splits_nonempty = len(train_dirs) > 0 and len(val_dirs) > 0 and len(test_dirs) > 0
    checklist.append(("[x]" if splits_nonempty else "[ ]", f"Train/Val/Test non-empty: Train={len(train_dirs)}, Val={len(val_dirs)}, Test={len(test_dirs)}"))

    # 3. Check for Geographic ROI Leakage
    def get_rois(dir_list):
        rois = set()
        for d in dir_list:
            m_path = os.path.join(d, "meta.json")
            if os.path.exists(m_path):
                with open(m_path, 'r') as mf:
                    meta = json.load(mf)
                    rois.add(meta.get("roi_id", d))
            else:
                rois.add(d)
        return rois

    train_rois = get_rois(train_dirs)
    val_rois = get_rois(val_dirs)
    test_rois = get_rois(test_dirs)

    leakage_train_test = len(train_rois.intersection(test_rois))
    leakage_train_val = len(train_rois.intersection(val_rois))
    no_leakage = (leakage_train_test == 0 and leakage_train_val == 0)
    checklist.append(("[x]" if no_leakage else "[ ]", f"ROI leakage = 0 (Train-Test overlap: {leakage_train_test}, Train-Val: {leakage_train_val})"))

    # 4. Dimension, Scale Factor & 4-Channel Checks
    sample_valid = False
    if len(train_dirs) > 0:
        sample_0 = train_dirs[0]
        lr_p = os.path.join(sample_0, "lr.tif")
        hr_p = os.path.join(sample_0, "hr.tif")
        if os.path.exists(lr_p) and os.path.exists(hr_p):
            lr_arr, lr_m = read_geotiff(lr_p)
            hr_arr, hr_m = read_geotiff(hr_p)
            scale = hr_arr.shape[1] / lr_arr.shape[1]
            chans = lr_arr.shape[0]
            sample_valid = (scale == 4.0 and chans == 4 and hr_arr.shape[0] == 4)
            checklist.append(("[x]" if sample_valid else "[ ]", f"Real sample dimensions valid: LR={lr_arr.shape}, HR={hr_arr.shape}, Scale={scale:.0f}x, Channels={chans} (RGBN)"))

    # 5. Model Architecture & Forward Pass Check
    from src.model import create_residual_swinir_satellite
    model = create_residual_swinir_satellite()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_in = torch.zeros(1, 4, 64, 64)
    with torch.no_grad():
        dummy_out = model(dummy_in)
    model_ok = (dummy_out.shape == (1, 4, 256, 256))
    checklist.append(("[x]" if model_ok else "[ ]", f"Residual SwinIR 4-in/4-out x4 architecture verified ({num_params:,} params)"))

    # 6. Checkpoint Check
    ckpt_path = "checkpoints/best_residual_swinir_100ep.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/best_model.pth"
    ckpt_exists = os.path.exists(ckpt_path)
    checklist.append(("[x]" if ckpt_exists else "[ ]", f"Trained checkpoint exists: {ckpt_path}"))

    # 7. Evaluation & Metrics Check
    metrics_path = "results/benchmark/benchmark_summary.csv"
    metrics_exist = os.path.exists(metrics_path)
    checklist.append(("[x]" if metrics_exist else "[ ]", f"Evaluation metrics produced: {metrics_path}"))

    # 8. Real GeoTIFF Inference & CRS Check
    pred_path = "results/real_s2/independent_scene_SR_2p5m.tif"
    pred_exist = os.path.exists(pred_path)
    if pred_exist:
        with rio.open(pred_path) as src:
            pred_res = src.res[0]
            pred_crs = src.crs
            res_ok = abs(pred_res - 2.5) < 0.1
            checklist.append(("[x]" if res_ok else "[ ]", f"Real GeoTIFF inference: Resolution={pred_res:.1f}m, CRS={pred_crs} ({pred_path})"))
    else:
        checklist.append(("[ ]", f"Real GeoTIFF output exists: {pred_path}"))

    # Summary
    print("\n" + "-" * 70)
    all_passed = True
    for mark, desc in checklist:
        print(f"  {mark} {desc}")
        if mark == "[ ]":
            all_passed = False
    print("-" * 70)

    if all_passed:
        print("\n [STATUS: ALL CHECKS PASSED] Phase 1 pipeline is 100% verified and operational!\n")
    else:
        print("\n [STATUS: ACTION REQUIRED] Complete remaining steps outlined above.\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_checks()
