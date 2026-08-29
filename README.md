# Sentinel-2 Super-Resolution Mapping (SRM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](#installation)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](#installation)

An operational deep learning pipeline applying a lightweight **Residual SwinIR $\times 4$** model to super-resolve 10m Sentinel-2 Level-2A imagery to a 2.5m spatial resolution across four spectral bands (Red, Green, Blue, Near-Infrared).

---

## 1. Overview

Sentinel-2 provides global, high-frequency optical imagery at a maximum spatial resolution of 10m. This project builds a super-resolution mapping pipeline to upsample 10m Sentinel-2 RGBN data to 2.5m resolution. The output is a super-resolved raster designed for mapping, visualization, and downsteam geospatial tasks.

```text
Sentinel-2 RGBN (10m) ──> Bicubic Upsampling (2.5m) ──> Residual SwinIR Model ──> Super-Resolved RGBN (2.5m)
```

---

## 2. Motivation

Higher-resolution satellite products are highly valuable for urban planning, agricultural monitoring, infrastructure mapping, and conservation management. However, standard sub-meter or 2m commercial satellite acquisitions can be expensive or have low temporal resolution. 

This project aims to reconstruct high-frequency spatial details (such as road edges, building boundaries, and field borders) from free, global Sentinel-2 data. 

> [!IMPORTANT]
> **Scientific Wording & Limits**: This model produces a **super-resolved spatial interpolation** based on learned statistical patterns. It does *not* create or guarantee genuine physical measurements that the original 10m sensor did not capture. It should be used as an enhancement tool, not as a substitute for direct high-resolution physical measurements.

---

## 3. Phase 1 Method & Formulation

Instead of forcing the network to predict the entire high-resolution image from scratch—which often causes severe color shifts or spectral distortion—the model learns a spatial residual:

$$\mathbf{X}_{\text{SR}} = \text{Bicubic}(\mathbf{X}_{\text{LR}}) + \mathcal{R}_\theta(\mathbf{X}_{\text{LR}})$$

Where:
- $\mathbf{X}_{\text{LR}}$ is the input 10m Sentinel-2 RGBN raster.
- $\text{Bicubic}(\mathbf{X}_{\text{LR}})$ is the low-frequency upsampled baseline.
- $\mathcal{R}_\theta(\mathbf{X}_{\text{LR}})$ is the high-frequency residual mapping learned by the SwinIR model.
- $\mathbf{X}_{\text{SR}}$ is the final super-resolved 2.5m output.

### Spectral Bands
The pipeline processes 4 channels in the exact order:
1. **Red** (B04)
2. **Green** (B03)
3. **Blue** (B02)
4. **Near-Infrared / NIR** (B08)

### Model Architecture
The network is based on a lightweight **SwinIR $\times 4$** model optimized for edge devices:
- **Parameters**: 912,244
- **Upscaler**: PixelShuffle $\times 4$
- **Target Profile**: 4GB GPU (e.g., NVIDIA GeForce RTX 2050) with CUDA and Automatic Mixed Precision (AMP).

---

## 4. Dataset

The pipeline is trained and evaluated using the **SEN2NAIPv2** dataset, pairing Sentinel-2 Level-2A (10m) surface reflectance with NAIP (2.5m) aerial imagery:
- **Train Set**: 199 samples
- **Validation Set**: 50 samples
- **Test Set**: 50 samples
- **ROI Isolation**: To prevent spatial data leakage, training, validation, and test splits are geographically isolated by Region of Interest (ROI) boundaries.

*Note: The complete dataset is not stored in this repository. Refer to [`data/README.md`](data/README.md) for download and setup instructions.*

---

## 5. Phase 1 Benchmark Results

The model was validated against the standard bicubic baseline on **50 unseen test samples**:

| Model / Baseline | PSNR Overall (dB) | SSIM | RMSE | SAM (Spectral Angle) |
| :--- | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | 35.00 | 0.8561 | 0.0200 | 2.28° |
| **Residual SwinIR** | **35.87** | **0.8903** | **0.0180** | **1.99°** |
| *Improvement* | *+0.87 dB* | *+0.0342* | *-0.0020* | *-0.29°* |

- **Beat Rate**: **48 out of 50 test samples** ($96\%$) beat the bicubic baseline across all primary structural and spectral evaluation metrics.
- Complete metric summaries can be reviewed in the [Phase 1 Results Report](docs/phase1_results.md).

---

## 6. Independent Real Sentinel-2 Validation

The pipeline was validated on a genuinely unseen, real Sentinel-2 L2A scene over Paris, France (`S2B_MSIL2A_20240822T104619_R051_T31UDQ_20240822T151147`):
- **Input GSD**: 10m
- **Output GSD**: 2.5m (exact $4.0\times$ upscaling)
- **CRS & Geographic Bounds**: Fully preserved in output GeoTIFF.
- **Tiled Inference**: Seamless sliding-window inference with 2D Hann window blending (0 seams, 0 NaNs, 0 Infs).
- **Runtime**: 1.28 seconds on NVIDIA GeForce RTX 2050 (4GB VRAM).

> [!NOTE]
> No quantitative ground-truth metrics (PSNR, SSIM, SAM) are computed for the independent scene, as concurrent physical 2.5m measurements are unavailable. Validation is confirmed via geospatial metadata integrity and visual edge sharping.

Detailed results and comparisons are available in the [Real Scene Validation Report](docs/real_s2_validation.md).

---

## 7. Repository Structure

```text
sentinel2-super-resolution-srm/
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── requirements.txt
│
├── configs/
│   └── phase1.yaml                 # Phase 1 training configuration parameters
│
├── src/                            # Core package
│   ├── __init__.py
│   ├── dataset.py                  # PyTorch dataset loader for RGBN pairs
│   ├── model.py                    # SwinIR network definition
│   ├── train.py                    # Training loops
│   ├── evaluate.py                 # Evaluation methods
│   ├── inference.py                # Generic inference wrapper
│   ├── inference_real_s2.py        # Core tiled inference and blending engine
│   ├── metrics.py                  # SAM, PSNR, SSIM, RMSE calculations
│   ├── preprocessing.py            # GeoTIFF loading and normalization
│   └── utils.py                    # Helpers, seeds, visual rendering
│
├── scripts/                        # Executable scripts
│   ├── download_real_s2.py         # Download real scene from STAC
│   ├── make_splits.py              # Generate dataset splits
│   ├── prepare_dataset.py          # Process and tile dataset
│   ├── prepare_real_s2_scene.py    # Preprocess real scene rasters
│   ├── run_full_diagnostics.py     # Generate diagnostics and benchmarks
│   └── verify_phase1.py            # Pipeline verification suite
│
├── docs/                           # Documentation
│   ├── methodology.md
│   ├── phase1_results.md
│   └── real_s2_validation.md
│
├── results/                        # Summary metrics & final comparison plots
│   ├── benchmark/
│   ├── diagnostics/
│   └── real_s2/
│
├── checkpoints/
│   └── README.md                   # Trained model checkpoint documentation
│
└── data/
    └── README.md                   # Dataset guidelines and structure
```

---

## 8. Installation

This project is tested on Windows 11 with CUDA 11.8+ / Python 3.10+.

1. **Clone the Repository**:
   ```powershell
   git clone https://github.com/krish10a/sentinel2-super-resolution-srm.git
   cd sentinel2-super-resolution-srm
   ```

2. **Create Virtual Environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 9. Usage

### A. Run Verification Suite
Verify that the environment, dependencies, and model weights are ready and operational:
```powershell
python scripts/verify_phase1.py
```

### B. Preprocess and Split Dataset
Prepare the `SEN2NAIPv2` dataset and partition it into train, validation, and test splits:
```powershell
python scripts/prepare_dataset.py --index data/index_sen2naipv2_unet.parquet --out_dir data/processed
python scripts/make_splits.py --data_dir data/processed --out_splits data/processed/splits.json
```

### C. Train the Model
Train the Residual SwinIR network on the prepared splits for 100 epochs:
```powershell
python -m src.train --epochs 100 --batch_size 2 --lr 4e-4
```

### D. Evaluate on Test Set
Evaluate the trained model against the test split and export metrics:
```powershell
python -m src.evaluate --checkpoint checkpoints/best_residual_swinir_100ep.pth --out_dir results/benchmark
```

### E. Run Inference on Real Sentinel-2 Scene
Apply the model to an unseen, arbitrary-sized real Sentinel-2 Level-2A GeoTIFF scene:
```powershell
# Download real scene via STAC API (Paris, France)
python scripts/download_real_s2.py --output data/real_s2/paris_scene_10m.tif

# Run super-resolution inference
python scripts/inference_real_s2.py --input data/real_s2/paris_scene_10m.tif --output results/real_s2/paris_scene_SR_2p5m.tif --checkpoint checkpoints/best_residual_swinir_100ep.pth
```

---

## 10. Hardware Configuration
- **GPU Target**: NVIDIA GeForce RTX 2050 (4GB VRAM) or equivalent.
- **CUDA mixed precision (AMP)** is enabled by default to stay within the 4GB VRAM boundary during training and inference.

---

## 11. Limitations & Scope
- **Geographic Coverage**: The model may generalize poorly to landscapes not represented in the `SEN2NAIPv2` dataset (e.g., hyper-arid deserts or snow-covered polar regions).
- **No Hallucination Prevention**: Like all generative super-resolution models, it can synthesize micro-features (like building textures) that might not match exact ground truth structure.
- **Operational Validation**: Larger-scale geographic and temporal validation is required before using the output rasters in operational pipelines.

---

## 12. Future Work (Phase 2)
High-level directions for Phase 2:
- **Loss Enhancements**: Integrating spectral-aware losses (e.g., SAM loss) and perceptual losses to improve texture sharpness.
- **Contextual Guidance**: Incorporating elevation (DEM) or land cover priors as additional model inputs.
- **Scale Flexibility**: Supporting dynamic downscaling or variable magnification scales.
- **Uncertainty Estimation**: Outputting pixel-wise confidence maps alongside the super-resolved raster.
