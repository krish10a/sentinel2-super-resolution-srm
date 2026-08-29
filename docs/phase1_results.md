# Phase 1 Benchmark Results

This document presents the quantitative evaluation metrics of the **Residual SwinIR $\times 4$** super-resolution prototype model compared to the standard bicubic interpolation baseline.

---

## 1. Quantitative Metrics Evaluation

The evaluation was performed across **50 unseen geographic test samples** from the `SEN2NAIPv2` dataset.

| Evaluation Metric | Bicubic Baseline (2.5m) | Residual SwinIR (100 Epochs) | Improvement / Gap |
| :--- | :--- | :--- | :--- |
| **PSNR Overall** | 35.00 dB | **35.87 dB** | **+0.87 dB** |
| **SSIM Overall** | 0.8561 | **0.8903** | **+0.0342** |
| **RMSE Overall** | 0.0200 | **0.0180** | **-0.0020** |
| **SAM (Spectral Angle)** | 2.28° | **1.99°** | **-0.29°** |

### Key Performance Highlight
- **48 out of 50 test samples** ($96\%$) successfully beat the bicubic baseline across all primary structural and spectral metrics.

---

## 2. Residual Super-Resolution Formulation

Instead of forcing the network to predict the entire 2.5m high-resolution image from scratch (which often leads to severe color shifts and pixel hallucinations), the model adopts a residual learning formulation:

$$\mathbf{X}_{\text{SR}} = \text{Bicubic}(\mathbf{X}_{\text{LR}}) + \mathcal{R}_\theta(\mathbf{X}_{\text{LR}})$$

Where:
- $\mathbf{X}_{\text{LR}}$ is the 10m input Sentinel-2 RGBN image.
- $\text{Bicubic}(\mathbf{X}_{\text{LR}})$ is the low-frequency bicubic upsampled baseline.
- $\mathcal{R}_\theta(\mathbf{X}_{\text{LR}})$ is the high-frequency residual details learned by the SwinIR network.
- $\mathbf{X}_{\text{SR}}$ is the final super-resolved 2.5m RGBN output.

---

## 3. Training Setup and Parameters

- **Training Split**: 199 samples (strictly isolated by ROI identifier, preventing spatial leakage).
- **Validation Split**: 50 samples.
- **Test Split**: 50 samples.
- **Optimization**: AdamW optimizer with Cosine Annealing learning rate schedule.
- **Precision**: 16-bit Mixed Precision (AMP) to fit within 4GB VRAM target (e.g., NVIDIA RTX 2050).
- **Epochs**: 100 epochs.

---

## 4. Scientific Scope & Limitations

> [!WARNING]
> - **No Physical Information Guarantee**: The super-resolved raster represents a learned spatial interpolation of the 10m input. It does *not* represent physical ground-truth measurements at 2.5m resolution.
> - **Dataset Dependency**: Performance is bounded by the geographic and seasonal distribution of the `SEN2NAIPv2` training dataset.
