# Dataset Policy and Instructions

To keep the repository clean and lightweight, raw and processed satellite datasets (such as individual GeoTIFF scenes) are **not** stored in this git repository.

---

## 1. Dataset Overview

This project uses the **SEN2NAIPv2** dataset for training and evaluation. It contains paired Sentinel-2 Level-2A (10m) and high-resolution NAIP (2.5m) rasters.

- **Total Samples**: 299
- **Train Split**: 199 samples
- **Validation Split**: 50 samples
- **Test Split**: 50 samples

---

## 2. Directory Structure

Place raw and processed datasets under the following directory layout:

```text
data/
├── index_sen2naipv2_unet.parquet   # Dataset index metadata
├── processed/
│   ├── splits.json                 # Pre-defined train/val/test split definitions
│   ├── sample_00001/
│   │   ├── lr.tif                  # 10m Sentinel-2 input (RGBN)
│   │   ├── hr.tif                  # 2.5m NAIP ground truth (RGBN)
│   │   └── meta.json               # Sample metadata (ROI, bounds, date)
│   └── ...
```

---

## 3. How to Obtain the Dataset

1. Download the index file `index_sen2naipv2_unet.parquet` (tracked in the repository for metadata consistency).
2. Download/generate the raw Sentinel-2 and NAIP matching scenes.
3. Run the dataset preparation script to generate the processed samples:
   ```powershell
   python scripts/prepare_dataset.py --index data/index_sen2naipv2_unet.parquet --out_dir data/processed
   ```
4. Define splits or use the pre-configured split definition:
   ```powershell
   python scripts/make_splits.py --data_dir data/processed --out_splits data/processed/splits.json
   ```

---

## 4. Preprocessing Expectations

- **Spectral Bands**: Input and targets must consist of 4 bands in the exact order: **Red (B04), Green (B03), Blue (B02), and NIR (B08)**.
- **Normalization**: Surface reflectance values (stored as uint16) are divided by $10000.0$ to map them to `[0.0, 1.0]` float32 tensors before training.
- **ROI Isolation**: Samples are split strictly by ROI (Region of Interest) IDs to prevent spatial data leakage between train, validation, and test datasets.
