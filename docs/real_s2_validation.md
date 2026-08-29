# Independent Real Sentinel-2 Scene Validation

This document outlines the validation process of the trained Phase 1 prototype on a genuinely unseen real-world Sentinel-2 Level-2A scene.

---

## 1. Validation Scene Parameters

The validation scene was acquired from the **Microsoft Planetary Computer STAC catalog** over Paris, France.

- **Scene Product ID**: `S2B_MSIL2A_20240822T104619_R051_T31UDQ_20240822T151147`
- **Acquisition Date**: August 22, 2024
- **Cloud Cover**: 0.12% (Near-perfect clear conditions)
- **Input Dimensions**: $256 \times 256$ pixels (10m GSD)
- **Output Dimensions**: $1024 \times 1024$ pixels (2.5m GSD)
- **Coordinate Reference System (CRS)**: `EPSG:32631` (UTM Zone 31N)

---

## 2. Band Ordering & Spatial Resolution

Both input and output rasters follow the strict 4-band ordering:
1. **Band 1**: Red (B04)
2. **Band 2**: Green (B03)
3. **Band 3**: Blue (B02)
4. **Band 4**: Near-Infrared / NIR (B08)

Output GSD is exactly $2.5\text{m} \times 2.5\text{m}$, verified from the GeoTIFF transform.

---

## 3. Validation Checklist

| Check | Result | Verification Status |
| :--- | :--- | :--- |
| **Genuinely Independent?** | Yes | Located in Paris, France (zero overlap with `SEN2NAIPv2` ROIs) |
| **Geospatial CRS Preserved?** | Yes | EPSG:32631 preserved correctly |
| **Affine Transform Updated?** | Yes | Scaled by exactly $4.0\times$ |
| **NaN / Inf Values?** | None | 0 NaNs, 0 Infs detected |
| **Seam / Tile Boundaries?** | None | Blended perfectly via 2D Hann window |
| **Inference Time** | 1.28 sec | Evaluated on NVIDIA GeForce RTX 2050 (4GB) |

---

## 4. Limitations & Scientific Wording

> [!IMPORTANT]
> **No Ground Truth Benchmark**: Because concurrent, physical 2.5m satellite sensor measurements are unavailable for this scene, no PSNR, SSIM, or SAM scores are computed.
> 
> **Resolution Representation**: The output GeoTIFF is a **2.5m-resolution super-resolved raster**, not a measured 2.5m physical product. The model sharpens structures (such as buildings, agricultural fields, and roads) but does not guarantee the recovery of true physical features that were not captured by the 10m Sentinel-2 sensor.
