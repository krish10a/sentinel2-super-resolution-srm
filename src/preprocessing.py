"""
Preprocessing and Geospatial I/O for Sentinel-2 & High-Res Super-Resolution.
Handles GeoTIFF reading/writing, CRS/Affine transform preservation,
reflectance normalization, nodata/NaN cleaning, and patch extraction.
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# Default reflectance scaling for Sentinel-2 L2A (10000 = 1.0 BOA reflectance)
DEFAULT_REFLECTANCE_MAX = 10000.0


def read_geotiff(filepath: str) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Reads a 4-band GeoTIFF file.
    Returns:
        data: np.ndarray [C, H, W] in float32
        meta: dictionary of geospatial metadata (CRS, transform, bounds, etc.)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GeoTIFF file not found: {filepath}")

    if HAS_RASTERIO:
        with rasterio.open(filepath) as src:
            data = src.read().astype(np.float32)  # [C, H, W]
            meta = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': 'float32',
                'nodata': src.nodata,
                'bounds': src.bounds
            }
            return data, meta
    else:
        # Fallback for systems without rasterio installed
        from PIL import Image, ImageSequence
        img = Image.open(filepath)
        frames = [np.array(frame, dtype=np.float32) for frame in ImageSequence.Iterator(img)]
        if len(frames) > 1:
            arr = np.stack(frames, axis=0)  # [C, H, W]
        else:
            arr = frames[0]
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            elif arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)  # [C, H, W]
        return arr, None


def write_geotiff(filepath: str, data: np.ndarray, meta: Optional[Dict[str, Any]] = None,
                  upscale_factor: float = 1.0, nodata_val: float = -9999.0):
    """
    Writes a [C, H, W] or [H, W] array to GeoTIFF preserving CRS and updating Affine transform.
    Args:
        filepath: Destination file path
        data: [C, H, W] in float32
        meta: Original metadata from read_geotiff
        upscale_factor: e.g. 4.0 for x4 SR (divides pixel resolution by 4)
        nodata_val: Nodata value
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    C, H, W = data.shape

    if HAS_RASTERIO and meta is not None and meta.get('transform') is not None:
        orig_transform = meta['transform']
        # Update affine transform for super-resolution:
        # pixel size in x and y is divided by upscale_factor
        new_transform = orig_transform * Affine.scale(1.0 / upscale_factor, 1.0 / upscale_factor)
        
        out_meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': nodata_val,
            'width': W,
            'height': H,
            'count': C,
            'crs': meta.get('crs'),
            'transform': new_transform,
            'compress': 'deflate'
        }
        
        with rasterio.open(filepath, 'w', **out_meta) as dst:
            dst.write(data.astype(np.float32))
    else:
        # Fallback rasterio simple or tifffile / PIL
        if HAS_RASTERIO:
            with rasterio.open(
                filepath, 'w',
                driver='GTiff',
                height=H, width=W,
                count=C,
                dtype='float32'
            ) as dst:
                dst.write(data.astype(np.float32))
        else:
            try:
                import tifffile
                tifffile.imwrite(filepath, data.astype(np.float32))
            except ImportError:
                # Save via PIL multipage or numpy array save
                from PIL import Image
                # Convert [C, H, W] to list of band PIL images
                band_imgs = [Image.fromarray(data[c]) for c in range(C)]
                band_imgs[0].save(filepath, save_all=True, append_images=band_imgs[1:])


def normalize_reflectance(data: np.ndarray, max_val: float = 10000.0, clip: bool = True) -> np.ndarray:
    """
    Normalizes Sentinel-2 / NAIP raw reflectance to [0.0, 1.0].
    If data is already in [0, 1] range (max <= 1.5), preserves it.
    If data is 8-bit [0, 255], scales by 255.0.
    If data is 12/16-bit [0, 10000], scales by max_val.
    """
    data = data.astype(np.float32)
    max_in_data = np.nanmax(data) if np.size(data) > 0 else 1.0
    
    if max_in_data <= 1.5:
        norm = data
    elif max_in_data <= 255.0:
        norm = data / 255.0
    else:
        norm = data / max_val

    if clip:
        norm = np.clip(norm, 0.0, 1.0)
    return norm


def denormalize_reflectance(data: np.ndarray, max_val: float = 10000.0, target_range: str = "original") -> np.ndarray:
    """
    Denormalizes model output [0.0, 1.0] back to target range.
    """
    data = np.clip(data, 0.0, 1.0)
    if target_range == "unit":
        return data.astype(np.float32)
    elif target_range == "uint8":
        return (data * 255.0).astype(np.uint8)
    else:
        return (data * max_val).astype(np.float32)


def is_valid_patch(patch: np.ndarray, max_nan_ratio: float = 0.0, nodata_val: Optional[float] = None) -> bool:
    """
    Checks if a patch contains NaNs, Infs, or nodata values.
    Must have 0 invalid values by default to prevent training corruption.
    """
    if np.any(np.isnan(patch)) or np.any(np.isinf(patch)):
        return False
    if nodata_val is not None:
        if np.any(patch == nodata_val):
            return False
    # Check for dead constant black/white images
    if np.all(patch == 0) or np.std(patch) < 1e-6:
        return False
    return True


def extract_patches(lr_img: np.ndarray, hr_img: np.ndarray,
                    lr_patch_size: int = 64, hr_patch_size: int = 256,
                    stride: Optional[int] = None) -> list:
    """
    Extracts aligned LR and HR patch pairs from full images.
    Args:
        lr_img: [C, H_lr, W_lr]
        hr_img: [C, H_hr, W_hr]
        lr_patch_size: size of LR patch (default 64)
        hr_patch_size: size of HR patch (default 256)
        stride: stride in LR pixels (default lr_patch_size for non-overlapping)
    Returns:
        list of (lr_patch, hr_patch)
    """
    assert lr_img.shape[0] == hr_img.shape[0], "Channel count mismatch between LR and HR!"
    scale = hr_patch_size // lr_patch_size
    assert scale == 4, f"Expected 4x scale factor, got {scale}"

    C, H_lr, W_lr = lr_img.shape
    stride = stride or lr_patch_size

    pairs = []
    for y in range(0, H_lr - lr_patch_size + 1, stride):
        for x in range(0, W_lr - lr_patch_size + 1, stride):
            lr_patch = lr_img[:, y:y + lr_patch_size, x:x + lr_patch_size]
            
            y_hr = y * scale
            x_hr = x * scale
            hr_patch = hr_img[:, y_hr:y_hr + hr_patch_size, x_hr:x_hr + hr_patch_size]
            
            if is_valid_patch(lr_patch) and is_valid_patch(hr_patch):
                pairs.append((lr_patch, hr_patch))

    return pairs
