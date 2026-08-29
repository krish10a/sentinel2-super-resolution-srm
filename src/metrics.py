"""
Evaluation Metrics for Satellite Super-Resolution (x4).
Includes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- RMSE (Root Mean Squared Error)
- SAM (Spectral Angle Mapper in degrees)
Calculates metrics Overall, RGB-only (Bands 0,1,2 = B04,B03,B02), and NIR-only (Band 3 = B08).
"""

from typing import Dict, Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts tensor [C, H, W] or [B, C, H, W] to float32 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.astype(np.float32)


def calculate_psnr(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    Shapes: [C, H, W] or [H, W]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    mse = np.mean((pred - target) ** 2)
    if mse == 0 or mse < 1e-12:
        return 100.0
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], max_val: float = 1.0, win_size: int = 11) -> float:
    """
    Computes Structural Similarity Index (SSIM) per channel and averages.
    Shapes: [C, H, W] or [H, W]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    if pred.ndim == 2:
        pred = pred[np.newaxis, :, :]
        target = target[np.newaxis, :, :]

    C, H, W = pred.shape
    ssims = []
    
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    for c in range(C):
        img1 = pred[c]
        img2 = target[c]
        
        # Mean filters
        mu1 = uniform_filter(img1, size=win_size)
        mu2 = uniform_filter(img2, size=win_size)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = uniform_filter(img1 * img1, size=win_size) - mu1_sq
        sigma2_sq = uniform_filter(img2 * img2, size=win_size) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, size=win_size) - mu1_mu2
        
        # SSIM map
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
        ssims.append(np.mean(ssim_map))

    return float(np.mean(ssims))


def calculate_rmse(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Computes Root Mean Squared Error (RMSE).
    Shapes: [C, H, W] or [H, W]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    mse = np.mean((pred - target) ** 2)
    return float(np.sqrt(mse))


def calculate_sam(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], eps: float = 1e-8) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees.
    Measures the angle between pixel spectra across channels.
    Lower is better (0 degrees is perfect spectral preservation).
    Shapes: [C, H, W]
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    if pred.ndim == 2:
        return 0.0
    
    # [C, H, W] -> [H*W, C]
    C, H, W = pred.shape
    p = pred.reshape(C, -1).T
    t = target.reshape(C, -1).T

    # Dot product along spectral dimension
    dot = np.sum(p * t, axis=1)
    norm_p = np.linalg.norm(p, axis=1)
    norm_t = np.linalg.norm(t, axis=1)

    denom = norm_p * norm_t
    valid_mask = denom > eps

    if np.sum(valid_mask) == 0:
        return 0.0

    cos_theta = dot[valid_mask] / denom[valid_mask]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    angles_deg = np.rad2deg(angles_rad)

    return float(np.mean(angles_deg))


def compute_all_metrics(pred: Union[torch.Tensor, np.ndarray],
                        target: Union[torch.Tensor, np.ndarray],
                        max_val: float = 1.0) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics:
    - Overall (4 channels)
    - RGB (Channels 0, 1, 2)
    - NIR (Channel 3)
    """
    p = to_numpy(pred)
    t = to_numpy(target)

    # Clip predictions to valid unit range
    p = np.clip(p, 0.0, max_val)
    t = np.clip(t, 0.0, max_val)

    if p.ndim == 4:
        p = p[0]
    if t.ndim == 4:
        t = t[0]

    assert p.shape == t.shape, f"Shape mismatch: {p.shape} vs {t.shape}"
    assert p.shape[0] == 4, f"Expected 4 channels, got {p.shape[0]}"

    # 1. Overall metrics (All 4 bands)
    psnr_all = calculate_psnr(p, t, max_val=max_val)
    ssim_all = calculate_ssim(p, t, max_val=max_val)
    rmse_all = calculate_rmse(p, t)
    sam_all = calculate_sam(p, t)

    # 2. RGB-only metrics (Bands 0, 1, 2 = Red, Green, Blue)
    p_rgb = p[0:3, :, :]
    t_rgb = t[0:3, :, :]
    psnr_rgb = calculate_psnr(p_rgb, t_rgb, max_val=max_val)
    ssim_rgb = calculate_ssim(p_rgb, t_rgb, max_val=max_val)
    rmse_rgb = calculate_rmse(p_rgb, t_rgb)
    sam_rgb = calculate_sam(p_rgb, t_rgb)

    # 3. NIR-only metrics (Band 3 = NIR)
    p_nir = p[3:4, :, :]
    t_nir = t[3:4, :, :]
    psnr_nir = calculate_psnr(p_nir, t_nir, max_val=max_val)
    ssim_nir = calculate_ssim(p_nir, t_nir, max_val=max_val)
    rmse_nir = calculate_rmse(p_nir, t_nir)

    return {
        "psnr_overall": psnr_all,
        "ssim_overall": ssim_all,
        "rmse_overall": rmse_all,
        "sam_overall_deg": sam_all,
        "psnr_rgb": psnr_rgb,
        "ssim_rgb": ssim_rgb,
        "rmse_rgb": rmse_rgb,
        "sam_rgb_deg": sam_rgb,
        "psnr_nir": psnr_nir,
        "ssim_nir": ssim_nir,
        "rmse_nir": rmse_nir
    }


def bicubic_baseline(lr: Union[torch.Tensor, np.ndarray], scale: int = 4) -> Union[torch.Tensor, np.ndarray]:
    """
    Standard bicubic baseline interpolation x4.
    """
    is_numpy = isinstance(lr, np.ndarray)
    if is_numpy:
        t_lr = torch.from_numpy(lr)
    else:
        t_lr = lr

    orig_dim = t_lr.ndim
    if orig_dim == 3:
        t_lr = t_lr.unsqueeze(0)  # [1, C, H, W]

    hr_bicubic = F.interpolate(t_lr, scale_factor=scale, mode='bicubic', align_corners=False)
    hr_bicubic = torch.clamp(hr_bicubic, 0.0, 1.0)

    if orig_dim == 3:
        hr_bicubic = hr_bicubic.squeeze(0)

    if is_numpy:
        return hr_bicubic.numpy()
    return hr_bicubic
