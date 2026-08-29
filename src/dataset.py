"""
PyTorch Dataset and DataLoader for SEN2NAIPv2 Satellite Super-Resolution (x4).
Supports lazy loading from disk, geographical split filtering, on-the-fly patch cropping,
data augmentations (rotations/flips), and normalization.
"""

import os
import json
import random
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import read_geotiff, normalize_reflectance, is_valid_patch


class SEN2NAIPDataset(Dataset):
    """
    SEN2NAIPv2 Dataset for 10m -> 2.5m Super-Resolution (4 bands: B04, B03, B02, B08).
    
    Directory layout per sample:
        <sample_dir>/
            lr.tif (10m 4-band GeoTIFF, e.g. 64x64 or larger)
            hr.tif (2.5m 4-band GeoTIFF, e.g. 256x256 or larger)
    """

    def __init__(self,
                 sample_dirs: List[str],
                 lr_patch_size: int = 64,
                 hr_patch_size: int = 256,
                 is_train: bool = True,
                 augment: bool = True,
                 max_samples: Optional[int] = None):
        super().__init__()
        self.sample_dirs = sample_dirs
        if max_samples is not None and max_samples > 0:
            self.sample_dirs = self.sample_dirs[:max_samples]

        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = hr_patch_size
        self.scale = hr_patch_size // lr_patch_size
        assert self.scale == 4, f"Expected 4x scale factor, got {self.scale}"
        self.is_train = is_train
        self.augment = augment and is_train

        # Verify samples exist
        self.valid_samples = []
        for s in self.sample_dirs:
            lr_path = os.path.join(s, "lr.tif")
            hr_path = os.path.join(s, "hr.tif")
            if os.path.exists(lr_path) and os.path.exists(hr_path):
                self.valid_samples.append((lr_path, hr_path))
            else:
                # Check for alternative naming conventions
                if os.path.exists(os.path.join(s, "LR.tif")) and os.path.exists(os.path.join(s, "HR.tif")):
                    self.valid_samples.append((os.path.join(s, "LR.tif"), os.path.join(s, "HR.tif")))

        if len(self.valid_samples) == 0:
            raise ValueError(f"No valid LR/HR pairs found in provided {len(sample_dirs)} sample directories!")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def _crop_pair(self, lr_img: np.ndarray, hr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crops an aligned [4, lr_patch_size, lr_patch_size] and [4, hr_patch_size, hr_patch_size]."""
        C, H_lr, W_lr = lr_img.shape
        _, H_hr, W_hr = hr_img.shape

        # If already exact size, return directly
        if H_lr == self.lr_patch_size and W_lr == self.lr_patch_size:
            return lr_img, hr_img

        max_y_lr = max(0, H_lr - self.lr_patch_size)
        max_x_lr = max(0, W_lr - self.lr_patch_size)

        if self.is_train:
            y_lr = random.randint(0, max_y_lr)
            x_lr = random.randint(0, max_x_lr)
        else:
            y_lr = max_y_lr // 2
            x_lr = max_x_lr // 2

        lr_crop = lr_img[:, y_lr:y_lr + self.lr_patch_size, x_lr:x_lr + self.lr_patch_size]
        
        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale
        hr_crop = hr_img[:, y_hr:y_hr + self.hr_patch_size, x_hr:x_hr + self.hr_patch_size]

        return lr_crop, hr_crop

    def _apply_augmentations(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronized spatial augmentations: horizontal flip, vertical flip, rot90."""
        # Horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()
        
        # Vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()

        # Random 90-degree rotations
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            lr = np.rot90(lr, k=rot_k, axes=(1, 2)).copy()
            hr = np.rot90(hr, k=rot_k, axes=(1, 2)).copy()

        return lr, hr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lr_path, hr_path = self.valid_samples[idx]

        # Lazy loading of GeoTIFFs
        lr_raw, lr_meta = read_geotiff(lr_path)
        hr_raw, hr_meta = read_geotiff(hr_path)

        # Ensure 4 channels
        assert lr_raw.shape[0] >= 4, f"LR must have >= 4 channels (got {lr_raw.shape[0]}) in {lr_path}"
        assert hr_raw.shape[0] >= 4, f"HR must have >= 4 channels (got {hr_raw.shape[0]}) in {hr_path}"

        lr_raw = lr_raw[0:4, :, :]
        hr_raw = hr_raw[0:4, :, :]

        # Normalize reflectance to [0.0, 1.0]
        lr_norm = normalize_reflectance(lr_raw)
        hr_norm = normalize_reflectance(hr_raw)

        # Crop patch pair
        lr_patch, hr_patch = self._crop_pair(lr_norm, hr_norm)

        # Clean NaNs or Infs if any slipped through
        lr_patch = np.nan_to_num(lr_patch, nan=0.0, posinf=1.0, neginf=0.0)
        hr_patch = np.nan_to_num(hr_patch, nan=0.0, posinf=1.0, neginf=0.0)

        # Augmentations
        if self.augment:
            lr_patch, hr_patch = self._apply_augmentations(lr_patch, hr_patch)

        # Convert to PyTorch tensors [4, H, W]
        lr_tensor = torch.from_numpy(lr_patch.astype(np.float32))
        hr_tensor = torch.from_numpy(hr_patch.astype(np.float32))

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "sample_dir": os.path.dirname(lr_path),
            "lr_path": lr_path,
            "hr_path": hr_path
        }


def get_dataloaders(splits_json: str,
                    batch_size: int = 2,
                    num_workers: int = 0,
                    train_limit: Optional[int] = 2000,
                    val_limit: Optional[int] = 200,
                    test_limit: Optional[int] = 200) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test DataLoaders from a geographical splits.json file.
    """
    if not os.path.exists(splits_json):
        raise FileNotFoundError(f"Splits file not found: {splits_json}")

    with open(splits_json, 'r') as f:
        splits = json.load(f)

    train_dirs = splits.get("train", [])
    val_dirs = splits.get("val", [])
    test_dirs = splits.get("test", [])

    print(f"[Dataset] Found splits - Train: {len(train_dirs)}, Val: {len(val_dirs)}, Test: {len(test_dirs)}")

    train_ds = SEN2NAIPDataset(train_dirs, is_train=True, augment=True, max_samples=train_limit)
    val_ds = SEN2NAIPDataset(val_dirs, is_train=False, augment=False, max_samples=val_limit)
    test_ds = SEN2NAIPDataset(test_dirs, is_train=False, augment=False, max_samples=test_limit)

    print(f"[Dataset] Using subset - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if len(train_ds) > batch_size else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
