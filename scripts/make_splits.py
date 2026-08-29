"""
Geographical Dataset Splitting Script.
Strictly separates dataset by Geographical Region / ROI to prevent spatial data leakage.
Splits are written to data/processed/splits.json.
"""

import os
import sys
import argparse
import json
import random
from collections import defaultdict

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_geographical_splits(data_dir: str,
                             output_splits_json: str,
                             target_train: int = 2000,
                             target_val: int = 200,
                             target_test: int = 200,
                             seed: int = 42):
    """
    Groups samples by ROI and assigns entire ROIs to Train, Val, or Test.
    """
    random.seed(seed)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Discover all sample directories
    entries = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith("sample_")])
    if len(entries) == 0:
        raise ValueError(f"No sample_XXXXX directories found in {data_dir}")

    print(f"[Splits] Found {len(entries)} total samples in {data_dir}")

    # Group by ROI
    roi_groups = defaultdict(list)
    for sample_path in entries:
        meta_file = os.path.join(sample_path, "meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    roi_id = meta.get("roi_id", "DEFAULT_ROI")
            except Exception:
                roi_id = "DEFAULT_ROI"
        else:
            # Fallback: cluster by integer range of sample numbers (e.g. 200 consecutive samples per ROI)
            sample_num = int(os.path.basename(sample_path).replace("sample_", ""))
            roi_id = f"ROI_{(sample_num - 1) // 200:03d}"

        roi_groups[roi_id].append(sample_path)

    all_rois = list(roi_groups.keys())
    random.shuffle(all_rois)
    print(f"[Splits] Clustered into {len(all_rois)} unique geographical ROIs: {all_rois}")

    train_samples = []
    val_samples = []
    test_samples = []

    # Assign entire ROIs
    for roi in all_rois:
        samples = roi_groups[roi]
        if len(test_samples) < target_test:
            test_samples.extend(samples)
            print(f"  --> Assigned ROI '{roi}' ({len(samples)} samples) to TEST")
        elif len(val_samples) < target_val:
            val_samples.extend(samples)
            print(f"  --> Assigned ROI '{roi}' ({len(samples)} samples) to VAL")
        else:
            train_samples.extend(samples)
            print(f"  --> Assigned ROI '{roi}' ({len(samples)} samples) to TRAIN")

    # Trim to exact target limits if needed
    train_samples = train_samples[:target_train]
    val_samples = val_samples[:target_val]
    test_samples = test_samples[:target_test]

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "metadata": {
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "num_test": len(test_samples),
            "total_rois": len(all_rois),
            "seed": seed
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_splits_json)), exist_ok=True)
    with open(output_splits_json, 'w') as f:
        json.dump(splits, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[Splits] Successfully saved geographical splits to: {output_splits_json}")
    print(f"  Train samples: {len(train_samples)} (from dedicated Train ROIs)")
    print(f"  Val samples:   {len(val_samples)} (from dedicated Val ROIs)")
    print(f"  Test samples:  {len(test_samples)} (from dedicated Test ROIs)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create Geographical Train/Val/Test Splits")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory of processed samples")
    parser.add_argument("--output_splits", type=str, default="data/processed/splits.json", help="Path to write splits.json")
    parser.add_argument("--train", type=int, default=200, help="Target number of training pairs")
    parser.add_argument("--val", type=int, default=50, help="Target number of validation pairs")
    parser.add_argument("--test", type=int, default=50, help="Target number of test pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ROI shuffle")
    args = parser.parse_args()

    make_geographical_splits(
        data_dir=args.data_dir,
        output_splits_json=args.output_splits,
        target_train=args.train,
        target_val=args.val,
        target_test=args.test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
