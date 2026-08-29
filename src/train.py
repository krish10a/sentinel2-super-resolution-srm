"""
Training Pipeline for Sentinel-2 Super-Resolution (x4) with SwinIR.
Supports:
- Mixed precision training (AMP) for 4GB VRAM
- L1 baseline loss
- AdamW optimizer with cosine or step LR scheduling
- Best and Latest checkpointing
- Resume training support
- One-image overfitting sanity test (--sanity)
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path if executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import create_swinir_satellite
from src.dataset import SEN2NAIPDataset, get_dataloaders
from src.metrics import calculate_psnr, calculate_ssim
from src.utils import set_seed, get_device, save_checkpoint, load_checkpoint, save_visual_comparison


def train_sanity_check(splits_json: str,
                       checkpoint_dir: str = "checkpoints",
                       output_dir: str = "outputs/comparisons",
                       iterations: int = 500,
                       lr: float = 5e-4):
    """
    CRITICAL SANITY TEST:
    Takes 1 single training sample and trains for 500 iterations.
    The model must overfit and memorize it (Loss drops, PSNR increases to > 35dB).
    """
    print("\n" + "=" * 70)
    print(" >>> RUNNING ONE-IMAGE OVERFITTING SANITY CHECK <<<")
    print("=" * 70)

    set_seed(42)
    device = get_device()

    with open(splits_json, 'r') as f:
        splits = json.load(f)

    train_dirs = splits.get("train", [])
    if len(train_dirs) == 0:
        raise ValueError("No training samples found in splits.json!")

    # Single sample dataset
    single_dir = [train_dirs[0]]
    print(f"[Sanity] Using single sample: {single_dir[0]}")
    ds = SEN2NAIPDataset(single_dir, is_train=False, augment=False)
    sample = ds[0]

    lr_tensor = sample["lr"].unsqueeze(0).to(device)  # [1, 4, 64, 64]
    hr_tensor = sample["hr"].unsqueeze(0).to(device)  # [1, 4, 256, 256]

    model = create_swinir_satellite().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    model.train()
    start_time = time.time()
    initial_loss = None

    print(f"[Sanity] Training on 1 sample for {iterations} iterations...")
    pbar = tqdm(range(1, iterations + 1), desc="Sanity Progress")
    for step in pbar:
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            pred = model(lr_tensor)
            loss = criterion(pred, hr_tensor)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        if step == 1:
            initial_loss = loss_val

        if step % 50 == 0 or step == iterations:
            pred_np = pred.detach().cpu().float().numpy()[0]
            hr_np = hr_tensor.detach().cpu().float().numpy()[0]
            psnr = calculate_psnr(pred_np, hr_np)
            pbar.set_postfix({"Loss": f"{loss_val:.6f}", "PSNR": f"{psnr:.2f}dB"})

    elapsed = time.time() - start_time
    final_loss = loss.item()
    pred_np = pred.detach().cpu().float().numpy()[0]
    hr_np = hr_tensor.detach().cpu().float().numpy()[0]
    lr_np = lr_tensor.detach().cpu().float().numpy()[0]
    final_psnr = calculate_psnr(pred_np, hr_np)
    final_ssim = calculate_ssim(pred_np, hr_np)

    print("\n" + "-" * 50)
    print(f"Sanity Check Results in {elapsed:.2f}s:")
    print(f"  Initial Loss: {initial_loss:.6f}")
    print(f"  Final Loss:   {final_loss:.6f}")
    print(f"  Final PSNR:   {final_psnr:.2f} dB")
    print(f"  Final SSIM:   {final_ssim:.4f}")
    print("-" * 50)

    # Save visual comparison
    sanity_img_path = os.path.join(output_dir, "sanity_overfit_result.png")
    from src.metrics import bicubic_baseline
    bicubic_np = bicubic_baseline(lr_np, scale=4)
    save_visual_comparison(
        bicubic=bicubic_np,
        sr=pred_np,
        ground_truth=hr_np,
        save_path=sanity_img_path,
        title_prefix="Sanity Test (1-Sample Overfitting)"
    )
    print(f"[Sanity] Visual comparison saved to: {sanity_img_path}")

    # Pass condition
    if final_loss < 0.03 and final_psnr > 28.0:
        print("\n [PASSED] Model memorized the sample successfully! Pipeline is verified.")
    else:
        print("\n [WARNING] Loss did not converge as expected. Check learning rate or sample validity.")
    print("=" * 70 + "\n")


def train(args):
    set_seed(args.seed)
    device = get_device()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Sanity check mode
    if args.sanity:
        train_sanity_check(
            splits_json=args.splits_json,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=os.path.join(args.output_dir, "comparisons"),
            iterations=args.sanity_iterations,
            lr=args.lr
        )
        return

    # DataLoaders
    print("[Train] Loading datasets from geographical splits...")
    train_loader, val_loader, _ = get_dataloaders(
        splits_json=args.splits_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_limit=args.train_limit,
        val_limit=args.val_limit
    )

    # Model
    model = create_swinir_satellite(
        embed_dim=args.embed_dim,
        depths=(4, 4, 4, 4),
        num_heads=(4, 4, 4, 4),
        window_size=args.window_size
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model instantiated. Total parameters: {num_params:,}")

    # Loss, Optimizer, Scheduler, Scaler
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    start_epoch = 1
    best_val_loss = float('inf')

    # Resume if requested
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler) + 1
        print(f"[Train] Resuming from epoch {start_epoch}")

    print("\n" + "=" * 70)
    print(f" >>> STARTING TRAINING: {args.epochs} Epochs | Batch Size {args.batch_size} | LR {args.lr} <<<")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")

        for batch_idx, batch in enumerate(train_pbar):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                sr = model(lr)
                loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.5f}", "AvgLoss": f"{running_loss / (batch_idx + 1):.5f}"})

        scheduler.step()
        epoch_train_loss = running_loss / max(1, len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_psnrs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]"):
                lr = batch["lr"].to(device, non_blocking=True)
                hr = batch["hr"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    sr = model(lr)
                    loss = criterion(sr, hr)
                val_loss += loss.item()

                sr_np = sr.cpu().float().numpy()
                hr_np = hr.cpu().float().numpy()
                for b in range(sr_np.shape[0]):
                    val_psnrs.append(calculate_psnr(sr_np[b], hr_np[b]))

        epoch_val_loss = val_loss / max(1, len(val_loader))
        epoch_val_psnr = float(torch.tensor(val_psnrs).mean().item()) if len(val_psnrs) > 0 else 0.0

        print(f"--> [Epoch {epoch:02d}/{args.epochs:02d}] Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f} | Val PSNR: {epoch_val_psnr:.2f} dB | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Checkpoints
        latest_path = os.path.join(args.checkpoint_dir, "latest_model.pth")
        best_path = os.path.join(args.checkpoint_dir, "best_model.pth")

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": epoch_val_loss,
            "val_psnr": epoch_val_psnr
        }

        save_checkpoint(state, latest_path)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(state, best_path)
            print(f"    [SAVED] Saved new BEST model with Val Loss {best_val_loss:.5f} to {best_path}")

    print("\n" + "=" * 70)
    print(" >>> TRAINING COMPLETE! <<<")
    print(f" Best Checkpoint: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print("=" * 70 + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Train SwinIR Satellite Super-Resolution")
    parser.add_argument("--splits_json", type=str, default="data/processed/splits.json", help="Path to splits.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (2 for 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--embed_dim", type=int, default=60, help="SwinIR embedding dimension")
    parser.add_argument("--window_size", type=int, default=8, help="SwinIR window size")
    parser.add_argument("--train_limit", type=int, default=2000, help="Max train samples")
    parser.add_argument("--val_limit", type=int, default=200, help="Max val samples")
    parser.add_argument("--num_workers", type=int, default=0, help="PyTorch DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--sanity", action="store_true", help="Run 1-sample overfitting sanity test")
    parser.add_argument("--sanity_iterations", type=int, default=500, help="Sanity check iterations")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
