# Model Checkpoints

This directory holds the trained PyTorch model checkpoints.

## Phase 1 Production Checkpoint

The final trained weights of the **Residual SwinIR $\times 4$** model are saved in:
- **File**: `checkpoints/best_residual_swinir_100ep.pth`
- **Size**: 12.66 MB

This file is tracked using **Git LFS** (Git Large File Storage) due to its binary format.

### How to Recreate/Retrain the Model

To retrain the model from scratch and generate new checkpoints, run:
```powershell
python -m src.train --epochs 100 --batch_size 2 --lr 4e-4
```
The training script will automatically save checkpoints inside this directory:
- `best_residual_swinir_100ep.pth`: Checkpoint with the lowest validation L1 loss.
- `latest_residual_swinir_100ep.pth`: The checkpoint from the most recent training epoch.
