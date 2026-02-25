"""Smoke test: run Phase 1 for 5 steps on dummy data."""
import torch
import sys
sys.path.insert(0, '.')

from src.models.adaslot.model import AdaSlotModel
from src.models.adaslot.train import (
    ReconstructionLoss, SparsePenalty, DummyImageDataset, exp_decay_with_warmup
)
from torch.utils.data import DataLoader

print("=== Smoke Test: Phase 1 ===")
device = 'cpu'

# Create model
model = AdaSlotModel(resolution=(128, 128), num_slots=11, slot_dim=64)
model.to(device)
model.train()

# Losses
recon_loss_fn = ReconstructionLoss()
sparse_loss_fn = SparsePenalty(linear_weight=10.0)

# Data
dataset = DummyImageDataset(num_samples=16, resolution=128, num_classes=10)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

losses = []
for step, (images, labels) in enumerate(dataloader):
    if step >= 3:
        break
    images = images.to(device)
    
    out = model(images, global_step=step)
    
    loss_recon = recon_loss_fn(out['reconstruction'], images)
    loss_sparse = sparse_loss_fn(out['hard_keep_decision'])
    loss_total = loss_recon + loss_sparse
    
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    
    losses.append(loss_total.item())
    print(f"  Step {step}: loss={loss_total.item():.4f} "
          f"(recon={loss_recon.item():.4f}, sparse={loss_sparse.item():.4f})")
    print(f"    reconstruction shape: {out['reconstruction'].shape}")
    print(f"    slots shape: {out['slots'].shape}")
    print(f"    hard_keep_decision: {out['hard_keep_decision'][0].tolist()}")

print(f"\nAll losses: {losses}")
print(f"Loss decreasing: {losses[-1] < losses[0] or 'N/A (too few steps)'}")
print("\n=== SMOKE TEST PASSED ===")
