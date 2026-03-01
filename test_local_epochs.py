"""
Local multi-epoch test on CIFAR-100 task 0 — mirrors Kaggle pipeline exactly.
Prints per-epoch stats and image/recon range to diagnose issues.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
from cont_src.training import AdaSlotTrainer, AdaSlotTrainerConfig
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from src.models.adaslot.model import AdaSlotModel
from torch.utils.data import Subset
from pathlib import Path
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = 64       # dataloader size (trainer will upscale to 128)
NUM_SLOTS = 11
SLOT_DIM = 64
D_H = 64
BATCH_SIZE = 32
N_EPOCHS = 3
P0_LR = 1e-3
P0_W_RECON = 1.0
P0_W_SPARSE = 10.0
P0_W_PRIM = 1.0
CKPT_PATH = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE} | IMG_SIZE={IMG_SIZE} | EPOCHS={N_EPOCHS}")

# ── Data — same normalisation as Kaggle ─────────────────────────────────────
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # → [-1, 1]
])
ds = CIFAR100("data", train=True, download=True, transform=tf)
# Use only first 10 classes (task 0) to mirror Kaggle N_TASKS=10 task split
task0_idx = [i for i, (_, y) in enumerate(ds) if y < 10][:2000]
ds_task0 = Subset(ds, task0_idx)
loader = DataLoader(ds_task0, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=0, drop_last=True)
print(f"Task-0 samples: {len(ds_task0)}  batches/epoch: {len(loader)}")

# ── Model ────────────────────────────────────────────────────────────────────

backbone = AdaSlotModel(
    resolution=(128, 128), num_slots=NUM_SLOTS, slot_dim=SLOT_DIM,
    num_iterations=3, feature_dim=SLOT_DIM, kvq_dim=128, low_bound=1,
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
backbone.load_state_dict(ckpt["state_dict"], strict=True)
print("Checkpoint loaded (56/56 keys)")

# Reset saturated Gumbel gate
for nm, m in backbone.named_modules():
    if "single_gumbel_score_network" in nm:
        for p in m.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
print("Gate reset OK")

prim_sel = PrimitiveSelector(slot_dim=SLOT_DIM, hidden_dim=D_H).to(DEVICE)


class Wrapper(nn.Module):
    def __init__(self, bb, ps):
        super().__init__()
        self.backbone = bb
        self.prim_sel = ps

    def forward(self, imgs, **kw):
        out = self.backbone(imgs, **kw)
        r = {"recon": out["reconstruction"], "mask": out["hard_keep_decision"],
             "slots": out["slots"], **out}
        H = self.prim_sel(out["slots"], slot_mask=out["hard_keep_decision"])
        r["primitives"] = H.unsqueeze(1)
        return r


slot_model = Wrapper(backbone, prim_sel).to(DEVICE)

# ── Diagnostic: check image & recon range before training ───────────────────
with torch.no_grad():
    imgs_sample, _ = next(iter(loader))
    imgs_128 = F.interpolate(imgs_sample.to(DEVICE), size=(
        128, 128), mode="bilinear", align_corners=False)
    out_sample = slot_model(imgs_128)
    recon_sample = out_sample["recon"]
    print(f"\nDiagnostics:")
    print(
        f"  Input range:  [{imgs_sample.min():.3f}, {imgs_sample.max():.3f}]  shape={tuple(imgs_sample.shape)}")
    print(
        f"  Input128 range: [{imgs_128.min():.3f}, {imgs_128.max():.3f}]  shape={tuple(imgs_128.shape)}")
    print(
        f"  Recon range:  [{recon_sample.min():.3f}, {recon_sample.max():.3f}]  shape={tuple(recon_sample.shape)}")
    raw_mse = F.mse_loss(recon_sample, imgs_128,
                         reduction="sum") / imgs_128.shape[0]
    print(
        f"  mse_sum (per image): {raw_mse:.1f}  (expected ~5000-20000 for [-1,1] images)")
    print(
        f"  hard_keep_decision mean: {out_sample['hard_keep_decision'].mean():.3f}")
    print()

# ── Train ────────────────────────────────────────────────────────────────────
cfg = AdaSlotTrainerConfig(
    lr=P0_LR, max_steps=0, max_epochs=N_EPOCHS,
    w_recon=P0_W_RECON, w_sparse=P0_W_SPARSE, w_prim=P0_W_PRIM,
    checkpoint_dir="/tmp/test_epochs", log_every_n_steps=10,
)
trainer = AdaSlotTrainer(config=cfg, slot_model=slot_model)

print(f"Training {N_EPOCHS} epochs ({len(loader)} batches each)...\n")
metrics = trainer.train(loader)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL METRICS")
print("="*60)
for k, v in sorted(metrics.items()):
    if isinstance(v, list) and v:
        print(f"  {k:25s}: first={v[0]:10.2f}  last={v[-1]:10.2f}  "
              f"{'↓ IMPROVING' if v[-1] < v[0] else '↑ DIVERGING'}")
    elif isinstance(v, (int, float)):
        print(f"  {k:25s}: {v:.4f}")
