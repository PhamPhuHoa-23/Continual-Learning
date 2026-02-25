"""Quick local test of Phase 0 AdaSlot fine-tune with user's config."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch, torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from pathlib import Path

# ---- User's config ----
IMG_SIZE   = 128
NUM_SLOTS  = 11
SLOT_DIM   = 64
D_H        = 64
BATCH_SIZE = 16       # small for local test
N_STEPS    = 20       # just 20 steps
P0_LR      = 4e-3
P0_W_RECON = 1.0
P0_W_SPARSE= 10.0
P0_W_PRIM  = 5.0
CKPT_PATH  = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT  = Path("data/cifar-100-python")

print(f"Device: {DEVICE}  IMG_SIZE={IMG_SIZE}  NUM_SLOTS={NUM_SLOTS}")

# ---- Data ----
tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
ds = CIFAR100(DATA_ROOT, train=True, download=True, transform=tf)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

# ---- Model ----
from src.models.adaslot.model import AdaSlotModel
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from cont_src.training import AdaSlotTrainer, AdaSlotTrainerConfig

backbone = AdaSlotModel(
    resolution=(IMG_SIZE, IMG_SIZE), num_slots=NUM_SLOTS, slot_dim=SLOT_DIM,
    num_iterations=3, feature_dim=SLOT_DIM, kvq_dim=128, low_bound=1,
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
backbone.load_state_dict(ckpt["state_dict"], strict=True)
print("Checkpoint loaded OK")

# Reset gate
for nm, m in backbone.named_modules():
    if "single_gumbel_score_network" in nm:
        for p in m.parameters():
            if p.dim() >= 2: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
print("Gate reset OK")

prim_sel = PrimitiveSelector(slot_dim=SLOT_DIM, hidden_dim=D_H).to(DEVICE)

class Wrapper(nn.Module):
    def __init__(self, bb, ps): super().__init__(); self.backbone=bb; self.prim_sel=ps
    def forward(self, imgs, **kw):
        out = self.backbone(imgs, **kw)
        r = {"recon": out["reconstruction"], "mask": out["hard_keep_decision"], "slots": out["slots"], **out}
        H = self.prim_sel(out["slots"], slot_mask=out["hard_keep_decision"])
        r["primitives"] = H.unsqueeze(1)
        return r

slot_model = Wrapper(backbone, prim_sel).to(DEVICE)

# ---- Train ----
cfg = AdaSlotTrainerConfig(
    lr=P0_LR, max_steps=N_STEPS, max_epochs=0,
    w_recon=P0_W_RECON, w_sparse=P0_W_SPARSE, w_prim=P0_W_PRIM,
    checkpoint_dir="/tmp/test_p0", log_every_n_steps=5,
)
trainer = AdaSlotTrainer(config=cfg, slot_model=slot_model, primitive_predictor=None)

print(f"\nRunning {N_STEPS} steps...")
metrics = trainer.train(loader)

print("\n=== Results ===")
for k, v in metrics.items():
    if isinstance(v, list) and v:
        print(f"  {k}: first={v[0]:.3f}  last={v[-1]:.3f}")
    elif isinstance(v, (int, float)):
        print(f"  {k}: {v:.4f}")
print("\nPhase 0 test PASSED")
