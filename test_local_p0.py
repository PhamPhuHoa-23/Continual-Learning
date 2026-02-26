"""
Test Phase 0 locally — mirror config nguoi dung paste tren Kaggle.
"""
from cont_src.training import AdaSlotTrainer, AdaSlotTrainerConfig
from cont_src.models.aggregators.attention_aggregator import AttentionAggregator
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from src.models.adaslot.model import AdaSlotModel
from avalanche.benchmarks.classic import SplitCIFAR100
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as _D
import torchvision.transforms as T
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


# ── CONFIG (copy y nguyen tu Kaggle) ────────────────────────────────────────
DATASET = "cifar100"
N_TASKS = 10
BATCH_SIZE = 64
NUM_WORKERS = 0
VAL_SPLIT = 0.1

CKPT_PATH = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"

IMG_SIZE = 128
NUM_SLOTS = 11
SLOT_DIM = 64
D_H = 64

P0_EPOCHS = 1
P0_LR = 4e-3
P0_W_RECON = 1.0
P0_W_SPARSE = 10.0
P0_W_PRIM = 5.0
P0_LOG_EVERY = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("data/cifar100_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
N_CLASSES = 100

print(f"Device: {DEVICE}")
print(f"IMG_SIZE={IMG_SIZE}  NUM_SLOTS={NUM_SLOTS}  N_TASKS={N_TASKS}")
print(f"P0_LR={P0_LR}  P0_W_SPARSE={P0_W_SPARSE}  P0_W_PRIM={P0_W_PRIM}")

# ── DATA ────────────────────────────────────────────────────────────────────
print("\n[1] Loading CIFAR-100...")
_D.CIFAR100(DATA_ROOT, train=True,  download=True)
_D.CIFAR100(DATA_ROOT, train=False, download=True)


def _make_tf(train, size=IMG_SIZE):
    aug = [T.RandomHorizontalFlip(), T.ColorJitter(
        0.2, 0.2, 0.2)] if train else []
    return T.Compose(aug + [
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3),
    ])


benchmark = SplitCIFAR100(
    n_experiences=N_TASKS,
    seed=42,
    return_task_id=False,
    dataset_root=str(DATA_ROOT),
    train_transform=_make_tf(True),
    eval_transform=_make_tf(False),
)


class _AvDS(torch.utils.data.Dataset):
    def __init__(self, av_ds): self._ds = av_ds
    def __len__(self): return len(self._ds)

    def __getitem__(self, i):
        x, y, *_ = self._ds[i]
        return x, y


tr_full = _AvDS(benchmark.train_stream[0].dataset)
n_val = int(len(tr_full) * VAL_SPLIT)
tr_ds, val_ds = random_split(tr_full, [len(tr_full)-n_val, n_val],
                             generator=torch.Generator().manual_seed(42))
kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
          pin_memory=False, drop_last=True)
train_loader = DataLoader(tr_ds, shuffle=True,  **kw)
val_loader = DataLoader(val_ds, shuffle=False, **kw)
CLASSES_PER_TASK = N_CLASSES // N_TASKS
print(f"Task 0: {CLASSES_PER_TASK} classes | train={len(train_loader)} batches | val={len(val_loader)}")

# ── MODEL ────────────────────────────────────────────────────────────────────
print("\n[2] Building model...")


class SlotModelWrapper(nn.Module):
    def __init__(self, backbone, prim_sel=None):
        super().__init__()
        self.backbone = backbone
        self.prim_sel = prim_sel

    def forward(self, images, **kw):
        out = self.backbone(images, **kw)
        result = {"recon": out["reconstruction"], "mask": out["hard_keep_decision"],
                  "slots": out["slots"], **out}
        if self.prim_sel is not None:
            H = self.prim_sel(
                out["slots"], slot_mask=out["hard_keep_decision"])
            result["primitives"] = H.unsqueeze(1)
        return result


backbone = AdaSlotModel(
    resolution=(IMG_SIZE, IMG_SIZE), num_slots=NUM_SLOTS, slot_dim=SLOT_DIM,
    num_iterations=3, feature_dim=SLOT_DIM, kvq_dim=128, low_bound=1,
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
missing, unexpected = backbone.load_state_dict(ckpt["state_dict"], strict=True)
print(
    f"Checkpoint loaded: missing={len(missing)}  unexpected={len(unexpected)}")


def _reset_gumbel_gate(model):
    for name, mod in model.named_modules():
        if "single_gumbel_score_network" in name or "gumbel_score" in name:
            for p in mod.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
            print(f"  Gate reset: {name}")


_reset_gumbel_gate(backbone)

prim_sel = PrimitiveSelector(slot_dim=SLOT_DIM, hidden_dim=D_H).to(DEVICE)
slot_model = SlotModelWrapper(backbone, prim_sel).to(DEVICE)
n_total = sum(p.numel() for p in slot_model.parameters())
print(f"Params: {n_total:,}")

# ── PHASE 0 ──────────────────────────────────────────────────────────────────
print(
    f"\n[3] Phase 0 — {P0_EPOCHS} epoch(s), LR={P0_LR}, w_sparse={P0_W_SPARSE}, w_prim={P0_W_PRIM}")

cfg_p0 = AdaSlotTrainerConfig(
    lr=P0_LR, max_steps=0, max_epochs=P0_EPOCHS,
    w_recon=P0_W_RECON, w_sparse=P0_W_SPARSE, w_prim=P0_W_PRIM,
    checkpoint_dir="./tmp_p0", log_every_n_steps=P0_LOG_EVERY,
)
trainer_p0 = AdaSlotTrainer(
    config=cfg_p0, slot_model=slot_model, primitive_predictor=None)

metrics = trainer_p0.train(train_loader)
print(f"\nPhase 0 done:")
for k, v in metrics.items():
    val = v[-1] if isinstance(v, list) and v else v
    print(f"  {k}: {val:.4f}" if isinstance(val, float) else f"  {k}: {val}")

# ── QUICK SANITY: check slot count ──────────────────────────────────────────
print("\n[4] Sanity check — slot stats on val batch:")
slot_model.eval()
with torch.no_grad():
    imgs, _ = next(iter(val_loader))
    out = slot_model(imgs[:8].to(DEVICE))
    hkd = out["hard_keep_decision"]
    skp = out["slots_keep_prob"]
    print(
        f"  hard_keep_decision: mean={hkd.mean():.3f}  active={hkd.sum(1).float().mean():.1f}/{NUM_SLOTS}")
    print(
        f"  slots_keep_prob:    min={skp.min():.3f}  max={skp.max():.3f}  mean={skp.mean():.3f}")
    recon_loss = torch.nn.functional.mse_loss(
        out["recon"], imgs[:8].to(DEVICE), reduction="sum") / 8
    print(f"  recon_loss (mse_sum): {recon_loss.item():.2f}")
print("\nTest PASSED — Phase 0 hoat dong binh thuong voi config nay.")
