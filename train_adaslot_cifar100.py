"""
Train AdaSlot on CIFAR-100 Task 1 with full loss suite:
  L = L_recon (MSE image) + L_sparse (AdaSlot) + L_primitive (concept-level KL)

Uses:
  - AdaSlotModel (src/models/adaslot) - 56/56 pretrained CLEVR10 weights
  - SparsePenalty, ReconstructionLoss, PrimitiveLoss (cont_src/losses)
  - PrimitiveSelector (cont_src/models/slot_attention/primitives)
"""

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from cont_src.losses.losses import SparsePenalty, PrimitiveLoss
from src.models.adaslot.model import AdaSlotModel
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))


# ── Config -------------------------------------------------------------------
CHECKPOINT = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"
BATCH_SIZE = 16
LR = 4e-4
NUM_STEPS = 200         # enough steps to see convergence
IMG_SIZE = 128         # AdaSlot trained on 128x128
SLOT_DIM = 64
PRIMITIVE_DIM = 64          # H dimension = slot_dim

# ── Loss weights ─────────────────────────────────────────────────────────────
# Keep recon/sparse at ORIGINAL AdaSlot ratio (mse_sum raw + linear_weight=10)
# so the model behaves like it did during CLEVR10 training.
# Only primitive is tuned separately to match sparse magnitude:
#   L_sparse at convergence ≈ 10 × 0.5 = 5   (keep_prob → 0.5 target)
#   L_prim   raw            ≈ 1–2
#   → W_PRIM = 5 gives L_prim_weighted ≈ 5–10  (same order as sparse)
W_RECON = 1.0     # mse_sum raw scale — matches original AdaSlot
SPARSE_LINEAR = 10.0    # original AdaSlot value
SPARSE_QUAD = 0.0
SPARSE_BIAS = 0.5
W_PRIM = 5.0     # scale primitive to ~sparse magnitude
TAU_PRIM = 10.0    # temperature for primitive loss

VIZ_OUT = Path("visualizations/recon_after_training.png")


def mse_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """sum over all dims, average over batch only — matches AdaSlot original (ocl/losses.py)."""
    return F.mse_loss(pred, target, reduction="sum") / pred.shape[0]


def save_recon_viz(model, loader, out_path: Path, n: int = 8):
    """Save a side-by-side grid: original | reconstruction after training."""
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        images_small, _ = next(iter(loader))
        images = F.interpolate(images_small, size=(IMG_SIZE, IMG_SIZE),
                               mode="bilinear", align_corners=False).to(DEVICE)
        out = model(images[:n])
        recon = out["reconstruction"][:n]           # (n, 3, H, W)
        active = out["hard_keep_decision"][:n].sum(dim=1).float()  # (n,)

    # denorm [-1,1] → [0,1]
    def to_img(t):
        return (t.cpu().clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).numpy()

    orig_imgs = to_img(images[:n])
    recon_imgs = to_img(recon)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(orig_imgs[i])
        axes[0, i].set_title(f"orig", fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_imgs[i])
        axes[1, i].set_title(f"recon\n({int(active[i])}slot)", fontsize=7)
        axes[1, i].axis("off")
    fig.suptitle(
        f"AdaSlot reconstruction after {NUM_STEPS} steps (CIFAR-100)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved reconstruction viz → {out_path}")
    model.train()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ---------------------------------------------------------------------


def get_loader():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = CIFAR100(root="data", train=True, download=False, transform=transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=True, drop_last=True)

# ── Build model & losses -----------------------------------------------------


def build():
    # AdaSlot backbone (pretrained)
    model = AdaSlotModel(
        resolution=(IMG_SIZE, IMG_SIZE),
        num_slots=11,
        slot_dim=SLOT_DIM,
        num_iterations=3,
        feature_dim=SLOT_DIM,
        kvq_dim=128,
        low_bound=1,
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print("AdaSlotModel: 56/56 keys loaded (strict)")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Primitive selector (new, random init) - aggregates slot_dim -> PRIMITIVE_DIM
    prim_sel = PrimitiveSelector(
        slot_dim=SLOT_DIM,
        hidden_dim=PRIMITIVE_DIM,
    ).to(DEVICE)
    print(
        f"PrimitiveSelector: {sum(p.numel() for p in prim_sel.parameters()):,} params")

    # Losses — recon/sparse ratio kept identical to original AdaSlot training
    loss_sparse = SparsePenalty(linear_weight=SPARSE_LINEAR,
                                quadratic_weight=SPARSE_QUAD, quadratic_bias=SPARSE_BIAS, weight=1.0)
    loss_prim = PrimitiveLoss(temperature=TAU_PRIM, weight=W_PRIM)

    # Optimizer: unfreeze whole AdaSlot + primitive selector
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(prim_sel.parameters()),
        lr=LR,
    )

    return model, prim_sel, loss_sparse, loss_prim, optimizer

# ── Training step ------------------------------------------------------------


def step(model, prim_sel, loss_sparse, loss_prim,
         optimizer, images_small, labels, global_step):

    # Resize 32x32 -> 128x128
    images = F.interpolate(images_small, size=(IMG_SIZE, IMG_SIZE),
                           mode="bilinear", align_corners=False).to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()

    # Forward
    out = model(images, global_step=global_step)

    reconstruction = out["reconstruction"]      # (B, 3, H, W)
    slots_keep_prob = out["slots_keep_prob"]      # (B, K)
    hard_keep = out["hard_keep_decision"]   # (B, K)
    slots = out["slots"]               # (B, K, slot_dim)

    # 1. L_recon: mse_sum (original AdaSlot) = sum over pixels / batch_size
    l_recon = W_RECON * mse_sum(reconstruction, images)

    # 2. L_sparse: Penalize keeping too many slots
    #    AdaSlot config uses hard_keep_decision (Gumbel straight-through -> gradients ok)
    l_sparse = loss_sparse(hard_keep)

    # 3. L_primitive: Aggregate slots -> H, then compute concept KL
    #    Pass hard_keep mask so inactive slots are masked out
    H = prim_sel(slots, slot_mask=hard_keep)     # (B, primitive_dim)
    l_prim = loss_prim(H, labels)

    # Total
    loss = l_recon + l_sparse + l_prim

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(prim_sel.parameters(), 1.0)
    optimizer.step()

    return {
        "total":    loss.item(),
        "recon":    l_recon.item(),
        "sparse":   l_sparse.item(),
        "prim":     l_prim.item(),
        "active_slots": hard_keep.sum(dim=1).float().mean().item(),
    }

# ── Main ---------------------------------------------------------------------


def main():
    print(f"Device: {DEVICE}")
    print("="*70)

    loader = get_loader()
    model, prim_sel, loss_sparse, loss_prim, optimizer = build()

    print(f"\nTraining {NUM_STEPS} steps, batch_size={BATCH_SIZE}")
    print("="*70)
    print(f"{'Step':>5}  {'Total':>8}  {'Recon':>8}  {'Sparse':>8}  {'Prim':>8}  {'ActiveSlots':>12}")
    print("-"*70)

    data_iter = iter(loader)
    log_every = 20

    running = {k: 0.0 for k in ["total", "recon",
                                "sparse", "prim", "active_slots"]}

    for step_i in range(1, NUM_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        images, labels = batch[0], batch[1]

        metrics = step(model, prim_sel, loss_sparse, loss_prim,
                       optimizer, images, labels, global_step=step_i)

        for k in running:
            running[k] += metrics[k]

        if step_i % log_every == 0:
            avg = {k: running[k] / log_every for k in running}
            print(f"{step_i:>5}  "
                  f"{avg['total']:>8.4f}  "
                  f"{avg['recon']:>8.4f}  "
                  f"{avg['sparse']:>8.4f}  "
                  f"{avg['prim']:>8.4f}  "
                  f"{avg['active_slots']:>12.2f}")
            running = {k: 0.0 for k in running}

    print("="*70)
    print("Done.")

    # ── Visualise reconstruction quality ─────────────────────────────────────
    save_recon_viz(model, loader, VIZ_OUT)


if __name__ == "__main__":
    main()
