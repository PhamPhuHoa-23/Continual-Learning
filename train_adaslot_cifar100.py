"""
Train AdaSlot on CIFAR-100 Task 1 with full loss suite:
  L = L_recon (MSE image) + L_sparse (AdaSlot, optional) + L_primitive

CLI args:
  --checkpoint    path to .ckpt  (default: CLEVR10)
  --num_steps     training steps (default: 200)
  --batch_size    batch size     (default: 16)
  --lr            learning rate  (default: 4e-4)
  --no_dynamic_slots   disable Gumbel gate (all slots kept, no sparse loss)
  --use_avalanche      use Avalanche SplitCIFAR100 task-1 data

Uses:
  - AdaSlotModel (src/models/adaslot) - pretrained CLEVR10 weights
  - SparsePenalty, PrimitiveLoss (cont_src/losses)
  - PrimitiveSelector (cont_src/models/slot_attention/primitives)
"""

import argparse

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from cont_src.losses.losses import SparsePenalty, PrimitiveLoss
from cont_src.models.adaslot_configs import build_adaslot_from_checkpoint, get_adaslot_config
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


# ── CLI args -----------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",  default="checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt")
    p.add_argument("--num_steps",   type=int, default=200)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--lr",          type=float, default=4e-4)
    p.add_argument("--no_dynamic_slots", action="store_true",
                   help="Disable Gumbel gate: all slots kept, sparse loss skipped")
    p.add_argument("--use_avalanche", action="store_true",
                   help="Use Avalanche SplitCIFAR100 for task-1 data")
    return p.parse_args()


# ── Config (constants kept for backward-compat, overridden by args) ---------
CHECKPOINT = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"
BATCH_SIZE = 16
LR = 4e-4
NUM_STEPS = 200
IMG_SIZE = 128
SLOT_DIM = 64
PRIMITIVE_DIM = 64

W_RECON = 1.0
SPARSE_LINEAR = 10.0
SPARSE_QUAD = 0.0
SPARSE_BIAS = 0.5
W_PRIM = 10.0
TAU_PRIM = 10.0

VIZ_OUT = Path("visualizations/recon_after_training.png")


def mse_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum over all dims, average over batch only — matches AdaSlot original."""
    return F.mse_loss(pred, target, reduction="sum") / pred.shape[0]


def save_recon_viz(model, loader, out_path: Path, img_size: int = IMG_SIZE, n: int = 8):
    """Save a side-by-side grid: original | reconstruction after training."""
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(loader))
        images_small = batch[0]
        images = F.interpolate(images_small, size=(img_size, img_size),
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


def get_loader(batch_size: int = BATCH_SIZE, use_avalanche: bool = False):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if use_avalanche:
        from avalanche.benchmarks.classic import SplitCIFAR100
        bench = SplitCIFAR100(
            n_experiences=10, seed=42,
            train_transform=transform, eval_transform=transform,
            dataset_root="./data",
        )
        task1_ds = bench.train_stream[0].dataset
        print(f"  Avalanche task-1: {len(task1_ds)} samples, "
              f"classes {bench.train_stream[0].classes_in_this_experience}")
        return DataLoader(task1_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    ds = CIFAR100(root="data", train=True, download=False, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=True, drop_last=True)

# ── Build model & losses -----------------------------------------------------


def _ckpt_name_from_path(path: str) -> str:
    """CLEVR10.ckpt -> 'clevr10', MOVi-C.ckpt -> 'movic', etc."""
    stem = Path(path).stem          # e.g. 'MOVi-C'
    return stem.lower().replace("-", "")   # 'movic'


def build(checkpoint: str = CHECKPOINT, lr: float = LR):
    """Build model via registry — arch dims auto-detected from checkpoint name.
    Returns: model, prim_sel, loss_sparse, loss_prim, optimizer, cfg
    """
    ckpt_name = _ckpt_name_from_path(checkpoint)
    cfg = get_adaslot_config(ckpt_name)

    # AdaSlot backbone (pretrained) — uses registry config
    model = build_adaslot_from_checkpoint(
        checkpoint_name=ckpt_name,
        ckpt_path=checkpoint,
        device=str(DEVICE),
        strict_load=True,
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    slot_dim = cfg.slot_dim

    # Primitive selector (new, random init)
    prim_sel = PrimitiveSelector(
        slot_dim=slot_dim,
        hidden_dim=slot_dim,
    ).to(DEVICE)
    print(f"PrimitiveSelector: {sum(p.numel() for p in prim_sel.parameters()):,} params")

    # Losses
    loss_sparse = SparsePenalty(linear_weight=SPARSE_LINEAR,
                                quadratic_weight=SPARSE_QUAD, quadratic_bias=SPARSE_BIAS, weight=1.0)
    loss_prim = PrimitiveLoss(temperature=TAU_PRIM, weight=W_PRIM)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(prim_sel.parameters()),
        lr=lr,
    )

    return model, prim_sel, loss_sparse, loss_prim, optimizer, cfg

# ── Training step ------------------------------------------------------------


def step(model, prim_sel, loss_sparse, loss_prim,
         optimizer, images_small, labels, global_step,
         dynamic_slots: bool = True, img_size: int = IMG_SIZE):

    # Resize CIFAR 32x32 -> model resolution
    images = F.interpolate(images_small, size=(img_size, img_size),
                           mode="bilinear", align_corners=False).to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()

    # Forward
    out = model(images, global_step=global_step, dynamic_slots=dynamic_slots)

    reconstruction = out["reconstruction"]      # (B, 3, H, W)
    slots_keep_prob = out["slots_keep_prob"]     # (B, K)
    hard_keep = out["hard_keep_decision"]        # (B, K)
    slots = out["slots"]                         # (B, K, slot_dim)

    # 1. L_recon
    l_recon = W_RECON * mse_sum(reconstruction, images)

    # 2. L_sparse — skipped when all slots are fixed (would just penalise all-ones)
    if dynamic_slots:
        l_sparse = loss_sparse(hard_keep)
    else:
        l_sparse = torch.tensor(0.0, device=DEVICE)

    # 3. L_primitive
    H = prim_sel(slots, slot_mask=hard_keep)     # (B, primitive_dim)
    l_prim = loss_prim(H, labels)

    loss = l_recon + l_sparse + l_prim

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(prim_sel.parameters(), 1.0)
    optimizer.step()

    return {
        "total":        loss.item(),
        "recon":        l_recon.item(),
        "sparse":       l_sparse.item(),
        "prim":         l_prim.item(),
        "active_slots": hard_keep.sum(dim=1).float().mean().item(),
    }

# ── Main ---------------------------------------------------------------------


def main():
    args = parse_args()
    dynamic = not args.no_dynamic_slots

    print(f"Device: {DEVICE}")
    print(
        f"dynamic_slots : {dynamic}  ({'Gumbel ON' if dynamic else 'ALL slots fixed'})")
    print(f"use_avalanche : {args.use_avalanche}")
    print(f"checkpoint    : {args.checkpoint}")
    print(f"num_steps     : {args.num_steps}")
    print("="*70)

    loader = get_loader(batch_size=args.batch_size,
                        use_avalanche=args.use_avalanche)
    model, prim_sel, loss_sparse, loss_prim, optimizer, cfg = build(
        checkpoint=args.checkpoint, lr=args.lr
    )
    img_size = cfg.resolution[0]
    print(f"img_size      : {img_size}  (from registry: {cfg.name})")
    print("="*70)

    num_steps = args.num_steps
    log_every = max(1, num_steps // 10)
    print(f"\nTraining {num_steps} steps, batch_size={args.batch_size}")
    print("="*70)
    print(f"{'Step':>5}  {'Total':>8}  {'Recon':>8}  {'Sparse':>8}  {'Prim':>8}  {'ActiveSlots':>12}")
    print("-"*70)

    data_iter = iter(loader)

    running = {k: 0.0 for k in ["total", "recon",
                                "sparse", "prim", "active_slots"]}

    for step_i in range(1, num_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        images, labels = batch[0], batch[1]

        metrics = step(model, prim_sel, loss_sparse, loss_prim,
                       optimizer, images, labels, global_step=step_i,
                       dynamic_slots=dynamic, img_size=img_size)

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
    save_recon_viz(model, loader, VIZ_OUT, img_size=img_size)


if __name__ == "__main__":
    main()
