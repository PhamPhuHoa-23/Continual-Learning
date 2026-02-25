"""
run_kaggle_pipeline.py
======================
Full continual-learning pipeline on CIFAR-100, designed to run on Kaggle
(single GPU, internet-off, checkpoint uploaded as a dataset).

Phases
------
  0. AdaSlot fine-tune  – backbone learns CIFAR-100 appearance
  I. Cluster init       – slot embeddings → HDBSCAN → VAE + Agent per cluster
  A. Agent warm-up      – L_agent only (anti-collapse)
  B. Agent full train   – soft routing + L_prim + L_SupCon
  C. SLDA fit           – closed-form, one pass
  E. Evaluate           – top-1 accuracy on held-out test split

Kaggle usage
------------
  1. Upload checkpoint as a dataset (Input → "+Add data" → upload .ckpt)
  2. Set CHECKPOINT path below
  3. Runtime → Run all

Local usage
-----------
  python run_kaggle_pipeline.py
  python run_kaggle_pipeline.py --steps0 500 --stepsA 100 --stepsB 300
"""

from __future__ import annotations

# ─── stdlib ──────────────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ─── third-party ─────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
try:
    from torchvision.datasets import ImageFolder as _ImageFolder  # for Tiny-ImageNet
except ImportError:
    _ImageFolder = None

# ─── project ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.models.adaslot.model import AdaSlotModel
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from cont_src.losses.losses import SparsePenalty, PrimitiveLoss
from cont_src.models.aggregators.attention_aggregator import AttentionAggregator
from cont_src.training import (
    AdaSlotTrainerConfig, AdaSlotTrainer,
    ClusterInitConfig, ClusterInitialiser,
    PhaseAConfig,     AgentPhaseATrainer,
    PhaseBConfig,     AgentPhaseBTrainer,
    SLDAConfig,       SLDATrainer, StreamLDA,
)
# extract_slots from cluster_init (works with any model wrapper)
from cont_src.training.cluster_init import extract_slots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("pipeline")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these to match your Kaggle input paths
# ═══════════════════════════════════════════════════════════════════════════

# ── Paths ──────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"
DATA_ROOT    = "data"                     # CIFAR-100 will download here
OUTPUT_DIR   = Path("outputs/pipeline")

# ── Model ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 128
NUM_SLOTS  = 11
SLOT_DIM   = 64
D_H        = 64           # agent hidden / aggregator output dim

# ── Data ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_WORKERS  = 2          # set 0 if multiprocessing causes issues on Kaggle
VAL_FRACTION = 0.1        # fraction of train set used as validation

# ── Phase 0 ────────────────────────────────────────────────────────────────
STEPS_0        = 300      # AdaSlot fine-tune steps (increase for better recon)
LR_0           = 4e-4
W_RECON        = 1.0      # mse_sum raw (matches original AdaSlot)
SPARSE_LINEAR  = 10.0
W_PRIM         = 5.0

# ── Cluster init ───────────────────────────────────────────────────────────
CLUSTER_METHOD = "hdbscan"    # "hdbscan" | "kmeans" | "dbscan" | "bayesian_gmm"
CLUSTER_KWARGS = {"min_cluster_size": 30, "min_samples": 5}
# If using kmeans, set n_clusters instead:
# CLUSTER_METHOD = "kmeans"; CLUSTER_KWARGS = {}; N_CLUSTERS = 8

N_CLUSTERS     = 8            # only used for fixed-count methods
VAE_LATENT_DIM = 16
VAE_EPOCHS     = 15
SCORING_MODE   = "generative" # "generative" | "mahal_z" | "mahal_slot"

# ── Phase A ────────────────────────────────────────────────────────────────
STEPS_A   = 50
LR_A      = 3e-4
GAMMA_A   = 1.0

# ── Phase B ────────────────────────────────────────────────────────────────
STEPS_B   = 200
LR_B      = 2e-4
GAMMA_B   = 1.0
ALPHA_B   = 0.3     # L_prim weight
BETA_B    = 0.3     # L_SupCon weight
T_INIT    = 2.0     # routing temperature start
T_FINAL   = 0.1    # routing temperature end

# ── SLDA ───────────────────────────────────────────────────────────────────
N_CLASSES = 100     # default; overridden by --n_classes arg

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET   = "cifar100"  # "cifar100" | "tiny_imagenet"; overridden by --dataset

# ═══════════════════════════════════════════════════════════════════════════
#  AdaSlot wrapper  (bridges model output keys to trainer interface)
# ═══════════════════════════════════════════════════════════════════════════

class AdaSlotWrapper(nn.Module):
    """
    Wraps AdaSlotModel so its output dict matches the interface expected by
    the trainers:

        AdaSlotModel output  →  wrapper output
        "reconstruction"     →  "recon"
        "hard_keep_decision" →  "mask"
        "slots"              →  "slots"   (unchanged)

    Also accepts an optional PrimitiveSelector to produce "primitives".
    """

    def __init__(
        self,
        model: AdaSlotModel,
        prim_selector: nn.Module | None = None,
    ):
        super().__init__()
        self.model         = model
        self.prim_selector = prim_selector

    def forward(self, images: torch.Tensor, **kw) -> dict:
        out = self.model(images, **kw)
        result = {
            "recon":  out["reconstruction"],
            "mask":   out["hard_keep_decision"],
            "slots":  out["slots"],
            # keep originals too in case downstream code needs them
            **out,
        }
        if self.prim_selector is not None:
            H = self.prim_selector(out["slots"], slot_mask=out["hard_keep_decision"])
            result["primitives"] = H.unsqueeze(1)   # (B,1,D) → averaged over slots by trainer
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  Data helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_transform(train: bool) -> T.Compose:
    tfms = [T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)]
    if train:
        tfms = [T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2)] + tfms
    return T.Compose(tfms)


def _tiny_imagenet_split(data_root: str, train: bool, transform):
    """Load Tiny-ImageNet train/val split from the official folder layout."""
    from torchvision.datasets import ImageFolder
    from pathlib import Path as _P
    import os as _os

    root = _P(data_root)
    # Try both possible extraction layouts
    candidates_train = [root / "tiny-imagenet-200" / "train",
                        root / "train"]
    candidates_val   = [root / "tiny-imagenet-200" / "val" / "images",
                        root / "tiny-imagenet-200" / "val",
                        root / "val" / "images",
                        root / "val"]
    candidates = candidates_train if train else candidates_val
    for c in candidates:
        if c.exists():
            return ImageFolder(str(c), transform=transform)
    raise FileNotFoundError(
        f"Could not find Tiny-ImageNet {'train' if train else 'val'} folder under {data_root}. "
        "Expected: tiny-imagenet-200/train  or  tiny-imagenet-200/val/images"
    )


def get_loaders(data_root: str, batch_size: int, val_fraction: float, num_workers: int,
               dataset: str = "cifar100"):
    """Returns train_loader, val_loader, test_loader."""
    if dataset == "cifar100":
        train_ds_full = CIFAR100(data_root, train=True,  download=True, transform=make_transform(True))
        test_ds       = CIFAR100(data_root, train=False, download=True, transform=make_transform(False))
    elif dataset == "tiny_imagenet":
        train_ds_full = _tiny_imagenet_split(data_root, train=True,  transform=make_transform(True))
        test_ds       = _tiny_imagenet_split(data_root, train=False, transform=make_transform(False))
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'cifar100' or 'tiny_imagenet'.")

    n_val   = int(len(train_ds_full) * val_fraction)
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = random_split(
        train_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, drop_last=True)
    return (
        DataLoader(train_ds,          shuffle=True,  **kw),
        DataLoader(val_ds,            shuffle=False, **kw),
        DataLoader(test_ds,           shuffle=False, **kw),
    )


# ─── resize-on-the-fly so we don't store 128×128 on disk ─────────────────────

class ResizeLoader:
    """Wraps a DataLoader, resizes images lazily to `size × size`."""
    def __init__(self, loader: DataLoader, size: int, device: torch.device):
        self.loader = loader
        self.size   = size
        self.device = device

    def __iter__(self):
        for imgs, labels in self.loader:
            imgs = F.interpolate(
                imgs.to(self.device), size=(self.size, self.size),
                mode="bilinear", align_corners=False,
            )
            yield imgs, labels.to(self.device)

    def __len__(self):
        return len(self.loader)


# ═══════════════════════════════════════════════════════════════════════════
#  Build model
# ═══════════════════════════════════════════════════════════════════════════

def build_model(checkpoint: str, device: torch.device) -> AdaSlotWrapper:
    backbone = AdaSlotModel(
        resolution     = (IMG_SIZE, IMG_SIZE),
        num_slots      = NUM_SLOTS,
        slot_dim       = SLOT_DIM,
        num_iterations = 3,
        feature_dim    = SLOT_DIM,
        kvq_dim        = 128,
        low_bound      = 1,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["state_dict"], strict=True)
    log.info(f"Loaded checkpoint: {checkpoint}  (56/56 keys)")

    prim_sel = PrimitiveSelector(slot_dim=SLOT_DIM, hidden_dim=D_H).to(device)

    return AdaSlotWrapper(backbone, prim_sel).to(device)


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_recon_grid(model: AdaSlotWrapper, loader, out_path: Path, n: int = 8,
                    device: torch.device = torch.device("cpu")):
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imgs, _ = next(iter(loader))
    imgs = F.interpolate(imgs, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    imgs = imgs[:n].to(device)
    out  = model(imgs)
    recon  = out["recon"][:n]
    active = out["mask"][:n].sum(dim=1).float()

    def to_np(t):
        return (t.cpu().clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).numpy()

    orig_np  = to_np(imgs)
    recon_np = to_np(recon)

    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].imshow(orig_np[i]);  axes[0, i].axis("off")
        axes[0, i].set_title("orig",    fontsize=7)
        axes[1, i].imshow(recon_np[i]); axes[1, i].axis("off")
        axes[1, i].set_title(f"recon\n({int(active[i])}s)", fontsize=7)
    fig.suptitle("AdaSlot reconstruction — CIFAR-100", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Saved reconstruction grid → {out_path}")
    model.train()


# ═══════════════════════════════════════════════════════════════════════════
#  Arg parser  (so Kaggle cells can override via os.environ or sys.argv)
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default=CHECKPOINT)
    p.add_argument("--data_root",     default=DATA_ROOT)
    p.add_argument("--output_dir",    default=str(OUTPUT_DIR))
    p.add_argument("--dataset",       default=DATASET,
                   choices=["cifar100", "tiny_imagenet"],
                   help="Dataset to train on")
    p.add_argument("--n_classes",     type=int,   default=N_CLASSES,
                   help="Number of output classes (100 for CIFAR-100, 200 for Tiny-ImageNet)")
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--steps0",        type=int,   default=STEPS_0,   help="Phase 0 steps")
    p.add_argument("--stepsA",        type=int,   default=STEPS_A,   help="Phase A steps")
    p.add_argument("--stepsB",        type=int,   default=STEPS_B,   help="Phase B steps")
    p.add_argument("--cluster_method",default=CLUSTER_METHOD)
    p.add_argument("--n_clusters",    type=int,   default=N_CLUSTERS)
    p.add_argument("--skip_phase0",   action="store_true", help="Skip AdaSlot fine-tune")
    p.add_argument("--skip_viz",      action="store_true", help="Skip visualisations")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Device: {device}")
    log.info(f"Output: {out_dir}")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("DATA")
    log.info("="*60)
    log.info(f"Dataset: {args.dataset}  (n_classes={args.n_classes})")
    train_loader, val_loader, test_loader = get_loaders(
        args.data_root, args.batch_size, VAL_FRACTION, NUM_WORKERS,
        dataset=args.dataset,
    )
    # Resize wrappers (lazy 32→128)
    train_rz = ResizeLoader(train_loader, IMG_SIZE, device)
    val_rz   = ResizeLoader(val_loader,   IMG_SIZE, device)
    test_rz  = ResizeLoader(test_loader,  IMG_SIZE, device)
    log.info(f"Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("MODEL")
    log.info("="*60)
    model = build_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total params: {n_params:,}")

    # ── Phase 0 : AdaSlot fine-tune ───────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("PHASE 0 — AdaSlot fine-tune")
    log.info("="*60)

    if not args.skip_phase0:
        cfg0 = AdaSlotTrainerConfig(
            lr            = LR_0,
            max_steps     = args.steps0,
            max_epochs    = 0,
            w_recon       = W_RECON,
            w_sparse      = SPARSE_LINEAR,
            w_prim        = W_PRIM,
            checkpoint_dir= str(out_dir / "phase0"),
            log_every_n_steps = 50,
        )
        trainer0 = AdaSlotTrainer(
            config              = cfg0,
            slot_model          = model,
            primitive_predictor = None,       # wrapper already has prim_selector inside
        )
        metrics0 = trainer0.train(train_rz)
        log.info(f"Phase 0 done: {metrics0}")

        # ── Save Phase 0 history for notebook plotting ─────────────────
        import json as _json
        if isinstance(metrics0, dict):
            # normalise to lists so JSON is serialisable
            hist_payload = {k: ([v] if not isinstance(v, list) else v)
                            for k, v in metrics0.items()}
            # ensure a 'step' key exists
            if "step" not in hist_payload:
                hist_payload["step"] = list(range(len(next(iter(hist_payload.values())))))
            with open(out_dir / "history.json", "w") as _f:
                _json.dump(hist_payload, _f)
            log.info(f"History saved → {out_dir / 'history.json'}")

        if not args.skip_viz:
            save_recon_grid(model, val_loader, out_dir / "recon_after_phase0.png",
                            device=device)
    else:
        log.info("Skipped (--skip_phase0)")

    # ── Cluster init ──────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("CLUSTER INIT")
    log.info("="*60)

    cluster_kwargs = dict(CLUSTER_KWARGS)
    cluster_cfg = ClusterInitConfig(
        method        = args.cluster_method,
        n_clusters    = args.n_clusters,
        method_kwargs = cluster_kwargs,
        vae_latent_dim= VAE_LATENT_DIM,
        vae_epochs    = VAE_EPOCHS,
        scoring_mode  = SCORING_MODE,
        max_batches_for_clustering = 100,   # ~3200 images at batch_size=32
        device        = str(device),
    )

    # NOTE: must use the resize wrapper so slots are extracted at 128×128
    slots_np = extract_slots(model, train_rz, cluster_cfg, device=device)

    init     = ClusterInitialiser(cluster_cfg)
    vaes, agents, cluster_result = init.run(
        slots_np,
        agent_input_dim  = D_H,
        agent_output_dim = D_H,
    )
    M = len(agents)
    log.info(f"Spawned {M} agents via '{args.cluster_method}'")

    # ── Aggregator ────────────────────────────────────────────────────────
    aggregator = AttentionAggregator(hidden_dim=D_H).to(device)
    log.info(f"AttentionAggregator: {sum(p.numel() for p in aggregator.parameters())} params")

    # ── Phase A : agent warm-up ───────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("PHASE A — Agent warm-up (L_agent only)")
    log.info("="*60)

    cfgA = PhaseAConfig(
        lr             = LR_A,
        max_steps      = args.stepsA,
        max_epochs     = 0,
        gamma          = GAMMA_A,
        routing_mode   = "hard",
        checkpoint_dir = str(out_dir / "phaseA"),
        log_every_n_steps = 10,
    )
    trainerA = AgentPhaseATrainer(
        config     = cfgA,
        slot_model = model,
        vaes       = vaes,
        agents     = agents,
    )
    metricsA = trainerA.train(train_rz)
    log.info(f"Phase A done: {metricsA}")

    # ── Phase B : full agent training ─────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("PHASE B — Full agent training (soft routing + labels)")
    log.info("="*60)

    cfgB = PhaseBConfig(
        lr                 = LR_B,
        max_steps          = args.stepsB,
        max_epochs         = 0,
        gamma              = GAMMA_B,
        alpha              = ALPHA_B,
        beta               = BETA_B,
        init_temperature   = T_INIT,
        final_temperature  = T_FINAL,
        temp_anneal        = "cosine",
        aggregator_mode    = "attention",
        freeze_routers     = True,
        checkpoint_dir     = str(out_dir / "phaseB"),
        log_every_n_steps  = 20,
    )
    trainerB = AgentPhaseBTrainer(
        config     = cfgB,
        slot_model = model,
        vaes       = vaes,
        agents     = agents,
        aggregator = aggregator,
    )
    metricsB = trainerB.train(train_rz)
    log.info(f"Phase B done: {metricsB}")

    # Freeze agents after Phase B
    for agent in agents:
        if hasattr(agent, "freeze"):
            agent.freeze()
        else:
            for p in agent.parameters():
                p.requires_grad_(False)
    log.info("Agents frozen.")

    # ── Phase C : SLDA fit ────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("PHASE C — SLDA incremental fit")
    log.info("="*60)

    slda = StreamLDA(
        n_classes   = args.n_classes,
        feature_dim = D_H,
        shrinkage   = 1e-4,
    )
    slda_cfg = SLDAConfig(
        feature_dim  = D_H,
        n_classes    = args.n_classes,
        shrinkage    = 1e-4,
        max_batches  = 0,     # full pass
        device       = str(device),
    )
    trainerC = SLDATrainer(
        config     = slda_cfg,
        slot_model = model,
        agents     = agents,
        aggregator = aggregator,
        slda       = slda,
        vaes       = vaes,
    )
    trainerC.fit(train_rz)
    log.info(f"SLDA fitted on {slda._n_total} samples")

    # ── Evaluation ────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("EVALUATION")
    log.info("="*60)

    val_metrics  = trainerC.evaluate(val_rz)
    test_metrics = trainerC.evaluate(test_rz)
    log.info(f"Val  accuracy: {val_metrics['accuracy']:.4f}  ({val_metrics['accuracy']*100:.1f}%)")
    log.info(f"Test accuracy: {test_metrics['accuracy']:.4f}  ({test_metrics['accuracy']*100:.1f}%)")

    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt_out = out_dir / "pipeline_final.pt"
    torch.save({
        "model":      model.state_dict(),
        "aggregator": aggregator.state_dict(),
        "agents":     [a.state_dict() for a in agents],
        "slda":       slda.state_dict(),
        "vaes":       [v.state_dict() for v in vaes],
        "val_acc":    val_metrics["accuracy"],
        "test_acc":   test_metrics["accuracy"],
        "config": {
            "method": args.cluster_method,
            "n_agents": M,
            "steps0": args.steps0,
            "stepsA": args.stepsA,
            "stepsB": args.stepsB,
        },
    }, ckpt_out)
    log.info(f"Checkpoint saved → {ckpt_out}")

    # ── Final visualisation ───────────────────────────────────────────────
    if not args.skip_viz:
        save_recon_grid(model, val_loader, out_dir / "recon_final.png", device=device)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info("="*60)
    log.info(f"  Dataset           : {args.dataset}")
    log.info(f"  Classes           : {args.n_classes}")
    log.info(f"  Clustering method : {args.cluster_method}")
    log.info(f"  Agents spawned    : {M}")
    log.info(f"  Val  accuracy     : {val_metrics['accuracy']*100:.1f}%")
    log.info(f"  Test accuracy     : {test_metrics['accuracy']*100:.1f}%")
    log.info(f"  Output dir        : {out_dir}")
    log.info("Done.")


if __name__ == "__main__":
    main()
