# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 0 -- AdaSlot Slot Training
#
# Fine-tune AdaSlot on **Avalanche SplitCIFAR-100 task-1 (experience 0)** with:
#   * L_recon (MSE) + L_primitive (concept KL)
#   * Per-batch tqdm, per-epoch loss logging, early stopping
#
# After training:
#   * Extract slot embeddings from test data
#   * PCA 2-D / 3-D scatter coloured by class
#   * K-Means + HDBSCAN clustering using cosine distance on min-max normalised slots

# %% [markdown]
# ## 0. Install / import

# %%
# Uncomment on Kaggle / fresh env:
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#                 "timm", "avalanche-lib", "hdbscan", "umap-learn"], check=True)

import os
import sys
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

# -- project root on sys.path -------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path(".")))

from cont_src.models.adaslot_configs import build_adaslot_from_checkpoint, get_adaslot_config
from cont_src.losses.losses import PrimitiveLoss
from cont_src.models.slot_attention.primitives import PrimitiveSelector

logging.basicConfig(level=logging.WARNING)
print("Imports OK")

# %% [markdown]
# ## 1. CONFIG -- edit everything here

# %%
# ===========================================================================
# CHECKPOINT
# ===========================================================================
CKPT_NAME = "clevr10"    # "clevr10" | "coco" | "movic" | "movie"

_CKPT_KAGGLE = {
    "clevr10": "/kaggle/input/adaslot-clevr10/CLEVR10.ckpt",
    "coco":    "/kaggle/input/adaslot-vit/COCO.ckpt",
    "movic":   "/kaggle/input/adaslot-vit/MOVi-C.ckpt",
    "movie":   "/kaggle/input/adaslot-vit/MOVi-E.ckpt",
}
_CKPT_LOCAL = {
    "clevr10": "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt",
    "coco":    "checkpoints/slot_attention/adaslot_real/COCO.ckpt",
    "movic":   "checkpoints/slot_attention/adaslot_real/MOVi-C.ckpt",
    "movie":   "checkpoints/slot_attention/adaslot_real/MOVi-E.ckpt",
}

_kaggle = _CKPT_KAGGLE.get(CKPT_NAME, "")
CKPT_PATH = _kaggle if os.path.exists(_kaggle) else _CKPT_LOCAL[CKPT_NAME]

# ===========================================================================
# DATASET
# ===========================================================================
DATASET       = "cifar100"   # only cifar100 for now
N_EXPERIENCES = 10           # how many experiences to split CIFAR-100 into
EXP_IDX       = 0            # experience index 0 = "task 1" (first 10 classes)
BATCH_SIZE    = 64
NUM_WORKERS   = 0
VAL_SPLIT     = 0.1          # fraction of train set used as validation

DATA_ROOT = Path("/kaggle/working/cifar100_data") \
    if os.path.exists("/kaggle") else Path("data")

# ===========================================================================
# TRAINING
# ===========================================================================
EPOCHS              = 10
LR                  = 4e-5
DYNAMIC_SLOTS       = False   # False = all slots kept, stable training
W_RECON             = 1.0
W_PRIM              = 10.0
TAU_PRIM            = 10.0
GRAD_CLIP           = 1.0

# Early stopping
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 0.5   # minimum improvement in val recon to reset patience

# ===========================================================================
# SLOT EXTRACTION
# ===========================================================================
EXTRACT_MAX_BATCHES = 100    # cap for extraction (set 0 = all)
EXTRACT_ACTIVE_ONLY = True   # keep only active slots (hard_keep == 1)

# ===========================================================================
# CLUSTERING
# ===========================================================================
KM_N_CLUSTERS           = 20
KM_N_INIT               = 10
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES      = 5

# ===========================================================================
# VISUALIZATION
# ===========================================================================
VIZ_MAX_SLOTS  = 6000        # down-sample for scatter plot
VIZ_OUTPUT_DIR = Path("visualizations/phase0_slots")

# ===========================================================================
# DEVICE
# ===========================================================================
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("=" * 60)
print(f"  Checkpoint : {CKPT_NAME}  --  {CKPT_PATH}")
print(f"  Dataset    : {DATASET}  exp={EXP_IDX}  batch={BATCH_SIZE}")
print(f"  Epochs     : {EPOCHS}  lr={LR}  dyn_slots={DYNAMIC_SLOTS}")
print(f"  W_recon    : {W_RECON}  W_prim: {W_PRIM}")
print(f"  Early stop : patience={EARLY_STOP_PATIENCE}  delta={EARLY_STOP_MIN_DELTA}")
print(f"  Device     : {DEVICE}")
print("=" * 60)

# %% [markdown]
# ## 2. Data -- Avalanche SplitCIFAR-100

# %%
from avalanche.benchmarks.classic import SplitCIFAR100

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# transforms: resize to model resolution, then normalize
cfg_reg = get_adaslot_config(CKPT_NAME)
_IMG_SIZE = cfg_reg.resolution[0]

_tf_train = T.Compose([
    T.Resize((_IMG_SIZE, _IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize((0.5,) * 3, (0.5,) * 3),
])
_tf_eval = T.Compose([
    T.Resize((_IMG_SIZE, _IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5,) * 3, (0.5,) * 3),
])

benchmark = SplitCIFAR100(
    n_experiences=N_EXPERIENCES,
    seed=SEED,
    return_task_id=False,
    dataset_root=str(DATA_ROOT),
    train_transform=_tf_train,
    eval_transform=_tf_eval,
)

# Avalanche returns (x, y, task_id); strip task_id
class _AvDS(torch.utils.data.Dataset):
    def __init__(self, av_ds):
        self._ds = av_ds
    def __len__(self):
        return len(self._ds)
    def __getitem__(self, i):
        x, y, *_ = self._ds[i]
        return x, int(y)

_tr_full = _AvDS(benchmark.train_stream[EXP_IDX].dataset)
_te_ds   = _AvDS(benchmark.test_stream[EXP_IDX].dataset)

_n_val = int(len(_tr_full) * VAL_SPLIT)
_n_tr  = len(_tr_full) - _n_val
tr_ds, val_ds = random_split(
    _tr_full, [_n_tr, _n_val],
    generator=torch.Generator().manual_seed(SEED),
)

_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
           pin_memory=True, drop_last=True)
train_loader = DataLoader(tr_ds,  shuffle=True,  **_kw)
val_loader   = DataLoader(val_ds, shuffle=False, **_kw)
test_loader  = DataLoader(_te_ds, shuffle=False,
                          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                          pin_memory=True, drop_last=False)

_exp_classes = list(benchmark.train_stream[EXP_IDX].classes_in_this_experience)
print(f"Experience {EXP_IDX}: {len(_tr_full)} train  {len(_te_ds)} test")
print(f"  Classes ({len(_exp_classes)}): {_exp_classes}")
print(f"  Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")
print(f"  Image size    : {_IMG_SIZE}x{_IMG_SIZE}")

# %% [markdown]
# ## 3. Build Model

# %%
model = build_adaslot_from_checkpoint(
    checkpoint_name=CKPT_NAME,
    ckpt_path=CKPT_PATH,
    device=str(DEVICE),
    strict_load=True,
).to(DEVICE)

cfg = get_adaslot_config(CKPT_NAME)
SLOT_DIM  = cfg.slot_dim
NUM_SLOTS = cfg.num_slots
IMG_SIZE  = cfg.resolution[0]

prim_sel = PrimitiveSelector(
    slot_dim=SLOT_DIM,
    hidden_dim=SLOT_DIM,
).to(DEVICE)

loss_prim = PrimitiveLoss(temperature=TAU_PRIM, weight=W_PRIM)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(prim_sel.parameters()),
    lr=LR,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel        : {CKPT_NAME}  ({n_params:,} params)")
print(f"PrimSelector : {sum(p.numel() for p in prim_sel.parameters()):,} params")
print(f"Slot dim     : {SLOT_DIM}   Slots: {NUM_SLOTS}")

# %% [markdown]
# ## 4. Training helpers

# %%
def _mse_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum over spatial/channel dims, average over batch."""
    return F.mse_loss(pred, target, reduction="sum") / pred.shape[0]


@torch.no_grad()
def _validate(model, prim_sel, loader, dynamic_slots):
    """Return mean recon + total validation loss."""
    model.eval(); prim_sel.eval()
    total_recon = 0.0; total_loss = 0.0; n = 0
    for images, labels in loader:
        images = images.to(DEVICE); labels = labels.to(DEVICE)
        out = model(images, dynamic_slots=dynamic_slots)
        l_r = W_RECON * _mse_sum(out["reconstruction"], images)
        hk  = out["hard_keep_decision"]
        H   = prim_sel(out["slots"], slot_mask=hk)
        l_p = loss_prim(H, labels)
        total_recon += l_r.item(); total_loss += (l_r + l_p).item(); n += 1
    model.train(); prim_sel.train()
    return total_recon / n, total_loss / n


def _train_one_epoch(model, prim_sel, optimizer, loader, epoch, dynamic_slots):
    """One epoch with per-batch tqdm. Returns dict of mean losses."""
    model.train(); prim_sel.train()
    totals = dict(recon=0.0, prim=0.0, total=0.0, active=0.0)
    pbar = tqdm(loader, desc=f"Ep {epoch:>3}", leave=False, unit="batch")
    for images, labels in pbar:
        images = images.to(DEVICE); labels = labels.to(DEVICE)
        optimizer.zero_grad()

        out   = model(images, dynamic_slots=dynamic_slots)
        recon = out["reconstruction"]
        hk    = out["hard_keep_decision"]   # (B, K) -- all-ones when dynamic=False
        slots = out["slots"]               # (B, K, D)

        l_recon = W_RECON * _mse_sum(recon, images)
        H       = prim_sel(slots, slot_mask=hk)
        l_prim  = loss_prim(H, labels)
        loss    = l_recon + l_prim

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(prim_sel.parameters(), GRAD_CLIP)
        optimizer.step()

        active = hk.sum(dim=1).float().mean().item()
        totals["recon"]  += l_recon.item()
        totals["prim"]   += l_prim.item()
        totals["total"]  += loss.item()
        totals["active"] += active
        pbar.set_postfix({
            "recon": f"{l_recon.item():.1f}",
            "prim":  f"{l_prim.item():.3f}",
            "act":   f"{active:.1f}",
        })

    n = len(loader)
    return {k: v / n for k, v in totals.items()}

# %% [markdown]
# ## 5. Phase 0 Training + Early Stopping

# %%
VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_best_ckpt_path = VIZ_OUTPUT_DIR / f"phase0_best_{CKPT_NAME}.pt"

best_val_recon   = float("inf")
patience_counter = 0
history = {"epoch": [], "train_recon": [], "train_prim": [],
           "val_recon": [], "val_total": [], "active_slots": []}

print(f"\nTraining {EPOCHS} epochs  |  early_stop patience={EARLY_STOP_PATIENCE}")
print("-" * 70)
print(f"{'Ep':>4}  {'tr_recon':>9}  {'tr_prim':>8}  "
      f"{'val_recon':>10}  {'val_total':>9}  {'active':>7}  {'note':>6}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    tr = _train_one_epoch(model, prim_sel, optimizer, train_loader, epoch, DYNAMIC_SLOTS)
    val_recon, val_total = _validate(model, prim_sel, val_loader, DYNAMIC_SLOTS)

    improved = val_recon < best_val_recon - EARLY_STOP_MIN_DELTA
    note = ""
    if improved:
        best_val_recon = val_recon
        patience_counter = 0
        torch.save({"model": model.state_dict(),
                    "prim_sel": prim_sel.state_dict(),
                    "epoch": epoch, "val_recon": val_recon},
                   _best_ckpt_path)
        note = "* saved"
    else:
        patience_counter += 1
        note = f"P {patience_counter}/{EARLY_STOP_PATIENCE}"

    history["epoch"].append(epoch)
    history["train_recon"].append(tr["recon"])
    history["train_prim"].append(tr["prim"])
    history["val_recon"].append(val_recon)
    history["val_total"].append(val_total)
    history["active_slots"].append(tr["active"])

    print(f"{epoch:>4}  {tr['recon']:>9.2f}  {tr['prim']:>8.4f}  "
          f"{val_recon:>10.2f}  {val_total:>9.4f}  {tr['active']:>7.2f}  {note}")

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n[STOP] Early stop at epoch {epoch}  (best val_recon = {best_val_recon:.2f})")
        break

print("-" * 70)
print(f"Best checkpoint -> {_best_ckpt_path}")

# -- Reload best weights -------------------------------------------------------
_ckpt_data = torch.load(_best_ckpt_path, map_location=DEVICE)
model.load_state_dict(_ckpt_data["model"])
prim_sel.load_state_dict(_ckpt_data["prim_sel"])
print(f"Reloaded best weights (epoch {_ckpt_data['epoch']}, val_recon={_ckpt_data['val_recon']:.2f})")

# %% [markdown]
# ### Training curves

# %%
_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

_axes[0].plot(history["epoch"], history["train_recon"], label="train recon")
_axes[0].plot(history["epoch"], history["val_recon"],   label="val recon", linestyle="--")
_axes[0].set_xlabel("Epoch"); _axes[0].set_ylabel("Recon loss")
_axes[0].set_title("Reconstruction loss"); _axes[0].legend(); _axes[0].grid(alpha=0.3)

_axes[1].plot(history["epoch"], history["train_prim"], label="train prim", color="orange")
_axes[1].set_xlabel("Epoch"); _axes[1].set_ylabel("Primitive loss")
_axes[1].set_title("Primitive loss"); _axes[1].legend(); _axes[1].grid(alpha=0.3)

plt.suptitle(f"Phase 0 -- {CKPT_NAME}  exp={EXP_IDX}", fontsize=11)
plt.tight_layout()
plt.savefig(VIZ_OUTPUT_DIR / "training_curves.png", dpi=120, bbox_inches="tight")
plt.show(); plt.close()
print("Training curves saved")

# %% [markdown]
# ## 6. Slot Extraction (test set)

# %%
@torch.no_grad()
def extract_slots(model, loader, max_batches=0, active_only=True):
    """
    Extract slot embeddings from loader.

    Returns:
        slots_np  : (N, slot_dim)  float32
        labels_np : (N,)           int
    """
    model.eval()
    all_slots, all_labels = [], []
    for i, (images, labels) in enumerate(tqdm(loader, desc="Extracting", leave=True)):
        if max_batches > 0 and i >= max_batches:
            break
        images = images.to(DEVICE)
        out    = model(images, dynamic_slots=DYNAMIC_SLOTS)
        slots  = out["slots"]              # (B, K, D)
        hk     = out["hard_keep_decision"] # (B, K)

        B, K, D = slots.shape
        if active_only:
            # keep only slots where hard_keep == 1
            mask = hk.bool()                               # (B, K)
            for b in range(B):
                active_s = slots[b][mask[b]]               # (n_active, D)
                n_active = active_s.shape[0]
                all_slots.append(active_s.cpu().float())
                all_labels.append(torch.full((n_active,), labels[b].item(), dtype=torch.long))
        else:
            all_slots.append(slots.reshape(B * K, D).cpu().float())
            all_labels.append(labels.unsqueeze(1).expand(B, K).reshape(-1))

    slots_np  = torch.cat(all_slots,  dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    model.train()
    return slots_np, labels_np


print("Extracting slot embeddings from test set ...")
slots_np, labels_np = extract_slots(
    model, test_loader,
    max_batches=EXTRACT_MAX_BATCHES,
    active_only=EXTRACT_ACTIVE_ONLY,
)
print(f"Extracted : {slots_np.shape}  labels: {labels_np.shape}")
print(f"Label range: {labels_np.min()} - {labels_np.max()}")

# %% [markdown]
# ## 7. Min-max normalisation + cosine-distance features

# %%
# 1. Min-max normalise each dimension to [0, 1]
slots_mm = minmax_scale(slots_np)                    # (N, D)  float64

# 2. L2-normalise for cosine distance (||x||=1  =>  ||x-y||^2 = 2(1-cos(x,y)))
slots_cos = normalize(slots_mm, norm="l2")           # (N, D)

n_total = slots_cos.shape[0]
print(f"Slots after min-max + L2 norm: {slots_cos.shape}")

# Down-sample for visualisation if needed
if n_total > VIZ_MAX_SLOTS:
    _idx = np.random.choice(n_total, VIZ_MAX_SLOTS, replace=False)
    _idx.sort()
    vis_slots  = slots_cos[_idx]
    vis_labels = labels_np[_idx]
    print(f"Down-sampled to {VIZ_MAX_SLOTS} for visualisation")
else:
    vis_slots  = slots_cos
    vis_labels = labels_np

n_vis = vis_slots.shape[0]

# %% [markdown]
# ## 8. PCA 2-D + 3-D visualisation -- coloured by class

# %%
pca2 = PCA(n_components=2, random_state=SEED)
pca3 = PCA(n_components=3, random_state=SEED)

emb2 = pca2.fit_transform(vis_slots)
emb3 = pca3.fit_transform(vis_slots)

_unique_labels = np.unique(vis_labels)
_n_cls         = len(_unique_labels)
_cmap          = plt.cm.get_cmap("tab20", _n_cls)
_col_map       = {lbl: _cmap(i) for i, lbl in enumerate(_unique_labels)}
_colors        = [_col_map[l] for l in vis_labels]

# -- 2-D scatter --------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 7))
for i, lbl in enumerate(_unique_labels):
    mask = vis_labels == lbl
    ax2.scatter(emb2[mask, 0], emb2[mask, 1],
                color=_cmap(i), s=5, alpha=0.6, label=str(lbl))
ax2.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax2.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax2.set_title(f"PCA 2-D -- slot embeddings  |  {CKPT_NAME}  exp={EXP_IDX}  (n={n_vis})")
if _n_cls <= 20:
    ax2.legend(markerscale=3, fontsize=7, ncol=2, loc="best")
plt.tight_layout()
plt.savefig(VIZ_OUTPUT_DIR / "pca2d.png", dpi=140, bbox_inches="tight")
plt.show(); plt.close()
print(f"PCA 2-D saved  (var explained: {sum(pca2.explained_variance_ratio_)*100:.1f}%)")

# -- 3-D scatter --------------------------------------------------------------
fig3 = plt.figure(figsize=(11, 8))
ax3  = fig3.add_subplot(111, projection="3d")
for i, lbl in enumerate(_unique_labels):
    mask = vis_labels == lbl
    ax3.scatter(emb2[mask, 0], emb2[mask, 1],      # reuse 2-D xy
                emb3[mask, 2],                      # 3rd PCA component
                color=_cmap(i), s=4, alpha=0.5, label=str(lbl))
ax3.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax3.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax3.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
ax3.set_title(f"PCA 3-D -- slot embeddings  |  {CKPT_NAME}  exp={EXP_IDX}")
if _n_cls <= 20:
    ax3.legend(markerscale=3, fontsize=6, ncol=2, loc="upper left")
plt.tight_layout()
plt.savefig(VIZ_OUTPUT_DIR / "pca3d.png", dpi=140, bbox_inches="tight")
plt.show(); plt.close()
print("PCA 3-D saved")

# %% [markdown]
# ## 9. Clustering -- K-Means + HDBSCAN
#
# **Distance**: cosine similarity on min-max normalised slot embeddings.
# For K-Means we L2-normalise so that Euclidean distance = cosine distance.
# For HDBSCAN we pass `metric='cosine'` directly.

# %%
# -- K-Means (cosine proxy via L2-norm) ---------------------------------------
print(f"K-Means  k={KM_N_CLUSTERS}  ...")
km = MiniBatchKMeans(
    n_clusters=KM_N_CLUSTERS,
    n_init=KM_N_INIT,
    random_state=SEED,
    batch_size=min(4096, slots_cos.shape[0]),
)
km_labels_all = km.fit_predict(slots_cos)       # use ALL extracted slots
km_labels_vis = km_labels_all[:n_vis] if n_total <= VIZ_MAX_SLOTS \
    else km.predict(vis_slots)

print(f"  Clusters found  : {KM_N_CLUSTERS}")
print(f"  Inertia         : {km.inertia_:.2f}")

# -- HDBSCAN (cosine metric, min-max norm input) -------------------------------
try:
    import hdbscan

    print(f"\nHDBSCAN  min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}  ...")
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="cosine",       # cosine distance on min-max normalised features
        core_dist_n_jobs=-1,
    )
    # HDBSCAN can be slow at large N -> fit on vis subset
    hdb.fit(vis_slots)
    hdb_labels = hdb.labels_                      # -1 = noise
    n_hdb_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise        = (hdb_labels == -1).sum()
    print(f"  Clusters found  : {n_hdb_clusters}")
    print(f"  Noise points    : {n_noise} / {len(hdb_labels)}")
    _hdbscan_ok = True
except ImportError:
    print("  hdbscan not installed -- skipping. Run: pip install hdbscan")
    hdb_labels     = np.zeros(n_vis, dtype=int)
    n_hdb_clusters = 0
    _hdbscan_ok    = False

# %% [markdown]
# ### 9a. Cluster visualisation in PCA space

# %%
_CLUST_CMAP = plt.cm.get_cmap("tab20", max(KM_N_CLUSTERS, n_hdb_clusters + 2))

fig_cl, axes_cl = plt.subplots(1, 2, figsize=(18, 7))

# K-Means
ax = axes_cl[0]
for k in range(KM_N_CLUSTERS):
    m = km_labels_vis == k
    ax.scatter(emb2[m, 0], emb2[m, 1],
               color=_CLUST_CMAP(k), s=6, alpha=0.6, label=str(k))
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(f"K-Means  k={KM_N_CLUSTERS}  |  {CKPT_NAME}  exp={EXP_IDX}")
ax.grid(alpha=0.2)

# HDBSCAN
ax = axes_cl[1]
_hdb_unique = sorted(set(hdb_labels))
for i, k in enumerate(_hdb_unique):
    m     = hdb_labels == k
    col   = "lightgrey" if k == -1 else _CLUST_CMAP(i)
    label = "noise" if k == -1 else str(k)
    ax.scatter(emb2[m, 0], emb2[m, 1],
               color=col, s=6, alpha=0.6 if k != -1 else 0.2, label=label)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(
    f"HDBSCAN  min_cs={HDBSCAN_MIN_CLUSTER_SIZE}  "
    f"-> {n_hdb_clusters} clusters  |  {CKPT_NAME}  exp={EXP_IDX}"
)
ax.grid(alpha=0.2)

plt.suptitle("Cosine-distance clustering of slot embeddings", fontsize=12)
plt.tight_layout()
plt.savefig(VIZ_OUTPUT_DIR / "clusters.png", dpi=140, bbox_inches="tight")
plt.show(); plt.close()
print("Cluster plot saved")

# %% [markdown]
# ### 9b. Class purity of K-Means clusters (how well clusters align with true labels)

# %%
_purity_rows = []
for k in range(KM_N_CLUSTERS):
    mask = km_labels_vis == k
    if mask.sum() == 0:
        continue
    cl_labels = vis_labels[mask]
    vals, counts = np.unique(cl_labels, return_counts=True)
    dominant_class  = vals[counts.argmax()]
    dominant_count  = counts.max()
    purity = dominant_count / mask.sum()
    _purity_rows.append((k, int(mask.sum()), int(dominant_class), float(purity)))

_purity_rows.sort(key=lambda x: -x[3])
mean_purity = np.mean([r[3] for r in _purity_rows])

print(f"\nK-Means cluster purity  (mean = {mean_purity:.3f})")
print(f"  {'Cluster':>8}  {'Size':>6}  {'Dom.class':>10}  {'Purity':>8}")
print(f"  {'-'*40}")
for k, sz, dc, pur in _purity_rows[:20]:
    print(f"  {k:>8}  {sz:>6}  {dc:>10}  {pur:>8.3f}")

# %% [markdown]
# ## 10. Save summary

# %%
summary = {
    "ckpt_name":        CKPT_NAME,
    "exp_idx":          EXP_IDX,
    "slot_dim":         SLOT_DIM,
    "num_slots":        NUM_SLOTS,
    "n_slots_extracted": int(slots_np.shape[0]),
    "pca2_var":         float(sum(pca2.explained_variance_ratio_)),
    "pca3_var":         float(sum(pca3.explained_variance_ratio_)),
    "km_clusters":      KM_N_CLUSTERS,
    "km_inertia":       float(km.inertia_),
    "km_mean_purity":   float(mean_purity),
    "hdbscan_clusters": n_hdb_clusters,
    "hdbscan_noise":    int((hdb_labels == -1).sum()) if _hdbscan_ok else -1,
    "best_val_recon":   float(best_val_recon),
    "history":          history,
}

_sum_path = VIZ_OUTPUT_DIR / f"summary_{CKPT_NAME}_exp{EXP_IDX}.json"
with open(_sum_path, "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 60)
print(f"  Slots extracted : {slots_np.shape[0]:,}")
print(f"  PCA 2D var      : {summary['pca2_var']*100:.1f}%")
print(f"  PCA 3D var      : {summary['pca3_var']*100:.1f}%")
print(f"  KM mean purity  : {mean_purity:.3f}")
if _hdbscan_ok:
    print(f"  HDBSCAN clusters: {n_hdb_clusters}  (noise {summary['hdbscan_noise']})")
print(f"  Summary -> {_sum_path}")
print(f"  Plots   -> {VIZ_OUTPUT_DIR}")
print("=" * 60)
