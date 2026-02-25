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
# # Continual Learning — Full Pipeline (Kaggle)
#
# **Phase 0** AdaSlot fine-tune → **Cluster Init** → **Phase A** Agent warm-up → **Phase B** Full training → **SLDA** fit → **Eval**
#
# Train inline — khong goi subprocess, config het o cell 4.


# %% [markdown]
# ## 1. Paths

# %%
import os, sys
from pathlib import Path

KAGGLE_WORKING = Path("/kaggle/working")
REPO_NAME      = "Continual-Learning"
REPO_PATH      = KAGGLE_WORKING / REPO_NAME

print(f"Repo path : {REPO_PATH}")
print(f"CWD       : {os.getcwd()}")


# %% [markdown]
# ## 2. Clone & Checkout

# %%
import subprocess

GIT_TOKEN  = "YOUR_GITHUB_PAT_HERE"   # Settings > Developer settings > Personal access tokens
GIT_USER   = "PhamPhuHoa-23"
GIT_REPO   = "Continual-Learning"
GIT_BRANCH = "prototype"

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout: print(r.stdout.strip())
    if r.stderr: print(r.stderr.strip())
    return r.returncode

clone_url = f"https://{GIT_USER}:{GIT_TOKEN}@github.com/{GIT_USER}/{GIT_REPO}.git"

if not REPO_PATH.exists():
    print("Cloning...")
    run(f"git clone {clone_url} {REPO_PATH}")
else:
    print("Pulling latest...")
    run(f"git -C {REPO_PATH} pull origin {GIT_BRANCH}")

run(f"git -C {REPO_PATH} checkout {GIT_BRANCH}")
run(f"git -C {REPO_PATH} log --oneline -3")

os.chdir(REPO_PATH)
sys.path.insert(0, str(REPO_PATH))
print(f"CWD: {os.getcwd()}")


# %% [markdown]
# ## 3. Install Dependencies

# %%
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q tqdm numpy matplotlib scikit-learn hdbscan umap-learn
!pip install -q -e .
!pip install -q avalanche-lib
print("Done!")


# %% [markdown]
# ## 4. CONFIG — chinh tat ca o day
#
# | Section | Mo ta |
# |---|---|
# | DATASET | `cifar100` hoac `tiny_imagenet` |
# | CHECKPOINT | Path toi CLEVR10.ckpt |
# | PHASE 0 | AdaSlot fine-tune |
# | CLUSTER | Chon method + hyperparams |
# | PHASE A | Agent warm-up |
# | PHASE B | Full agent training |
# | SLDA | Closed-form classifier |

# %%
import torch

# ===========================================================================
# DATASET
# ===========================================================================
DATASET   = "cifar100"      # "cifar100" | "tiny_imagenet"
N_TASKS   = 1               # so task muon train (1 = chi task 1)
BATCH_SIZE  = 64
NUM_WORKERS = 2
VAL_SPLIT   = 0.1           # fraction of train set dung lam validation

# ===========================================================================
# CHECKPOINT  (upload CLEVR10.ckpt len Kaggle dataset roi dien path)
# ===========================================================================
CKPT_PATH = "/kaggle/input/adaslot-clevr10/CLEVR10.ckpt"
import os as _os
if not _os.path.exists(CKPT_PATH):
    CKPT_PATH = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"

# ===========================================================================
# MODEL
# ===========================================================================
IMG_SIZE   = 128
NUM_SLOTS  = 11
SLOT_DIM   = 64
D_H        = 64     # agent hidden dim / aggregator output dim

# ===========================================================================
# PHASE 0 — AdaSlot fine-tune
# ===========================================================================
P0_EPOCHS    = 2            # so epoch (1 epoch ≈ 700 buoc voi CIFAR-100 bs=64)
P0_LR        = 4e-4
P0_W_RECON   = 1.0          # recon loss weight
P0_W_SPARSE  = 10.0         # sparsity penalty weight
P0_W_PRIM    = 5.0          # primitive loss weight
P0_LOG_EVERY = 50

# ===========================================================================
# CLUSTER INIT
# ===========================================================================
CLUSTER_METHOD  = "hdbscan"   # "hdbscan" | "kmeans" | "dbscan" | "gmm" | "bayesian_gmm"
CLUSTER_KWARGS  = {"min_cluster_size": 30, "min_samples": 5}
# KMeans/GMM: them {"n_clusters": 8} vao CLUSTER_KWARGS
MAX_BATCH_CLUST = 50          # so batch dung de extract slots (0 = dung tat ca)
                              # 50 batches x 64 x ~11 slots ~ 35k -> subsample xuong MAX_SLOTS_CLUST
MAX_SLOTS_CLUST = 20_000      # hard cap truoc khi feed vao HDBSCAN (0 = khong cap)
                              # HDBSCAN la O(n^2) memory -> giu <= 20k de tranh OOM
VAE_LATENT_DIM  = 16
VAE_EPOCHS      = 20
SCORING_MODE    = "generative"  # "generative" | "mahal_z" | "mahal_slot"

# ===========================================================================
# PHASE A — Agent warm-up (hard routing, L_agent only)
# ===========================================================================
PA_EPOCHS    = 1
PA_LR        = 3e-4
PA_GAMMA     = 1.0
PA_LOG_EVERY = 20

# ===========================================================================
# PHASE B — Full training (soft routing + L_prim + L_SupCon)
# ===========================================================================
PB_EPOCHS         = 2
PB_LR             = 2e-4
PB_GAMMA          = 1.0       # L_agent weight
PB_ALPHA          = 0.3       # L_prim weight
PB_BETA           = 0.3       # L_SupCon weight
PB_T_INIT         = 2.0       # routing temperature start
PB_T_FINAL        = 0.1       # routing temperature end
PB_TEMP_ANNEAL    = "cosine"  # "cosine" | "linear" | "constant"
PB_FREEZE_ROUTERS = True
PB_LOG_EVERY      = 20

# ===========================================================================
# SLDA
# ===========================================================================
SLDA_SHRINKAGE = 1e-4

# ===========================================================================
# OUTPUT
# ===========================================================================
OUTPUT_DIR = Path(f"/kaggle/working/pipeline_out/{DATASET}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# DEVICE
# ===========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DS_META = {
    "cifar100":      {"n_classes": 100, "data_subdir": "cifar100_data"},
    "tiny_imagenet": {"n_classes": 200, "data_subdir": "tiny_imagenet_data"},
}
assert DATASET in _DS_META, f"Unknown dataset '{DATASET}'"
DATA_ROOT = Path(f"/kaggle/working/{_DS_META[DATASET]['data_subdir']}")
N_CLASSES = _DS_META[DATASET]["n_classes"]

print("=" * 55)
print(f"  Dataset    : {DATASET}  ({N_CLASSES} classes)")
print(f"  Device     : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  Checkpoint : {CKPT_PATH}")
print(f"  Output     : {OUTPUT_DIR}")
print(f"  Phase 0    : {P0_EPOCHS} epochs")
print(f"  Phase A    : {PA_EPOCHS} epochs")
print(f"  Phase B    : {PB_EPOCHS} epochs")
print("=" * 55)


# %% [markdown]
# ## 5. Download Dataset

# %%
import torchvision.datasets as _D

DATA_ROOT.mkdir(parents=True, exist_ok=True)

if DATASET == "cifar100":
    _D.CIFAR100(DATA_ROOT, train=True,  download=True)
    _D.CIFAR100(DATA_ROOT, train=False, download=True)
    print("CIFAR-100 ready")

elif DATASET == "tiny_imagenet":
    import urllib.request, zipfile
    _url  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    _dest = DATA_ROOT / "tiny-imagenet-200.zip"
    if not _dest.exists():
        urllib.request.urlretrieve(_url, _dest,
            reporthook=lambda b,bs,t: print(f"  {b*bs/1e6:.0f}/{t/1e6:.0f} MB", end="\r")
                         if b % 200 == 0 else None)
    with zipfile.ZipFile(_dest) as z:
        z.extractall(DATA_ROOT)
    print("Tiny-ImageNet ready")


# %% [markdown]
# ## 6. Import (lan 1 — co the loi do Avalanche, binh thuong)
#
# > Neu cell nay loi, cu chay tiep cell 7 de re-import sach.

# %%
try:
    import json, random, logging
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm.notebook import tqdm
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader, random_split

    from src.models.adaslot.model import AdaSlotModel
    from cont_src.models.slot_attention.primitives import PrimitiveSelector
    from cont_src.models.aggregators.attention_aggregator import AttentionAggregator
    from cont_src.training import (
        AdaSlotTrainer,    AdaSlotTrainerConfig,
        ClusterInitialiser, ClusterInitConfig,
        AgentPhaseATrainer, PhaseAConfig,
        AgentPhaseBTrainer, PhaseBConfig,
        SLDATrainer,       SLDAConfig, StreamLDA,
    )
    from cont_src.training.cluster_init import extract_slots
    print("Import lan 1 OK")
except Exception as e:
    print(f"Import lan 1 loi: {e}")
    print("-> Chay cell tiep theo de re-import")


# %% [markdown]
# ## 7. Import (lan 2 — sach)

# %%
import json, random, logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split

from src.models.adaslot.model import AdaSlotModel
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from cont_src.models.aggregators.attention_aggregator import AttentionAggregator
from cont_src.training import (
    AdaSlotTrainer,    AdaSlotTrainerConfig,
    ClusterInitialiser, ClusterInitConfig,
    AgentPhaseATrainer, PhaseAConfig,
    AgentPhaseBTrainer, PhaseBConfig,
    SLDATrainer,       SLDAConfig, StreamLDA,
)
from cont_src.training.cluster_init import extract_slots

print("Imports OK!")


# %% [markdown]
# ## 8. Data Loaders

# %%
def _make_tf(train):
    base = [T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)]
    if train:
        base = [T.RandomHorizontalFlip(), T.ColorJitter(0.2,0.2,0.2)] + base
    return T.Compose(base)

def _get_loaders():
    if DATASET == "cifar100":
        tr = CIFAR100(DATA_ROOT, train=True,  download=False, transform=_make_tf(True))
        te = CIFAR100(DATA_ROOT, train=False, download=False, transform=_make_tf(False))
    elif DATASET == "tiny_imagenet":
        from torchvision.datasets import ImageFolder
        _candidates_tr = [DATA_ROOT/"tiny-imagenet-200"/"train", DATA_ROOT/"train"]
        _candidates_te = [DATA_ROOT/"tiny-imagenet-200"/"val"/"images",
                          DATA_ROOT/"tiny-imagenet-200"/"val", DATA_ROOT/"val"]
        _tr_dir = next(p for p in _candidates_tr if p.exists())
        _te_dir = next(p for p in _candidates_te if p.exists())
        tr = ImageFolder(str(_tr_dir), transform=_make_tf(True))
        te = ImageFolder(str(_te_dir), transform=_make_tf(False))
    n_val   = int(len(tr) * VAL_SPLIT)
    n_train = len(tr) - n_val
    train_ds, val_ds = random_split(tr, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
              pin_memory=True, drop_last=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(te,       shuffle=False, **kw))

train_loader, val_loader, test_loader = _get_loaders()
print(f"Train: {len(train_loader)} batches  Val: {len(val_loader)}  Test: {len(test_loader)}")

# Resize wrapper: CIFAR-100 is 32x32, AdaSlot needs 128x128
class ResizeLoader:
    def __init__(self, loader, size=IMG_SIZE):
        self.loader, self.size = loader, size
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        for imgs, labels in self.loader:
            imgs = F.interpolate(imgs.to(DEVICE), (self.size, self.size),
                                 mode="bilinear", align_corners=False)
            yield imgs, labels.to(DEVICE)

train_rz = ResizeLoader(train_loader)
val_rz   = ResizeLoader(val_loader)
test_rz  = ResizeLoader(test_loader)
print("ResizeLoaders ready")


# %% [markdown]
# ## 9. Build Model

# %%
# Wrapper: maps AdaSlotModel output keys to trainer convention
class SlotModelWrapper(nn.Module):
    def __init__(self, backbone, prim_sel=None):
        super().__init__()
        self.backbone  = backbone
        self.prim_sel  = prim_sel
    def forward(self, images, **kw):
        out = self.backbone(images, **kw)
        result = {
            "recon": out["reconstruction"],
            "mask":  out["hard_keep_decision"],
            "slots": out["slots"],
            **out,
        }
        if self.prim_sel is not None:
            H = self.prim_sel(out["slots"], slot_mask=out["hard_keep_decision"])
            result["primitives"] = H.unsqueeze(1)
        return result

backbone = AdaSlotModel(
    resolution=(IMG_SIZE, IMG_SIZE),
    num_slots=NUM_SLOTS,
    slot_dim=SLOT_DIM,
    num_iterations=3,
    feature_dim=SLOT_DIM,
    kvq_dim=128,
    low_bound=1,
).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
missing, unexpected = backbone.load_state_dict(ckpt["state_dict"], strict=True)
print(f"Checkpoint loaded: {CKPT_PATH}")
print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

prim_sel   = PrimitiveSelector(slot_dim=SLOT_DIM, hidden_dim=D_H).to(DEVICE)
slot_model = SlotModelWrapper(backbone, prim_sel).to(DEVICE)
aggregator = AttentionAggregator(hidden_dim=D_H).to(DEVICE)

n_total = sum(p.numel() for p in slot_model.parameters())
print(f"Total trainable params: {n_total:,}")


# %% [markdown]
# ## 10. Phase 0 — AdaSlot Fine-tune
#
# Train backbone + PrimitiveSelector on reconstruction + sparsity + primitive loss.

# %%
cfg_p0 = AdaSlotTrainerConfig(
    lr               = P0_LR,
    max_steps        = 0,
    max_epochs       = P0_EPOCHS,
    w_recon          = P0_W_RECON,
    w_sparse         = P0_W_SPARSE,
    w_prim           = P0_W_PRIM,
    checkpoint_dir   = str(OUTPUT_DIR / "phase0"),
    log_every_n_steps= P0_LOG_EVERY,
)

trainer_p0 = AdaSlotTrainer(
    config              = cfg_p0,
    slot_model          = slot_model,
    primitive_predictor = None,   # wrapper already has prim_sel
)

print("Phase 0: AdaSlot fine-tune...")
metrics_p0 = trainer_p0.train(train_rz)
print(f"Phase 0 done: {metrics_p0}")

# Save history for plotting
with open(OUTPUT_DIR / "history_p0.json", "w") as f:
    json.dump({k: ([v] if not isinstance(v, list) else v)
               for k, v in metrics_p0.items()}, f)


# %% [markdown]
# ### Recon visualisation after Phase 0

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@torch.no_grad()
def save_recon_grid(model, loader, path, n=8):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = F.interpolate(imgs[:n].to(DEVICE), (IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False)
    out   = model(imgs)
    recon = out["recon"]
    active = out["mask"].sum(dim=1).float()
    def to_np(t):
        return (t.cpu().clamp(-1,1)*0.5+0.5).permute(0,2,3,1).numpy()
    orig_np, recon_np = to_np(imgs), to_np(recon)
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0,i].imshow(orig_np[i]);  axes[0,i].axis("off"); axes[0,i].set_title("orig",  fontsize=7)
        axes[1,i].imshow(recon_np[i]); axes[1,i].axis("off"); axes[1,i].set_title(f"recon\n({int(active[i])}s)", fontsize=7)
    fig.suptitle(f"AdaSlot recon — {DATASET}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.show(); plt.close()
    model.train()

save_recon_grid(slot_model, val_loader, OUTPUT_DIR / "recon_phase0.png")
print("Saved recon grid")


# %% [markdown]
# ## 11. Cluster Init
#
# Extract slot embeddings -> cluster -> spawn 1 SlotVAE + 1 Agent per cluster.

# %%
cfg_clust = ClusterInitConfig(
    method                    = CLUSTER_METHOD,
    method_kwargs             = CLUSTER_KWARGS,
    max_batches_for_clustering= MAX_BATCH_CLUST,
    max_slots_for_clustering  = MAX_SLOTS_CLUST,
    vae_latent_dim            = VAE_LATENT_DIM,
    vae_epochs                = VAE_EPOCHS,
    scoring_mode              = SCORING_MODE,
    device                    = str(DEVICE),
)

print(f"Extracting slots with method='{CLUSTER_METHOD}'...")
slots_np = extract_slots(slot_model, train_rz, cfg_clust, device=DEVICE)
print(f"Slots shape: {slots_np.shape}")

initialiser = ClusterInitialiser(cfg_clust)
vaes, agents, cluster_result = initialiser.run(
    slots_np,
    agent_input_dim  = D_H,
    agent_output_dim = D_H,
)
M = len(agents)
print(f"Spawned {M} agents  ({CLUSTER_METHOD})")


# %% [markdown]
# ## 12. Phase A — Agent Warm-up (hard routing, L_agent only)

# %%
cfg_pa = PhaseAConfig(
    lr               = PA_LR,
    max_steps        = 0,
    max_epochs       = PA_EPOCHS,
    gamma            = PA_GAMMA,
    routing_mode     = "hard",
    checkpoint_dir   = str(OUTPUT_DIR / "phaseA"),
    log_every_n_steps= PA_LOG_EVERY,
)

trainer_pa = AgentPhaseATrainer(
    config     = cfg_pa,
    slot_model = slot_model,
    vaes       = vaes,
    agents     = agents,
)

print("Phase A: agent warm-up...")
metrics_pa = trainer_pa.train(train_rz)
print(f"Phase A done: {metrics_pa}")


# %% [markdown]
# ## 13. Phase B — Full Training (soft routing + L_prim + L_SupCon)

# %%
cfg_pb = PhaseBConfig(
    lr                = PB_LR,
    max_steps         = 0,
    max_epochs        = PB_EPOCHS,
    gamma             = PB_GAMMA,
    alpha             = PB_ALPHA,
    beta              = PB_BETA,
    init_temperature  = PB_T_INIT,
    final_temperature = PB_T_FINAL,
    temp_anneal       = PB_TEMP_ANNEAL,
    freeze_routers    = PB_FREEZE_ROUTERS,
    aggregator_mode   = "attention",
    checkpoint_dir    = str(OUTPUT_DIR / "phaseB"),
    log_every_n_steps = PB_LOG_EVERY,
)

trainer_pb = AgentPhaseBTrainer(
    config     = cfg_pb,
    slot_model = slot_model,
    vaes       = vaes,
    agents     = agents,
    aggregator = aggregator,
)

print("Phase B: full agent training...")
metrics_pb = trainer_pb.train(train_rz)
print(f"Phase B done: {metrics_pb}")

# Freeze agents
for ag in agents:
    if hasattr(ag, "freeze"):
        ag.freeze()
    else:
        for p in ag.parameters():
            p.requires_grad_(False)
print("Agents frozen")


# %% [markdown]
# ## 14. SLDA — Incremental Fit (closed-form, 1 pass)

# %%
slda = StreamLDA(
    n_classes   = N_CLASSES,
    feature_dim = D_H,
    shrinkage   = SLDA_SHRINKAGE,
)

cfg_slda = SLDAConfig(
    feature_dim = D_H,
    n_classes   = N_CLASSES,
    shrinkage   = SLDA_SHRINKAGE,
    max_batches = 0,
    device      = str(DEVICE),
)

trainer_slda = SLDATrainer(
    config     = cfg_slda,
    slot_model = slot_model,
    agents     = agents,
    aggregator = aggregator,
    slda       = slda,
    vaes       = vaes,
)

print("SLDA: fitting...")
trainer_slda.fit(train_rz)
print(f"SLDA fitted on {slda._n_total} samples")


# %% [markdown]
# ## 15. Evaluation

# %%
val_metrics  = trainer_slda.evaluate(val_rz)
test_metrics = trainer_slda.evaluate(test_rz)

print("=" * 45)
print(f"  Val  accuracy : {val_metrics['accuracy']*100:.2f}%")
print(f"  Test accuracy : {test_metrics['accuracy']*100:.2f}%")
print("=" * 45)


# %% [markdown]
# ## 16. Save Checkpoint

# %%
ckpt_out = OUTPUT_DIR / "pipeline_final.pt"
torch.save({
    "slot_model": slot_model.state_dict(),
    "aggregator": aggregator.state_dict(),
    "agents":     [a.state_dict() for a in agents],
    "slda":       slda.state_dict(),
    "vaes":       [v.state_dict() for v in vaes],
    "val_acc":    val_metrics["accuracy"],
    "test_acc":   test_metrics["accuracy"],
    "config": {
        "dataset": DATASET, "n_classes": N_CLASSES,
        "cluster_method": CLUSTER_METHOD, "n_agents": M,
        "p0_epochs": P0_EPOCHS, "pa_epochs": PA_EPOCHS, "pb_epochs": PB_EPOCHS,
    },
}, ckpt_out)
print(f"Checkpoint saved: {ckpt_out}")


# %% [markdown]
# ## 17. Training Curves

# %%
hist_path = OUTPUT_DIR / "history_p0.json"
if hist_path.exists():
    with open(hist_path) as f:
        h = json.load(f)
    keys = [k for k in h if k != "step" and h[k]]
    step = h.get("step", list(range(len(h[keys[0]]))))
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in keys:
        ax.plot(step, h[k], label=k)
    ax.set_title("Phase 0 losses"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=120)
    plt.show(); plt.close()
    print("Curves saved")
else:
    print(f"No history file at {hist_path}")


# %% [markdown]
# ## 18. Summary

# %%
ckpts = sorted(OUTPUT_DIR.rglob("*.pt"))
print("=" * 55)
print(f"  Dataset       : {DATASET}")
print(f"  N agents      : {M}")
print(f"  Val  accuracy : {val_metrics['accuracy']*100:.2f}%")
print(f"  Test accuracy : {test_metrics['accuracy']*100:.2f}%")
print(f"  Output dir    : {OUTPUT_DIR}")
print(f"  Checkpoints   : {len(ckpts)}")
for c in ckpts:
    print(f"    {c.name}  ({c.stat().st_size/1e6:.1f} MB)")
print("=" * 55)
print("Done! Download checkpoints tu tab Output tren Kaggle.")
