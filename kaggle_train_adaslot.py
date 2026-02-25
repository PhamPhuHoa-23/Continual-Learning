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
# **Phase 0** AdaSlot fine-tune → **Cluster Init** → **Phase A** Agent warm-up →
# **Phase B** Full training → **SLDA** fit → **Eval**
#
# For each task after task 0:
# * Phase B fine-tune (agents) + SLDA update
# * Evaluate **all** seen tasks → build forgetting matrix **R[i][j]**
# * Report per-task accuracy, forgetting, and BWT after every task

# %% [markdown]
# ## 1. Paths  *(stdlib only — project not cloned yet)*

# %%
import os
import sys
import shutil as _sh
import subprocess
import importlib as _il
from pathlib import Path

KAGGLE_WORKING = Path("/kaggle/working")
REPO_NAME      = "Continual-Learning"
REPO_PATH      = KAGGLE_WORKING / REPO_NAME

print(f"Repo path : {REPO_PATH}")
print(f"CWD       : {os.getcwd()}")

# %% [markdown]
# ## 2. Clone & Checkout

# %%
# Settings > Developer settings > Personal access tokens
GIT_TOKEN  = "YOUR_GITHUB_PAT_HERE"
GIT_USER   = "PhamPhuHoa-23"
GIT_REPO   = "Continual-Learning"
GIT_BRANCH = "prototype"


def _run(cmd: str) -> int:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout.strip())
    if r.stderr:
        print(r.stderr.strip())
    return r.returncode


clone_url = f"https://{GIT_USER}:{GIT_TOKEN}@github.com/{GIT_USER}/{GIT_REPO}.git"

if not REPO_PATH.exists():
    print("Cloning ...")
    _run(f"git clone {clone_url} {REPO_PATH}")

# fetch + reset --hard: dam bao file tren Kaggle LUON khop voi origin
_run(f"git -C {REPO_PATH} fetch origin {GIT_BRANCH}")
_run(f"git -C {REPO_PATH} checkout --force {GIT_BRANCH}")
_run(f"git -C {REPO_PATH} reset --hard origin/{GIT_BRANCH}")
_run(f"git -C {REPO_PATH} log --oneline -3")

os.chdir(REPO_PATH)
sys.path.insert(0, str(REPO_PATH))
print(f"CWD: {os.getcwd()}")

# Xoa __pycache__ de Python khong load bytecode cu sau git pull
for _p in REPO_PATH.rglob("__pycache__"):
    _sh.rmtree(_p, ignore_errors=True)
_il.invalidate_caches()
print("__pycache__ cleared")

# %% [markdown]
# ## 3. Install Dependencies

# %%
get_ipython().system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
get_ipython().system("pip install -q tqdm numpy matplotlib scikit-learn hdbscan umap-learn")
get_ipython().system("pip install -q -e .")
get_ipython().system("pip install -q avalanche-lib")
print("Dependencies installed!")

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
import os as _os
import torch

# ===========================================================================
# DATASET
# ===========================================================================
DATASET     = "cifar100"   # "cifar100" | "tiny_imagenet"
N_TASKS     = 10           # so task muon train (1 = chi task 0)
BATCH_SIZE  = 64
NUM_WORKERS = 0
VAL_SPLIT   = 0.1          # fraction of train set dung lam validation

# ===========================================================================
# CHECKPOINT  (upload CLEVR10.ckpt len Kaggle dataset roi dien path)
# ===========================================================================
CKPT_PATH = "/kaggle/input/adaslot-clevr10/CLEVR10.ckpt"
if not _os.path.exists(CKPT_PATH):
    CKPT_PATH = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"

# ===========================================================================
# MODEL
# ===========================================================================
IMG_SIZE  = 128   # decoder la 4x stride-2 ConvTranspose -> output 128x128
NUM_SLOTS = 7
SLOT_DIM  = 64
D_H       = 64    # agent hidden dim / aggregator output dim

# ===========================================================================
# PHASE 0 — AdaSlot fine-tune
# ===========================================================================
P0_EPOCHS    = 1
P0_LR        = 4e-5
P0_W_RECON   = 1.0
P0_W_SPARSE  = 10.0
P0_W_PRIM    = 5.0
P0_LOG_EVERY = 50

# ===========================================================================
# CLUSTER INIT
# ===========================================================================
CLUSTER_METHOD  = "hdbscan"   # "hdbscan" | "kmeans" | "dbscan" | "gmm"
CLUSTER_KWARGS  = {"min_cluster_size": 30, "min_samples": 5}
MAX_BATCH_CLUST = 50
MAX_SLOTS_CLUST = 20_000      # HDBSCAN O(n^2) -> keep <= 20k
VAE_LATENT_DIM  = 32
VAE_EPOCHS      = 1
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
PB_T_INIT         = 2.0
PB_T_FINAL        = 0.1
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
DATA_ROOT        = Path(f"/kaggle/working/{_DS_META[DATASET]['data_subdir']}")
N_CLASSES        = _DS_META[DATASET]["n_classes"]
CLASSES_PER_TASK = N_CLASSES // N_TASKS

print("=" * 55)
print(f"  Dataset    : {DATASET}  ({N_CLASSES} classes, {N_TASKS} tasks)")
print(f"  Device     : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  Checkpoint : {CKPT_PATH}")
print(f"  Output     : {OUTPUT_DIR}")
print(f"  Phase 0    : {P0_EPOCHS} epochs  |  Phase A : {PA_EPOCHS}  |  Phase B : {PB_EPOCHS}")
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
    import urllib.request
    import zipfile
    _url  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    _dest = DATA_ROOT / "tiny-imagenet-200.zip"
    if not _dest.exists():
        urllib.request.urlretrieve(
            _url, _dest,
            reporthook=lambda b, bs, t: (
                print(f"  {b*bs/1e6:.0f}/{t/1e6:.0f} MB", end="\r")
                if b % 200 == 0 else None
            ),
        )
    with zipfile.ZipFile(_dest) as z:
        z.extractall(DATA_ROOT)
    print("Tiny-ImageNet ready")

# %% [markdown]
# ## 6. Project Imports  *(repo is on disk now)*

# %%
import json
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, random_split
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet

from src.models.adaslot.model import AdaSlotModel
from cont_src.models.slot_attention.primitives import PrimitiveSelector
from cont_src.models.aggregators.attention_aggregator import AttentionAggregator
from cont_src.training import (
    AdaSlotTrainer,     AdaSlotTrainerConfig,
    ClusterInitialiser, ClusterInitConfig,
    AgentPhaseATrainer, PhaseAConfig,
    AgentPhaseBTrainer, PhaseBConfig,
    SLDATrainer,        SLDAConfig, StreamLDA,
)
from cont_src.training.cluster_init import extract_slots

print("All imports OK")

# %% [markdown]
# ## 7. Data — Avalanche Benchmark
#
# `SplitCIFAR100` / `SplitTinyImageNet` chia classes theo task tu dong.
# Resize 32->128 xu ly thang trong transform — khong can ResizeLoader nua.

# %%
def _make_tf(train: bool, size: int = IMG_SIZE):
    aug = [T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2)] if train else []
    return T.Compose(aug + [
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


# Avalanche AvalancheDataset tra ve (x, y, task_id); bo task_id
class _AvDS(torch.utils.data.Dataset):
    def __init__(self, av_ds):
        self._ds = av_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        x, y, *_ = self._ds[i]
        return x, y


if DATASET == "cifar100":
    benchmark = SplitCIFAR100(
        n_experiences=N_TASKS,
        seed=42,
        return_task_id=False,
        dataset_root=str(DATA_ROOT),
        train_transform=_make_tf(True),
        eval_transform=_make_tf(False),
    )
else:
    benchmark = SplitTinyImageNet(
        n_experiences=N_TASKS,
        seed=42,
        return_task_id=False,
        dataset_root=str(DATA_ROOT),
        train_transform=_make_tf(True),
        eval_transform=_make_tf(False),
    )


def get_task_loaders(task_id: int):
    """Return (train_loader, val_loader, test_loader) for task `task_id`."""
    tr_full = _AvDS(benchmark.train_stream[task_id].dataset)
    te_ds   = _AvDS(benchmark.test_stream[task_id].dataset)
    n_val   = int(len(tr_full) * VAL_SPLIT)
    n_tr    = len(tr_full) - n_val
    tr_ds, val_ds = random_split(
        tr_full, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
              pin_memory=True, drop_last=True)
    return (DataLoader(tr_ds,  shuffle=True,  **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(te_ds,  shuffle=False, **kw))


train_loader, val_loader, test_loader = get_task_loaders(0)
print(f"Task 0  |  classes 0-{CLASSES_PER_TASK-1}  "
      f"|  Train: {len(train_loader)} batches  "
      f"Val: {len(val_loader)}  Test: {len(test_loader)}")
print(f"Total tasks: {N_TASKS}  |  Classes per task: {CLASSES_PER_TASK}")

# %% [markdown]
# ## 8. Build Model

# %%
class SlotModelWrapper(nn.Module):
    """Maps AdaSlotModel output keys to the trainer convention."""

    def __init__(self, backbone, prim_sel=None):
        super().__init__()
        self.backbone = backbone
        self.prim_sel = prim_sel

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


# Reset gate: CLEVR10 checkpoint saturated (slots_keep_prob ~1.0 on CLEVR)
# Re-init lets the gate learn proper keep/drop on the new domain.
def _reset_gumbel_gate(model: nn.Module) -> None:
    for name, mod in model.named_modules():
        if "single_gumbel_score_network" in name or "gumbel_score" in name:
            for p in mod.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
            print(f"  Gate reset: {name}")


_reset_gumbel_gate(backbone)
print("Gate weights re-initialised")

prim_sel   = PrimitiveSelector(slot_dim=SLOT_DIM, hidden_dim=D_H).to(DEVICE)
slot_model = SlotModelWrapper(backbone, prim_sel).to(DEVICE)
aggregator = AttentionAggregator(hidden_dim=D_H).to(DEVICE)

n_total = sum(p.numel() for p in slot_model.parameters())
print(f"Total params: {n_total:,}")

# %% [markdown]
# ## 9. Phase 0 — AdaSlot Fine-tune
#
# Train backbone + PrimitiveSelector on reconstruction + sparsity + primitive loss.

# %%
cfg_p0 = AdaSlotTrainerConfig(
    lr=P0_LR,
    max_steps=0,
    max_epochs=P0_EPOCHS,
    w_recon=P0_W_RECON,
    w_sparse=P0_W_SPARSE,
    w_prim=P0_W_PRIM,
    checkpoint_dir=str(OUTPUT_DIR / "phase0"),
    log_every_n_steps=P0_LOG_EVERY,
)

trainer_p0 = AdaSlotTrainer(
    config=cfg_p0,
    slot_model=slot_model,
    primitive_predictor=None,
)

print("Phase 0: AdaSlot fine-tune ...")
metrics_p0 = trainer_p0.train(train_loader)
print(f"Phase 0 done: {metrics_p0}")

with open(OUTPUT_DIR / "history_p0.json", "w") as f:
    json.dump({k: ([v] if not isinstance(v, list) else v)
               for k, v in metrics_p0.items()}, f)

# %% [markdown]
# ### Recon visualisation after Phase 0

# %%
matplotlib.use("Agg")


@torch.no_grad()
def save_recon_grid(model, loader, path, n: int = 8):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs   = imgs[:n].to(DEVICE)
    out    = model(imgs)
    recon  = out["recon"]
    active = out["mask"].sum(dim=1).float()

    def _np(t):
        return (t.cpu().clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).numpy()

    orig_np, recon_np = _np(imgs), _np(recon)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(orig_np[i])
        axes[0, i].axis("off")
        axes[0, i].set_title("orig", fontsize=7)
        axes[1, i].imshow(recon_np[i])
        axes[1, i].axis("off")
        axes[1, i].set_title(f"recon\n({int(active[i])}s)", fontsize=7)
    fig.suptitle(f"AdaSlot recon — {DATASET}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    model.train()


save_recon_grid(slot_model, val_loader, OUTPUT_DIR / "recon_phase0.png")
print("Recon grid saved")

# %% [markdown]
# ## 10. Cluster Init
#
# Extract slot embeddings -> cluster -> spawn 1 SlotVAE + 1 Agent per cluster.

# %%
cfg_clust = ClusterInitConfig(
    method=CLUSTER_METHOD,
    method_kwargs=CLUSTER_KWARGS,
    max_batches_for_clustering=MAX_BATCH_CLUST,
    max_slots_for_clustering=MAX_SLOTS_CLUST,
    vae_latent_dim=VAE_LATENT_DIM,
    vae_epochs=VAE_EPOCHS,
    scoring_mode=SCORING_MODE,
    device=str(DEVICE),
)

print(f"Extracting slots (method='{CLUSTER_METHOD}') ...")
slots_np = extract_slots(slot_model, train_loader, cfg_clust, device=DEVICE)
print(f"Slots shape: {slots_np.shape}")

initialiser = ClusterInitialiser(cfg_clust)
vaes, agents, cluster_result = initialiser.run(
    slots_np,
    agent_input_dim=D_H,
    agent_output_dim=D_H,
)
M = len(agents)
print(f"Spawned {M} agents  ({CLUSTER_METHOD})")

# %% [markdown]
# ## 11. Phase A — Agent Warm-up  *(hard routing, L_agent only)*

# %%
cfg_pa = PhaseAConfig(
    lr=PA_LR,
    max_steps=0,
    max_epochs=PA_EPOCHS,
    gamma=PA_GAMMA,
    routing_mode="hard",
    checkpoint_dir=str(OUTPUT_DIR / "phaseA"),
    log_every_n_steps=PA_LOG_EVERY,
)

trainer_pa = AgentPhaseATrainer(
    config=cfg_pa,
    slot_model=slot_model,
    vaes=vaes,
    agents=agents,
)

print("Phase A: agent warm-up ...")
metrics_pa = trainer_pa.train(train_loader)
print(f"Phase A done: {metrics_pa}")

# %% [markdown]
# ## 12. Phase B — Full Training  *(soft routing + L_prim + L_SupCon)*

# %%
cfg_pb = PhaseBConfig(
    lr=PB_LR,
    max_steps=0,
    max_epochs=PB_EPOCHS,
    gamma=PB_GAMMA,
    alpha=PB_ALPHA,
    beta=PB_BETA,
    init_temperature=PB_T_INIT,
    final_temperature=PB_T_FINAL,
    temp_anneal=PB_TEMP_ANNEAL,
    freeze_routers=PB_FREEZE_ROUTERS,
    aggregator_mode="attention",
    checkpoint_dir=str(OUTPUT_DIR / "phaseB"),
    log_every_n_steps=PB_LOG_EVERY,
)

trainer_pb = AgentPhaseBTrainer(
    config=cfg_pb,
    slot_model=slot_model,
    vaes=vaes,
    agents=agents,
    aggregator=aggregator,
)

print("Phase B: full agent training ...")
metrics_pb = trainer_pb.train(train_loader)
print(f"Phase B done: {metrics_pb}")

# Freeze agents after task-0 Phase B
for ag in agents:
    for p in ag.parameters():
        p.requires_grad_(False)
print("Agents frozen")

# %% [markdown]
# ## 13. SLDA — Fit on Task 0

# %%
slda = StreamLDA(
    n_classes=N_CLASSES,
    feature_dim=D_H,
    shrinkage=SLDA_SHRINKAGE,
)

cfg_slda = SLDAConfig(
    feature_dim=D_H,
    n_classes=N_CLASSES,
    shrinkage=SLDA_SHRINKAGE,
    max_batches=0,
    device=str(DEVICE),
)

trainer_slda = SLDATrainer(
    config=cfg_slda,
    slot_model=slot_model,
    agents=agents,
    aggregator=aggregator,
    slda=slda,
    vaes=vaes,
)

print("SLDA: fitting on task 0 ...")
trainer_slda.fit(train_loader)
print(f"SLDA fitted  ({slda._n_total} samples)")

# %% [markdown]
# ## 14. Evaluate Task 0  *(seed the forgetting matrix)*

# %%
val_m0  = trainer_slda.evaluate(val_loader)
test_m0 = trainer_slda.evaluate(test_loader)

print("=" * 45)
print(f"  Task 0 val  : {val_m0['accuracy']*100:.2f}%")
print(f"  Task 0 test : {test_m0['accuracy']*100:.2f}%")
print("=" * 45)

# ---------------------------------------------------------------------------
# Forgetting matrix  R[i][j] = test-acc on task i  after  training up to task j
#
#   R[i][i]     -> accuracy right after learning task i (best case)
#   R[i][j > i] -> accuracy on task i after learning more tasks (shows forgetting)
#
# After all N_TASKS are trained:
#   Forgetting_i  = R[i][i] - R[i][N_TASKS-1]
#   BWT           = (1/(T-1)) * sum_{i=0}^{T-2} (R[i][T-1] - R[i][i])
# ---------------------------------------------------------------------------
R       = [[None] * N_TASKS for _ in range(N_TASKS)]
R[0][0] = test_m0["accuracy"]

task_results = [{"task": 0,
                 "val_acc":  val_m0["accuracy"],
                 "test_acc": test_m0["accuracy"]}]

print("Forgetting matrix initialised (R[0][0] set).")

# %% [markdown]
# ## 15. Continual Learning — Tasks 1+
#
# For each new task t:
# 1. **Phase B** fine-tune existing agents on task t data
# 2. **SLDA** incremental update on task t data
# 3. **Evaluate all seen tasks 0..t** -> fill a column of the forgetting matrix
# 4. Print per-task accuracy and running BWT

# %%
for t in range(1, N_TASKS):
    print(f"\n{'='*60}")
    print(f"  TASK {t}  |  classes "
          f"{t * CLASSES_PER_TASK} - {(t + 1) * CLASSES_PER_TASK - 1}")
    print(f"{'='*60}")

    t_train, t_val, t_test = get_task_loaders(t)

    # -----------------------------------------------------------------------
    # Phase B: fine-tune agents on task t
    # -----------------------------------------------------------------------
    for ag in agents:
        for p in ag.parameters():
            p.requires_grad_(True)

    trainer_pb_t = AgentPhaseBTrainer(
        config=cfg_pb,
        slot_model=slot_model,
        vaes=vaes,
        agents=agents,
        aggregator=aggregator,
    )
    print(f"  Task {t} - Phase B ...")
    trainer_pb_t.train(t_train)

    for ag in agents:
        for p in ag.parameters():
            p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # SLDA: incremental update on task t
    # -----------------------------------------------------------------------
    print(f"  Task {t} - SLDA update ...")
    trainer_slda.fit(t_train)

    # -----------------------------------------------------------------------
    # Evaluate ALL seen tasks  ->  fill column t of the forgetting matrix
    # -----------------------------------------------------------------------
    print(f"  Task {t} - evaluating all seen tasks ...")
    for i in range(t + 1):
        _, _, i_test = get_task_loaders(i)
        m      = trainer_slda.evaluate(i_test)
        R[i][t] = m["accuracy"]
        print(f"    R[task={i}][after_task={t}] = {m['accuracy']*100:.2f}%")

    t_val_m = trainer_slda.evaluate(t_val)
    task_results.append({
        "task":     t,
        "val_acc":  t_val_m["accuracy"],
        "test_acc": R[t][t],
    })

    # Running BWT (average backward transfer over tasks learned before t)
    bwt_terms = [
        R[i][t] - R[i][i]
        for i in range(t)
        if R[i][t] is not None and R[i][i] is not None
    ]
    running_bwt  = sum(bwt_terms) / len(bwt_terms) if bwt_terms else float("nan")
    avg_acc_now  = sum(R[i][t] for i in range(t + 1) if R[i][t] is not None) / (t + 1)

    print(f"\n  After task {t}:")
    print(f"    Avg test acc (tasks 0-{t}) : {avg_acc_now * 100:.2f}%")
    print(f"    Running BWT               : {running_bwt * 100:+.2f}%")
    print(f"    SLDA total samples        : {slda._n_total}")

# %% [markdown]
# ## 16. Forgetting Analysis & Plots
#
# * Bar chart: accuracy right after learning vs. accuracy at end of all tasks
# * Forgetting bar chart with BWT annotation
# * Full forgetting matrix heatmap (R[i][j])

# %%
last_t = len(task_results) - 1   # index of the last trained task

acc_initial = []   # R[i][i]      - right after learning task i
acc_final   = []   # R[i][last_t] - after all training
forgetting  = []   # acc_initial[i] - acc_final[i]

for i in range(last_t + 1):
    ai = R[i][i]
    af = R[i][last_t]
    acc_initial.append(ai if ai is not None else float("nan"))
    acc_final.append(  af if af is not None else float("nan"))
    forgetting.append(
        (ai - af) if (ai is not None and af is not None) else float("nan")
    )

bwt_terms = [
    R[i][last_t] - R[i][i]
    for i in range(last_t)
    if R[i][last_t] is not None and R[i][i] is not None
]
bwt = float(np.nanmean(bwt_terms)) if bwt_terms else float("nan")

# ── Console report ───────────────────────────────────────────────────────────
print("=" * 65)
print(f"  Final results  ({last_t + 1} tasks trained)")
print("  " + "-" * 61)
print(f"  {'Task':>5}  {'acc@learn':>10}  {'acc@end':>9}  {'forgetting':>11}")
print("  " + "-" * 61)
for i in range(last_t + 1):
    print(f"  {i:>5}  {acc_initial[i]*100:>9.2f}%"
          f"  {acc_final[i]*100:>8.2f}%"
          f"  {forgetting[i]*100:>+10.2f}%")
print("  " + "-" * 61)
print(f"  BWT            : {bwt*100:+.2f}%")
print(f"  Avg final acc  : {float(np.nanmean(acc_final))*100:.2f}%")
print("=" * 65)

# ── Save JSON ────────────────────────────────────────────────────────────────
cl_metrics = {
    "n_tasks":      N_TASKS,
    "last_trained": last_t,
    "task_results": task_results,
    "acc_initial":  [float(x) for x in acc_initial],
    "acc_final":    [float(x) for x in acc_final],
    "forgetting":   [float(x) for x in forgetting],
    "bwt":          float(bwt),
    "avg_final_acc": float(np.nanmean(acc_final)),
    "R": [[R[i][j] for j in range(N_TASKS)] for i in range(N_TASKS)],
}
with open(OUTPUT_DIR / "cl_metrics.json", "w") as f:
    json.dump(cl_metrics, f, indent=2)
print(f"Metrics saved -> {OUTPUT_DIR / 'cl_metrics.json'}")

# ── Plots ────────────────────────────────────────────────────────────────────
matplotlib.use("Agg")
x = np.arange(last_t + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# --- Plot 1: per-task accuracy (initial vs final) ---
ax = axes[0]
w  = 0.35
ax.bar(x - w/2, [v * 100 for v in acc_initial], w,
       label="acc@learn", color="steelblue")
ax.bar(x + w/2, [v * 100 for v in acc_final], w,
       label="acc@end",   color="salmon")
ax.set_xlabel("Task")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Per-task accuracy")
ax.set_xticks(x)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# --- Plot 2: forgetting per task ---
ax = axes[1]
colours = ["red" if f > 0 else "green" for f in forgetting]
ax.bar(x, [f * 100 for f in forgetting], color=colours)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Task")
ax.set_ylabel("Forgetting (%)")
ax.set_title(f"Forgetting  |  BWT = {bwt*100:+.2f}%")
ax.set_xticks(x)
ax.grid(axis="y", alpha=0.3)

# --- Plot 3: forgetting matrix heatmap ---
ax = axes[2]
if last_t > 0:
    mat = np.full((last_t + 1, last_t + 1), float("nan"))
    for i in range(last_t + 1):
        for j in range(last_t + 1):
            if R[i][j] is not None:
                mat[i, j] = R[i][j] * 100
    im = ax.imshow(mat, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Test acc (%)")
    ax.set_xlabel("After task j")
    ax.set_ylabel("Task i")
    ax.set_title("Forgetting matrix R[i][j]")
    ax.set_xticks(range(last_t + 1))
    ax.set_yticks(range(last_t + 1))
    for i in range(last_t + 1):
        for j in range(last_t + 1):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if mat[i, j] > 50 else "white")
else:
    ax.text(0.5, 0.5, "Only 1 task trained\n(N_TASKS = 1)",
            ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")

plt.suptitle(
    f"Continual Learning — {DATASET}  ({last_t+1}/{N_TASKS} tasks)",
    y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cl_analysis.png", dpi=130, bbox_inches="tight")
plt.show()
plt.close()
print("CL analysis plot saved")

# Phase 0 training curves
hist_path = OUTPUT_DIR / "history_p0.json"
if hist_path.exists():
    with open(hist_path) as f:
        h = json.load(f)
    keys = [k for k in h if k != "step" and h[k]]
    step = h.get("step", list(range(len(h[keys[0]]))))
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in keys:
        ax.plot(step, h[k], label=k)
    ax.set_title("Phase 0 losses")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves_p0.png", dpi=120)
    plt.show()
    plt.close()
    print("Phase 0 loss curves saved")

# %% [markdown]
# ## 17. Save Final Checkpoint

# %%
ckpt_out = OUTPUT_DIR / "pipeline_final.pt"
torch.save({
    "slot_model":    slot_model.state_dict(),
    "aggregator":    aggregator.state_dict(),
    "agents":        [a.state_dict() for a in agents],
    "slda":          slda.state_dict(),
    "vaes":          [v.state_dict() for v in vaes],
    "task_results":  task_results,
    "cl_metrics":    cl_metrics,
    "config": {
        "dataset":           DATASET,
        "n_classes":         N_CLASSES,
        "n_tasks":           N_TASKS,
        "classes_per_task":  CLASSES_PER_TASK,
        "cluster_method":    CLUSTER_METHOD,
        "n_agents":          M,
        "p0_epochs":         P0_EPOCHS,
        "pa_epochs":         PA_EPOCHS,
        "pb_epochs":         PB_EPOCHS,
    },
}, ckpt_out)
print(f"Checkpoint saved: {ckpt_out}")

# %% [markdown]
# ## 18. Summary

# %%
ckpts = sorted(OUTPUT_DIR.rglob("*.pt"))
print("=" * 65)
print(f"  Dataset       : {DATASET}")
print(f"  Tasks trained : {last_t + 1} / {N_TASKS}  ({CLASSES_PER_TASK} classes/task)")
print(f"  N agents      : {M}")
print()
print(f"  {'Task':>5}  {'val %':>8}  {'test %':>8}  {'forgetting':>12}")
print(f"  {'-'*45}")
for i, r in enumerate(task_results):
    fg = forgetting[i] * 100 if not np.isnan(forgetting[i]) else float("nan")
    print(f"  {r['task']:>5}  {r['val_acc']*100:>8.2f}  "
          f"{r['test_acc']*100:>8.2f}  {fg:>+11.2f}%")
print(f"  {'-'*45}")
print(f"  Avg final test acc : {float(np.nanmean(acc_final))*100:.2f}%")
print(f"  BWT                : {bwt*100:+.2f}%")
print(f"  SLDA samples seen  : {slda._n_total}")
print()
print(f"  Output dir    : {OUTPUT_DIR}")
print(f"  Checkpoints   : {len(ckpts)}")
for c in ckpts:
    print(f"    {c.name}  ({c.stat().st_size/1e6:.1f} MB)")
print("=" * 65)
print("Done! Download checkpoints + plots from the Output tab on Kaggle.")
