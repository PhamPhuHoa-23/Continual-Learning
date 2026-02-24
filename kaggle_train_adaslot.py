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
# # Train AdaSlot on Kaggle with Primitive Loss
#
# This notebook trains AdaSlot for 2000+ epochs on full CIFAR-100 Task 1 dataset.
#
# **Features:**
# - Full dataset (no test_mode)
# - 2000 epochs for thorough training
# - Primitive Loss (CompSLOT paper)
# - Automatic checkpointing
# - Kaggle GPU acceleration

# %% [markdown]
# ## 1. Setup Environment

# %%
import os
import sys
import subprocess
from pathlib import Path

# Kaggle paths
KAGGLE_WORKING = Path("/kaggle/working")
REPO_NAME = "Continual-Learning"
REPO_PATH = KAGGLE_WORKING / REPO_NAME

print(f"Working directory: {os.getcwd()}")
print(f"Kaggle working: {KAGGLE_WORKING}")
print(f"Repo path: {REPO_PATH}")

# %% [markdown]
# ## 2. Clone Repository

# %%
# Clone your repository
if not REPO_PATH.exists():
    print("Cloning repository...")
    !git clone https://github.com/PhamPhuHoa-23/Continual-Learning.git /kaggle/working/Continual-Learning
    !cd /kaggle/working/Continual-Learning && git checkout prototype
    print("✅ Repository cloned successfully!")
else:
    print("✅ Repository already exists")
    !cd /kaggle/working/Continual-Learning && git pull origin prototype

# Change to repo directory
os.chdir(REPO_PATH)
print(f"Current directory: {os.getcwd()}")

# %% [markdown]
# ## 3. Install Dependencies

# %%
print("Installing dependencies...")

# Core dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional requirements
!pip install -q tqdm numpy matplotlib scikit-learn hdbscan umap-learn

# Install in development mode
!pip install -q -e .

print("✅ Dependencies installed!")

# Verify imports
try:
    import torch
    import torchvision
    from src.models.adaslot.model import AdaSlotModel
    from src.losses.primitive import PrimitiveSelector, ConceptLearningLoss
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    raise

# %% [markdown]
# ## 4. Check GPU and System Info

# %%
import torch

print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU available! Training will be slow.")

print("=" * 60)

# %% [markdown]
# ## 5. Setup CIFAR-100 Data
#
# CIFAR-100 will be automatically downloaded by torchvision.

# %%
import torchvision.datasets as datasets

# Download CIFAR-100 (will cache in ~/.cache or /root/.cache)
print("Downloading CIFAR-100...")
_ = datasets.CIFAR100(root='./data', train=True, download=True)
_ = datasets.CIFAR100(root='./data', train=False, download=True)
print("✅ CIFAR-100 downloaded!")

# %% [markdown]
# ## 6. Training Configuration

# %%
# Training hyperparameters
CONFIG = {
    # Model
    "num_slots": 7,
    "slot_dim": 64,
    "adaslot_resolution": 128,
    
    # Data
    "n_tasks": 10,
    "n_classes_per_task": 10,
    "batch_size": 64,
    "workers": 2,  # Kaggle can handle 2 workers
    
    # Training
    "epochs": 2000,  # Long training for convergence
    "lr": 3e-4,
    "sparse_weight": 1.0,
    
    # Primitive Loss (CompSLOT Paper)
    "use_primitive_loss": True,
    "primitive_alpha": 10.0,
    "primitive_temp": 10.0,
    
    # Checkpointing
    "save_interval": 100,  # Save every 100 epochs
    "output_dir": "checkpoints/adaslot_runs",
    
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

print("=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"{key:25s}: {value}")
print("=" * 60)

# %% [markdown]
# ## 7. Import Training Modules

# %%
import argparse
import json
import random
import logging
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm
import torch.nn.functional as F

from src.models.adaslot.model import AdaSlotModel
from src.data.continual_cifar100 import get_continual_cifar100_loaders
from src.losses.primitive import PrimitiveSelector, ConceptLearningLoss

print("✅ Modules imported successfully!")

# %% [markdown]
# ## 8. Setup Functions

# %%
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_run_directory(base_dir: str):
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"kaggle_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_config(config: dict, run_dir: Path):
    """Save configuration to JSON."""
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved to {config_path}")

# Set seed
set_seed(CONFIG["seed"])
print(f"✅ Random seed set to {CONFIG['seed']}")

# %% [markdown]
# ## 9. Load Data (Task 1 only)

# %%
print("Loading CIFAR-100 Task 1...")

train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
    n_tasks=CONFIG["n_tasks"],
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["workers"],
    seed=CONFIG["seed"],
    max_samples_per_task=None,  # Full dataset
)

# Use only Task 1
train_loader = train_loaders[0]
test_loader = test_loaders[0]
task1_classes = class_order[:CONFIG["n_classes_per_task"]]

print(f"✅ Task 1 classes: {task1_classes}")
print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Test batches: {len(test_loader)}")
print(f"✅ Train samples: {len(train_loader.dataset)}")
print(f"✅ Test samples: {len(test_loader.dataset)}")

# %% [markdown]
# ## 10. Initialize Model

# %%
print("Initializing AdaSlot model...")

# Create model
model = AdaSlotModel(
    num_slots=CONFIG["num_slots"],
    slot_dim=CONFIG["slot_dim"]
)
model.to(CONFIG["device"])

# Create primitive selector
primitive_selector = PrimitiveSelector(
    slot_dim=CONFIG["slot_dim"],
    temperature=None  # Auto: 100/√D_s
).to(CONFIG["device"])

# Create concept learning loss
concept_loss_fn = ConceptLearningLoss(
    alpha=CONFIG["primitive_alpha"],
    temperature_p=CONFIG["primitive_temp"]
).to(CONFIG["device"])

# Optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(primitive_selector.parameters()),
    lr=CONFIG["lr"]
)

print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✅ Selector parameters: {sum(p.numel() for p in primitive_selector.parameters()):,}")
print(f"✅ Total trainable parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in primitive_selector.parameters()):,}")

# %% [markdown]
# ## 11. Create Output Directory

# %%
RUN_DIR = create_run_directory(CONFIG["output_dir"])
save_config(CONFIG, RUN_DIR)
print(f"✅ Run directory: {RUN_DIR}")

# %% [markdown]
# ## 12. Training Loop

# %%
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Epochs: {CONFIG['epochs']}")
print(f"Device: {CONFIG['device']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Primitive Loss: α={CONFIG['primitive_alpha']}, τ_p={CONFIG['primitive_temp']}")
print("=" * 60)

# Training history
history = {
    'epoch': [],
    'train_loss': [],
    'train_recon': [],
    'train_primitive': [],
    'train_sparse': [],
    'test_loss': [],
}

best_test_loss = float('inf')

# Main training loop
for epoch in range(CONFIG["epochs"]):
    # ─── Training ───────────────────────────────────────────────────────
    model.train()
    primitive_selector.train()
    
    epoch_losses = {
        'total': 0.0,
        'recon': 0.0,
        'primitive': 0.0,
        'sparse': 0.0
    }
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False)
    
    for X, y in pbar:
        X = X.to(CONFIG["device"])
        y = y.to(CONFIG["device"])
        
        # Resize to AdaSlot resolution
        if X.shape[-1] != CONFIG["adaslot_resolution"]:
            X = F.interpolate(
                X, size=(CONFIG["adaslot_resolution"], CONFIG["adaslot_resolution"]),
                mode="bilinear", align_corners=False
            )
        
        # Forward
        out = model(X, global_step=epoch)
        
        # Extract primitives
        primitives, weights = primitive_selector(out["slots"])
        
        # Compute losses
        losses = concept_loss_fn(
            reconstructed=out["reconstruction"],
            target=X,
            primitives=primitives,
            labels=y
        )
        
        loss = losses['total']
        loss_recon = losses['recon']
        loss_prim = losses['primitive']
        
        # Sparsity penalty
        loss_sparse = CONFIG["sparse_weight"] * out["hard_keep_decision"].float().mean()
        loss = loss + loss_sparse
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        torch.nn.utils.clip_grad_norm_(primitive_selector.parameters(), 100.0)
        optimizer.step()
        
        # Track losses
        epoch_losses['total'] += loss.item()
        epoch_losses['recon'] += loss_recon.item()
        epoch_losses['primitive'] += loss_prim.item()
        epoch_losses['sparse'] += loss_sparse.item()
        n_batches += 1
        
        pbar.set_postfix({
            'recon': f"{loss_recon.item():.2f}",
            'prim': f"{loss_prim.item():.3f}",
            'sparse': f"{loss_sparse.item():.3f}"
        })
    
    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(n_batches, 1)
    
    # ─── Evaluation ─────────────────────────────────────────────────────
    model.eval()
    test_recon_loss = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(CONFIG["device"])
            
            if X.shape[-1] != CONFIG["adaslot_resolution"]:
                X = F.interpolate(
                    X, size=(CONFIG["adaslot_resolution"], CONFIG["adaslot_resolution"]),
                    mode="bilinear", align_corners=False
                )
            
            out = model(X, global_step=epoch)
            loss = F.mse_loss(out["reconstruction"], X, reduction="mean")
            test_recon_loss += loss.item()
            test_batches += 1
    
    avg_test_loss = test_recon_loss / max(test_batches, 1)
    
    # ─── Logging ────────────────────────────────────────────────────────
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(epoch_losses['total'])
    history['train_recon'].append(epoch_losses['recon'])
    history['train_primitive'].append(epoch_losses['primitive'])
    history['train_sparse'].append(epoch_losses['sparse'])
    history['test_loss'].append(avg_test_loss)
    
    # Print progress every 10 epochs or milestones
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train: {epoch_losses['total']:.4f} "
              f"(recon={epoch_losses['recon']:.2f}, prim={epoch_losses['primitive']:.3f}, sparse={epoch_losses['sparse']:.3f}) | "
              f"Test: {avg_test_loss:.4f}")
    
    # ─── Checkpointing ──────────────────────────────────────────────────
    if (epoch + 1) % CONFIG["save_interval"] == 0:
        ckpt_path = RUN_DIR / f"adaslot_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'primitive_selector_state_dict': primitive_selector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': epoch_losses,
            'test_loss': avg_test_loss,
            'history': history,
            'config': CONFIG
        }, ckpt_path)
        print(f"✅ Checkpoint saved: {ckpt_path}")
    
    # Save best model
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_path = RUN_DIR / "adaslot_best.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'primitive_selector_state_dict': primitive_selector.state_dict(),
            'test_loss': avg_test_loss,
            'config': CONFIG
        }, best_path)
        print(f"✅ Best model updated (test_loss={avg_test_loss:.4f}): {best_path}")

print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

# %% [markdown]
# ## 13. Final Save

# %%
# Save final model
final_path = RUN_DIR / "adaslot_final.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'primitive_selector_state_dict': primitive_selector.state_dict(),
    'history': history,
    'config': CONFIG
}, final_path)
print(f"✅ Final model saved: {final_path}")

# Save training history
history_path = RUN_DIR / "training_history.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=4)
print(f"✅ Training history saved: {history_path}")

# %% [markdown]
# ## 14. Plot Training Curves

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training loss
axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train Loss')
axes[0, 0].plot(history['epoch'], history['test_loss'], label='Test Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Reconstruction loss
axes[0, 1].plot(history['epoch'], history['train_recon'], label='Train Recon')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Reconstruction Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Primitive loss
axes[1, 0].plot(history['epoch'], history['train_primitive'], label='Primitive Loss', color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Primitive Loss (Intra-class Consistency)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Sparse loss
axes[1, 1].plot(history['epoch'], history['train_sparse'], label='Sparse Loss', color='orange')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Sparsity Penalty')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RUN_DIR / "training_curves.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Training curves saved: {plot_path}")

# %% [markdown]
# ## 15. Visualize Slots

# %%
print("Visualizing slot attention masks...")

# Load best model
best_ckpt = torch.load(RUN_DIR / "adaslot_best.pt")
model.load_state_dict(best_ckpt['model_state_dict'])
model.eval()

# Get a batch of test images
test_iter = iter(test_loader)
X, y = next(test_iter)
X = X[:8].to(CONFIG["device"])  # First 8 images

# Resize
if X.shape[-1] != CONFIG["adaslot_resolution"]:
    X_resized = F.interpolate(
        X, size=(CONFIG["adaslot_resolution"], CONFIG["adaslot_resolution"]),
        mode="bilinear", align_corners=False
    )
else:
    X_resized = X

# Forward pass
with torch.no_grad():
    out = model(X_resized, global_step=0)

slots = out["slots"]  # (B, K, D)
recon = out["reconstruction"]  # (B, 3, H, W)
masks = out.get("masks", None)  # (B, K, H, W) if available

# Visualize
B, K, D = slots.shape

fig, axes = plt.subplots(B, K + 2, figsize=(3 * (K + 2), 3 * B))
if B == 1:
    axes = axes.reshape(1, -1)

for b in range(B):
    # Original
    img = X[b].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[b, 0].imshow(img)
    axes[b, 0].set_title(f"Class {y[b].item()}")
    axes[b, 0].axis('off')
    
    # Reconstruction
    recon_img = recon[b].cpu().permute(1, 2, 0).numpy()
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
    axes[b, 1].imshow(recon_img)
    axes[b, 1].set_title("Recon")
    axes[b, 1].axis('off')
    
    # Slot masks
    if masks is not None:
        for k in range(K):
            mask = masks[b, k].cpu().numpy()
            axes[b, k + 2].imshow(mask, cmap='hot')
            axes[b, k + 2].set_title(f"Slot {k}")
            axes[b, k + 2].axis('off')

plt.tight_layout()
viz_path = RUN_DIR / "slot_visualization.png"
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Slot visualization saved: {viz_path}")

# %% [markdown]
# ## 16. Summary

# %%
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Total epochs: {CONFIG['epochs']}")
print(f"Best test loss: {best_test_loss:.4f}")
print(f"Final test loss: {history['test_loss'][-1]:.4f}")
print(f"Run directory: {RUN_DIR}")
print(f"Best checkpoint: {RUN_DIR / 'adaslot_best.pt'}")
print(f"Final checkpoint: {RUN_DIR / 'adaslot_final.pt'}")
print("=" * 60)

# List all checkpoints
checkpoints = sorted(RUN_DIR.glob("*.pt"))
print(f"\nSaved checkpoints ({len(checkpoints)}):")
for ckpt in checkpoints:
    size_mb = ckpt.stat().st_size / 1e6
    print(f"  - {ckpt.name} ({size_mb:.1f} MB)")

print("\n✅ All done! You can now download the checkpoints from Kaggle output.")

# %% [markdown]
# ## 17. Download Instructions
#
# **To download checkpoints from Kaggle:**
#
# 1. Go to Kaggle notebook's Output tab
# 2. Navigate to `checkpoints/adaslot_runs/kaggle_run_xxx/`
# 3. Download:
#    - `adaslot_best.pt` - Best model (lowest test loss)
#    - `adaslot_final.pt` - Final model
#    - `training_history.json` - Loss curves
#    - `training_curves.png` - Visualization
#
# **Or use Kaggle API:**
# ```bash
# kaggle kernels output [username]/[kernel-name] -p ./checkpoints
# ```
#
# **Use checkpoint locally:**
# ```python
# import torch
# from src.models.adaslot.model import AdaSlotModel
#
# ckpt = torch.load('adaslot_best.pt')
# model = AdaSlotModel(num_slots=7, slot_dim=64)
# model.load_state_dict(ckpt['model_state_dict'])
# ```
