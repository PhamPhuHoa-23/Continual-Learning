# Kaggle Training Guide for AdaSlot

This guide helps you train AdaSlot on Kaggle with full dataset and 2000+ epochs.

## 📦 Files

- **`kaggle_train_adaslot.ipynb`** - Jupyter notebook (recommended for direct upload)
- **`kaggle_train_adaslot.py`** - Jupytext format (alternative)

## 🚀 Quick Start

### Option 1: Direct Upload to Kaggle

1. **Upload Notebook**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Click "File" → "Upload Notebook"
   - Select `kaggle_train_adaslot.ipynb`

2. **Enable GPU**
   - Click "Settings" (right sidebar)
   - Under "Accelerator", select **GPU T4 x2** or **GPU P100**
   - Click "Save"

3. **Enable Internet**
   - In Settings, turn ON "Internet"
   - Required to clone GitHub repo

4. **Run All Cells**
   - Click "Run All" or press `Shift+Enter` through cells
   - Training will start automatically

### Option 2: Import from GitHub

1. **Create New Kaggle Notebook**
2. **In First Cell:**
   ```python
   !git clone https://github.com/PhamPhuHoa-23/Continual-Learning.git /kaggle/working/Continual-Learning
   !cd /kaggle/working/Continual-Learning && git checkout prototype
   ```
3. **Copy-paste cells** from `kaggle_train_adaslot.ipynb`

## ⚙️ Configuration

Default settings in the notebook:

```python
CONFIG = {
    # Model
    "num_slots": 7,
    "slot_dim": 64,
    "adaslot_resolution": 128,
    
    # Data
    "batch_size": 64,
    "workers": 2,
    
    # Training
    "epochs": 2000,  # ← Change this for longer/shorter training
    "lr": 3e-4,
    
    # Primitive Loss
    "primitive_alpha": 10.0,
    "primitive_temp": 10.0,
    
    # Checkpointing
    "save_interval": 100,  # Save every 100 epochs
}
```

**To modify:**
- Edit the `CONFIG` dictionary in Cell #6
- Or add arguments after the cell

## 📊 What Happens During Training

### Phase 1: Setup (5-10 minutes)
- Clones repository from GitHub
- Installs PyTorch, torchvision, scikit-learn, etc.
- Downloads CIFAR-100 dataset (~170MB)

### Phase 2: Training (varies by epochs)
- **2000 epochs** ≈ 6-8 hours on GPU T4 x2
- **1000 epochs** ≈ 3-4 hours
- **500 epochs** ≈ 1.5-2 hours

### Phase 3: Checkpointing
- Saves checkpoint every 100 epochs
- Total checkpoints: `epochs / save_interval + 2`
  - Example: 2000 epochs → 22 files (20 interval + best + final)

## 💾 Output Files

After training, you'll find:

```
/kaggle/working/Continual-Learning/checkpoints/adaslot_runs/kaggle_run_YYYYMMDD_HHMMSS/
├── adaslot_best.pt                 # Best model (lowest test loss)
├── adaslot_final.pt                # Final model
├── adaslot_epoch100.pt             # Checkpoint at epoch 100
├── adaslot_epoch200.pt             # Checkpoint at epoch 200
├── ...
├── adaslot_epoch2000.pt            # Checkpoint at epoch 2000
├── config.json                     # Training configuration
├── training_history.json           # Loss curves data
├── training_curves.png             # Loss visualization
└── slot_visualization.png          # Slot attention masks
```

## 📥 Download Checkpoints

### Method 1: Kaggle UI
1. Click **"Output"** tab (right sidebar)
2. Navigate to `checkpoints/adaslot_runs/kaggle_run_xxx/`
3. Click download icon next to each file

### Method 2: Kaggle API (Local Machine)
```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials (download from https://www.kaggle.com/settings)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download outputs
kaggle kernels output [username]/[kernel-name] -p ./checkpoints
```

### Method 3: Create Dataset
1. Click "Save Version"
2. After completion, go to "Output" tab
3. Click "New Dataset" button
4. This creates a Kaggle dataset with all outputs
5. You can use this dataset in other notebooks

## 🔄 Resume Training

If Kaggle session times out, you can resume:

```python
# In notebook, add before training loop:
checkpoint_path = "/kaggle/input/your-dataset/adaslot_epoch1000.pt"

if Path(checkpoint_path).exists():
    print("Resuming from checkpoint...")
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model_state_dict'])
    primitive_selector.load_state_dict(ckpt['primitive_selector_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    history = ckpt['history']
else:
    start_epoch = 0
    history = {...}

# Change: for epoch in range(CONFIG["epochs"])
# To:     for epoch in range(start_epoch, CONFIG["epochs"])
```

## 🔍 Monitor Training

### Real-time Progress

The notebook shows:
```
Epoch 10/2000 | Train: 45.2341 (recon=42.15, prim=0.087, sparse=2.997) | Test: 40.1234
Epoch 20/2000 | Train: 38.5612 (recon=35.88, prim=0.072, sparse=2.609) | Test: 35.8901
...
```

### Training Curves

Cell #14 generates plots:
- **Total Loss**: Train vs Test
- **Reconstruction Loss**: Image reconstruction quality
- **Primitive Loss**: Intra-class consistency
- **Sparsity Penalty**: Slot selection sparsity

### Slot Visualization

Cell #15 shows:
- Original images
- Reconstructions
- Individual slot attention masks

## 🎯 Expected Results

### Loss Progression (2000 epochs)

| Epoch | Train Loss | Test Loss | Primitive Loss |
|-------|-----------|-----------|----------------|
| 100   | ~45-50    | ~42-48    | ~0.08-0.10     |
| 500   | ~35-40    | ~33-38    | ~0.05-0.07     |
| 1000  | ~28-33    | ~27-32    | ~0.03-0.05     |
| 2000  | ~22-27    | ~23-28    | ~0.02-0.04     |

*Note: Actual values depend on initialization and hyperparameters*

### Signs of Good Training

✅ **Good:**
- Test loss decreasing steadily
- Primitive loss converging to ~0.02-0.05
- Slots showing distinct object parts
- Reconstruction quality improving

❌ **Bad:**
- Test loss increasing (overfitting)
- Primitive loss exploding (>1.0)
- All slots look identical (collapse)
- Reconstruction getting worse

## 🛠️ Troubleshooting

### Issue: Out of Memory

**Solution:**
```python
# Reduce batch size
CONFIG["batch_size"] = 32  # or 16

# Reduce resolution
CONFIG["adaslot_resolution"] = 64  # or 96
```

### Issue: Training Too Slow

**Check:**
- GPU enabled? (Settings → Accelerator → GPU T4 x2)
- Using `workers=2`? (default is correct)
- Unnecessary logging? (already optimized)

**Speed up:**
```python
# Reduce save frequency
CONFIG["save_interval"] = 200  # instead of 100

# Train fewer epochs
CONFIG["epochs"] = 1000  # instead of 2000
```

### Issue: Repository Clone Fails

**Check:**
- Internet enabled? (Settings → Internet → ON)
- Correct branch name? (should be `prototype`)

**Manual fix:**
```python
# In Cell #2, add verbose output:
!git clone -v https://github.com/PhamPhuHoa-23/Continual-Learning.git /kaggle/working/Continual-Learning
```

### Issue: Import Errors

**Solution:**
```python
# Re-run Cell #3 (Install Dependencies)
# If still fails, install individually:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install scikit-learn hdbscan umap-learn
!pip install -e /kaggle/working/Continual-Learning
```

### Issue: CIFAR-100 Download Hangs

**Solution:**
```python
# In Cell #5, add timeout:
import socket
socket.setdefaulttimeout(120)

# Then re-run download
```

## 🧪 Quick Test (Before Full Training)

To verify everything works:

```python
# Change in Cell #6:
CONFIG["epochs"] = 10  # Just 10 epochs
CONFIG["save_interval"] = 5
```

Run all cells. Should complete in ~5-10 minutes.

If successful, edit back to 2000 epochs and restart notebook.

## 💡 Optimization Tips

### For Fastest Training (Lower Quality)
```python
CONFIG = {
    "epochs": 500,
    "batch_size": 128,
    "save_interval": 500,  # Only save final
    "adaslot_resolution": 64,
}
```

### For Best Quality (Slower)
```python
CONFIG = {
    "epochs": 5000,
    "batch_size": 32,
    "save_interval": 100,
    "adaslot_resolution": 128,
    "lr": 1e-4,  # Lower learning rate
}
```

### For GPU Memory Efficiency
```python
CONFIG = {
    "batch_size": 16,
    "adaslot_resolution": 96,
    "num_slots": 5,  # Fewer slots
}
```

## 📈 Hyperparameter Tuning

Key hyperparameters to adjust:

### Primitive Loss Alpha (α)
- **Higher (15-20)**: Stronger intra-class consistency, may underfit
- **Lower (5-8)**: Weaker constraint, better reconstruction
- **Default: 10.0** - Good balance

### Primitive Loss Temperature (τ_p)
- **Higher (15-20)**: Softer similarity distribution
- **Lower (5-8)**: Sharper focus on similar samples
- **Default: 10.0** - Standard setting

### Learning Rate
- **Higher (5e-4)**: Faster convergence, may overshoot
- **Lower (1e-4)**: Slower but more stable
- **Default: 3e-4** - Good compromise

### Number of Slots
- **More (10-15)**: More fine-grained decomposition
- **Fewer (3-5)**: Faster training, coarser concepts
- **Default: 7** - From CompSLOT paper

## 🔬 Experiment Tracking

To compare multiple runs:

1. **Change CONFIG**
2. **Save Version** (creates new notebook version)
3. **Compare outputs** across versions

Or use Weights & Biases:

```python
# Add in Cell #7:
!pip install wandb
import wandb

wandb.init(project="adaslot-cifar100", config=CONFIG)

# In training loop:
wandb.log({
    "train_loss": epoch_losses['total'],
    "test_loss": avg_test_loss,
    "epoch": epoch
})
```

## 📚 Use Trained Model Locally

After downloading checkpoint:

```python
import torch
from src.models.adaslot.model import AdaSlotModel
from src.losses.primitive import PrimitiveSelector

# Load checkpoint
ckpt = torch.load('adaslot_best.pt')

# Create model
model = AdaSlotModel(num_slots=7, slot_dim=64)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Create primitive selector
primitive_selector = PrimitiveSelector(slot_dim=64)
primitive_selector.load_state_dict(ckpt['primitive_selector_state_dict'])

# Inference
with torch.no_grad():
    output = model(image_tensor, global_step=0)
    primitives, weights = primitive_selector(output["slots"])
```

## 🎓 Next Steps

After training AdaSlot:

1. **Evaluate on all tasks**
   - Use `train_compositional.py` with pretrained AdaSlot
   - Compare Task 1 performance

2. **Visualize learned concepts**
   - Use `visualize_adaslot.py` with downloaded checkpoint
   - Analyze which slots capture which object parts

3. **Ablation studies**
   - Train without primitive loss (set `use_primitive_loss=False`)
   - Compare reconstruction vs primitive loss weight

4. **Continual learning**
   - Train agent+SLDA on top of frozen AdaSlot
   - Measure forgetting across tasks

## 🆘 Getting Help

If you encounter issues:

1. Check this README
2. Review error messages in notebook output
3. Check GitHub Issues: https://github.com/PhamPhuHoa-23/Continual-Learning/issues
4. Ask in Kaggle notebook comments

## 📝 Citation

If you use this code, please cite the CompSLOT paper:

```bibtex
@inproceedings{compslot2026,
  title={Plug-and-Play Compositionality for Continual Learning},
  author={...},
  booktitle={ICLR},
  year={2026}
}
```

---

**Happy Training! 🚀**

Created: February 24, 2026
