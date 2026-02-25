# Development Rules & Guidelines

This document defines comprehensive rules and guidelines for code development, structure, and reproducibility in this project. **All contributors must follow these rules strictly.**

---

## 📋 Table of Contents

1. [Code Quality Standards](#code-quality-standards)
2. [Research & Documentation First](#research--documentation-first)
3. [Project Structure](#project-structure)
4. [Reproducibility Requirements](#reproducibility-requirements)
5. [Testing Standards](#testing-standards)
6. [Documentation Requirements](#documentation-requirements)
7. [Git & Version Control](#git--version-control)
8. [Dataset Handling](#dataset-handling)
9. [Model & Checkpoint Management](#model--checkpoint-management)
10. [Communication & Reporting](#communication--reporting)

---

## 🎯 Code Quality Standards

### ❌ NEVER DO THIS

1. **NO DUMMY CODE**
   ```python
   # ❌ BAD: Placeholder/dummy implementation
   def train_model():
       pass
   
   # ❌ BAD: Simplified code that doesn't actually work
   def compute_loss():
       return 0.0
   ```

2. **NO INCOMPLETE IMPLEMENTATIONS**
   ```python
   # ❌ BAD: Unfinished code
   def load_data():
       # TODO: implement this later
       return None
   ```

3. **NO HARDCODED VALUES WITHOUT EXPLANATION**
   ```python
   # ❌ BAD: Magic numbers
   loss = x * 0.001 + y * 100
   
   # ✅ GOOD: Documented constants
   RECONSTRUCTION_WEIGHT = 0.001  # Balances recon vs other losses
   PRIMITIVE_WEIGHT = 100  # From CompSLOT paper (Section 3.2)
   loss = x * RECONSTRUCTION_WEIGHT + y * PRIMITIVE_WEIGHT
   ```

### ✅ ALWAYS DO THIS

1. **IMPLEMENT FULLY FUNCTIONAL CODE**
   - Every function must work as intended
   - No placeholders or stubs in production code
   - Test your code before committing

2. **USE PROPER ERROR HANDLING**
   ```python
   # ✅ GOOD: Proper error handling
   def load_checkpoint(path):
       """Load model checkpoint with proper error handling."""
       if not os.path.exists(path):
           raise FileNotFoundError(f"Checkpoint not found: {path}")
       
       try:
           checkpoint = torch.load(path, map_location='cpu')
           return checkpoint
       except Exception as e:
           raise RuntimeError(f"Failed to load checkpoint: {e}")
   ```

3. **WRITE IDIOMATIC, CLEAN CODE**
   - Follow PEP 8 for Python
   - Use descriptive variable names
   - Keep functions focused and small
   - Add type hints where applicable

---

## 🔬 Research & Documentation First

### Before Writing Any Code:

1. **RESEARCH THOROUGHLY**
   - Read the relevant papers (check `docs/` and `paper/`)
   - Search online for similar implementations
   - Check PyTorch/TensorFlow documentation
   - Review existing code in the repository

2. **READ DOCUMENTATION**
   - **Dataset Documentation**: Before using any dataset (CIFAR-100, Tiny-ImageNet, etc.), read:
     - Official dataset documentation
     - Dataset statistics (classes, samples, splits)
     - Expected input/output formats
     - Preprocessing requirements
   
   - **Library Documentation**: Read docs for:
     - PyTorch/TensorFlow APIs
     - Third-party libraries (Avalanche, etc.)
     - Custom modules in this repo

3. **UNDERSTAND THE PROBLEM**
   - Don't code blindly
   - Understand the mathematical formulation
   - Understand the architecture/algorithm
   - Ask questions if unclear

### If You Cannot Implement Something:

**REPORT IT IMMEDIATELY**

```python
# ✅ GOOD: Clear reporting of limitations
def complex_algorithm():
    """
    IMPLEMENTATION NOTE:
    This algorithm requires advanced knowledge of [specific topic].
    Current status: NOT IMPLEMENTED
    
    Reason: [specific reason why it cannot be implemented]
    
    References needed:
    - [Paper/documentation needed]
    - [Expertise required]
    
    TODO: [what needs to be done to complete this]
    """
    raise NotImplementedError(
        "This requires further research on [specific topic]. "
        "See docstring for details."
    )
```

**Don't:**
- Write dummy code pretending it works
- Simplify the algorithm without documentation
- Leave silent bugs

---

## 📁 Project Structure

### Directory Organization

```
Continual-Learning/
├── src/                      # Source code (production)
│   ├── models/              # Model architectures
│   ├── data/                # Data loading & processing
│   ├── losses/              # Loss functions
│   ├── utils/               # Utilities
│   └── ...
├── tests/                   # ALL test files go here
│   ├── test_*.py           # Unit tests
│   └── data/               # Test-specific data utilities
├── configs/                 # Configuration files
├── checkpoints/            # Model checkpoints (gitignored)
├── docs/                   # Documentation
├── paper/                  # LaTeX paper source
├── bins/                   # Deprecated/debug files (gitignored)
│   ├── debug_*.py          # Debug scripts
│   ├── quick_*.py          # Quick test scripts
│   └── *.bat               # Platform-specific scripts
├── train_*.py              # Training scripts (root level OK)
├── config.yaml             # Main config
├── requirements.txt        # Dependencies
├── README.md               # Main documentation
└── RULE.md                 # This file
```

### File Placement Rules

1. **Production Code** → `src/`
   - All reusable modules
   - Model implementations
   - Data loaders
   - Loss functions
   
2. **Tests** → `tests/`
   - Unit tests
   - Integration tests
   - Verification scripts
   - Test utilities

3. **Debug/Temporary** → `bins/`
   - Debug scripts
   - Quick experiments
   - Platform-specific automation (.bat, .sh)
   - Deprecated code

4. **Training Scripts** → Root level OK
   - `train_*.py`
   - `tune_*.py`
   - Keep these accessible for easy execution

5. **Configs** → `configs/` or `config_variants/`
   - YAML configuration files
   - Hyperparameter variants

---

## 🔁 Reproducibility Requirements

### Mandatory for All Experiments

1. **RANDOM SEEDS**
   ```python
   def set_seed(seed: int):
       """Set all random seeds for reproducibility."""
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
           # For deterministic behavior (may impact performance)
           torch.backends.cudnn.deterministic = True
           torch.backends.cudnn.benchmark = False
   ```

2. **CONFIGURATION TRACKING**
   - Save `config.json` with every experiment
   - Include:
     - All hyperparameters
     - Model architecture details
     - Dataset info
     - Random seed
     - Git commit hash
     - Timestamp

3. **CHECKPOINT MANAGEMENT**
   ```python
   # ✅ GOOD: Comprehensive checkpoint
   checkpoint = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'config': config,
       'metrics': metrics,
       'random_state': torch.get_rng_state(),
       'timestamp': datetime.now().isoformat(),
       'git_commit': get_git_commit_hash()
   }
   ```

4. **LOGGING**
   - Log all important metrics
   - Log hyperparameters
   - Log system info (GPU, PyTorch version, CUDA version)
   - Save logs to file, not just stdout

---

## 🧪 Testing Standards

### Test Everything

1. **Unit Tests** for individual functions
   ```python
   def test_slot_attention_forward():
       """Test SlotAttention forward pass shape and values."""
       model = SlotAttention(num_slots=5, slot_dim=64)
       x = torch.randn(2, 16, 128)  # (B, N, D)
       
       slots = model(x)
       
       # Test shape
       assert slots.shape == (2, 5, 64)
       
       # Test no NaN/Inf
       assert not torch.isnan(slots).any()
       assert not torch.isinf(slots).any()
   ```

2. **Integration Tests** for pipelines
   ```python
   def test_training_pipeline():
       """Test full training pipeline runs without errors."""
       config = get_test_config()
       model = create_model(config)
       train_loader = get_test_data()
       
       # Test one training step
       loss = train_one_epoch(model, train_loader, config)
       
       assert loss > 0
       assert not np.isnan(loss)
   ```

3. **Gradient Tests** for custom modules
   ```python
   def test_gradients_flow():
       """Ensure gradients flow through custom module."""
       model = CustomModule()
       x = torch.randn(2, 10, requires_grad=True)
       
       output = model(x)
       loss = output.sum()
       loss.backward()
       
       # Check gradients exist
       assert x.grad is not None
       assert not torch.isnan(x.grad).any()
   ```

### Test File Naming

- `test_<module>.py` for unit tests
- `test_<feature>_integration.py` for integration tests
- All test files must be in `tests/` directory

---

## 📚 Documentation Requirements

### Code Documentation

1. **Docstrings for All Public Functions**
   ```python
   def compute_primitive_loss(
       primitives: torch.Tensor,
       labels: torch.Tensor,
       temperature: float = 10.0
   ) -> torch.Tensor:
       """
       Compute primitive loss for intra-class consistency.
       
       This implements the concept learning loss from the CompSLOT paper
       (Section 3.2), which encourages primitives of the same class to be
       similar in the embedding space.
       
       Args:
           primitives: Primitive representations, shape (B, D)
           labels: Class labels, shape (B,)
           temperature: Temperature parameter for softmax, default 10.0
                       (from paper Table 1)
       
       Returns:
           Scalar loss value
       
       References:
           CompSLOT paper, Section 3.2, Equation 5
       """
   ```

2. **Comments for Complex Logic**
   ```python
   # From CompSLOT paper Eq. 5: L_prim = -log(exp(sim(p_i, p_j)/τ) / Z)
   # where p_i, p_j are primitives from same class
   similarity = F.cosine_similarity(p1, p2, dim=-1)
   loss = -torch.log(torch.exp(similarity / temperature) / partition)
   ```

3. **README for Each Module**
   - `src/models/adaslot/README.md`
   - `src/losses/README.md`
   - Explain the purpose and usage

### Experiment Documentation

Create a markdown file for each major experiment:

```markdown
# Experiment: AdaSlot with Primitive Loss

## Objective
Train AdaSlot on CIFAR-100 Task 1 with primitive loss.

## Configuration
- Model: AdaSlot (7 slots, dim=64)
- Loss: MSE + Primitive Loss (α=10, τ=10)
- Epochs: 2000
- Batch size: 64

## Results
- Final test loss: 0.0234
- Checkpoint: checkpoints/adaslot_runs/run_20260225/best.pt

## Analysis
...
```

---

## 🔧 Git & Version Control

### Commit Guidelines

1. **Clear Commit Messages**
   ```
   feat: Add primitive loss implementation
   
   - Implement PrimitiveSelector module
   - Add ConceptLearningLoss from CompSLOT paper
   - Add unit tests for primitive loss computation
   
   Refs: CompSLOT paper Section 3.2
   ```

2. **Commit Frequency**
   - Commit after each logical change
   - Don't commit broken code
   - Test before committing

3. **What NOT to Commit**
   - Checkpoints (*.pt, *.pth)
   - Large datasets
   - Debug outputs
   - `__pycache__/`
   - Personal config files

### Branch Strategy

- `main`: Production-ready code
- `dev`: Development branch
- `feature/X`: New features
- `bugfix/X`: Bug fixes
- `experiment/X`: Experimental code

---

## 📊 Dataset Handling

### Before Using Any Dataset

1. **READ THE DOCUMENTATION**
   - Official dataset page
   - Paper describing the dataset
   - Preprocessing requirements

2. **VERIFY DATA INTEGRITY**
   ```python
   def verify_cifar100():
       """Verify CIFAR-100 dataset is correctly loaded."""
       train_data = CIFAR100(root='./data', train=True)
       test_data = CIFAR100(root='./data', train=False)
       
       # Check counts
       assert len(train_data) == 50000
       assert len(test_data) == 10000
       
       # Check classes
       assert len(train_data.classes) == 100
       
       print("✓ CIFAR-100 dataset verified")
   ```

3. **DOCUMENT DATA SPLITS**
   ```python
   # Clear documentation of splits
   CIFAR100_TASK_CONFIG = {
       'n_tasks': 10,
       'n_classes_per_task': 10,
       'train_samples_per_class': 500,  # 50000 / 100
       'test_samples_per_class': 100,   # 10000 / 100
       'class_order': [...]  # Explicit class ordering
   }
   ```

### Data Loading Best Practices

```python
def get_data_loader(
    dataset_name: str,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 42
) -> DataLoader:
    """
    Get data loader with proper configuration.
    
    Args:
        dataset_name: Name of dataset ('cifar100', 'tiny-imagenet', etc.)
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
    
    Returns:
        Configured DataLoader
    """
    # Set seed for data shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    dataset = load_dataset(dataset_name)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,  # Reproducible shuffling
        persistent_workers=True if num_workers > 0 else False
    )
    
    return loader
```

---

## 💾 Model & Checkpoint Management

### Checkpoint Naming Convention

```
checkpoints/
├── adaslot_runs/
│   ├── run_20260225_143022/
│   │   ├── config.json
│   │   ├── adaslot_epoch100.pt
│   │   ├── adaslot_epoch200.pt
│   │   ├── adaslot_best.pt
│   │   ├── adaslot_final.pt
│   │   └── training_history.json
│   └── run_20260225_150000/
│       └── ...
```

### Checkpoint Contents

```python
# ✅ GOOD: Complete checkpoint
checkpoint = {
    # Model
    'model_state_dict': model.state_dict(),
    'model_config': model_config,
    
    # Training state
    'epoch': epoch,
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    
    # Metrics
    'train_loss': train_loss,
    'val_loss': val_loss,
    'metrics': metrics_dict,
    
    # Reproducibility
    'config': full_config,
    'random_state': torch.get_rng_state(),
    'numpy_random_state': np.random.get_state(),
    'python_random_state': random.getstate(),
    
    # Metadata
    'timestamp': datetime.now().isoformat(),
    'git_commit': get_git_commit_hash(),
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
}
```

---

## 💬 Communication & Reporting

### When You Encounter Problems

**DO:**
1. Document the problem clearly
2. Show what you tried
3. Show error messages
4. Explain why you're stuck
5. Ask specific questions

**Example:**
```
Problem: Gradient vanishing in SlotAttention module

What I tried:
1. Checked gradient flow - gradients become < 1e-8 after layer 3
2. Tried gradient clipping (max_norm=1.0) - didn't help
3. Reduced learning rate to 1e-5 - still vanishing

Error/Symptom:
- Attention weights become uniform after 10 iterations
- Slots don't differentiate

Questions:
- Should we use layer normalization before attention?
- Is the initialization correct for slot vectors?

References:
- Original Slot Attention paper mentions Xavier init, but unclear for slots
```

### Progress Reporting

For long experiments, create progress logs:

```markdown
## Training Progress: AdaSlot 2000 Epochs

**Started:** 2026-02-25 14:30

### Epoch 100
- Train loss: 0.234
- Test loss: 0.198
- Status: ✓ Converging normally
- Notes: Primitive loss decreasing steadily

### Epoch 500
- Train loss: 0.089
- Test loss: 0.095
- Status: ✓ Good generalization
- Notes: Recon quality improved significantly

### Epoch 1000
- Train loss: 0.045
- Test loss: 0.051
- Status: ⚠️ Slight overfitting
- Notes: May need more regularization

...
```

---

## 🚀 Quick Reference Checklist

Before committing code, verify:

- [ ] Code is fully functional (no dummy/pass implementations)
- [ ] Researched the problem thoroughly
- [ ] Read relevant documentation
- [ ] Added proper error handling
- [ ] Wrote unit tests
- [ ] Added docstrings
- [ ] Removed debug print statements
- [ ] Removed hardcoded paths
- [ ] Set random seeds for reproducibility
- [ ] Saved configuration with checkpoints
- [ ] Files are in correct directories
- [ ] No large files (checkpoints/datasets) committed
- [ ] Code follows PEP 8
- [ ] Tested the code

---

## 📖 Additional Resources

### Key Documents in This Repo

- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration documentation
- [CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md) - Checkpoint management
- [PRIMITIVE_LOSS_GUIDE.md](PRIMITIVE_LOSS_GUIDE.md) - Primitive loss details

### External Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Avalanche Framework](https://avalanche.continualai.org/)
- CompSLOT Paper: Check `docs/` or `paper/`

---

## ⚖️ Final Note

**Quality > Speed**

Take time to:
- Understand what you're implementing
- Research properly
- Write clean, tested code
- Document your work

If something is unclear or you can't implement it properly, **report it**. Don't submit broken or dummy code.

> "Code is read more often than it is written." - Focus on clarity and correctness.

---

**Last Updated:** 2026-02-25  
**Maintainers:** Project Team  
**Questions:** See README.md for contact information
