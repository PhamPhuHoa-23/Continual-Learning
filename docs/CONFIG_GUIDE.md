# Configuration Guide 📋

Complete guide to configuring the Slot Multi-Agent System for continual learning.

---

## Quick Start 🚀

### 1. Load Base Config

```python
from src.utils.config import load_config

# Load default config
cfg = load_config('config.yaml')
cfg.print_summary()

# Access values with dot notation
print(cfg.agents.num_agents)  # 50
print(cfg.selection.strategy)  # 'topk_estimator'
```

### 2. Use Pre-made Variants

```python
# Baseline (adaptive slots, simple top-k)
cfg = load_config('config_variants/01_baseline_adaptive_slots.yaml')

# UCB bandit exploration
cfg = load_config('config_variants/02_ucb_bandit_exploration.yaml')

# Large prototypes (512 instead of 256)
cfg = load_config('config_variants/03_large_prototypes.yaml')

# Ensemble tree
cfg = load_config('config_variants/04_ensemble_tree.yaml')

# Thompson Sampling
cfg = load_config('config_variants/05_thompson_sampling.yaml')

# Fixed slots (no AdaSlot)
cfg = load_config('config_variants/06_fixed_slots.yaml')
```

### 3. Create Custom Experiment

```python
from src.utils.config import create_experiment_config

cfg = create_experiment_config(
    base_config_path='config.yaml',
    experiment_name='my_experiment',
    # Override with double underscore for nested keys
    selection__strategy='bandit_ucb',
    selection__k=5,
    agents__num_prototypes=512,
    training__phase1_agents__epochs=20
)
```

---

## Configuration Sections 📚

### 1. Slot Attention Configuration

**Key Parameters:**

| Parameter | Type | Description | Recommended |
|-----------|------|-------------|-------------|
| `adaptive` | bool | Use AdaSlot (dynamic slot count) | `true` |
| `min_slots` | int | Minimum slots (if adaptive) | `3` |
| `max_slots` | int | Maximum slots (if adaptive) | `10` |
| `num_slots` | int | Fixed slots (if not adaptive) | `7` |
| `slot_dim` | int | Slot embedding dimension | `64` |
| `num_iterations` | int | Attention iterations | `3` |

**Example:**

```yaml
slot_attention:
  adaptive: true      # Enable AdaSlot
  min_slots: 3        # At least 3 slots
  max_slots: 10       # At most 10 slots
  slot_dim: 64        # 64-dim embeddings
  num_iterations: 3   # 3 attention rounds
```

**Why Adaptive?**
- ✅ Different images need different slot counts
- ✅ More flexible representation
- ✅ Better handles occlusion and clutter
- ❌ Slightly slower than fixed

---

### 2. Agents Configuration ⚙️

**Key Parameters:**

| Parameter | Type | Description | Recommended |
|-----------|------|-------------|-------------|
| `num_agents` | int | Total agents in pool | `50` |
| `num_prototypes` | int | **CRITICAL!** Concepts per agent | `256` |
| `hidden_dim` | int | MLP hidden dimension | `256` |
| `num_blocks` | int | Residual blocks | `3` |
| `dropout` | float | Dropout rate | `0.1` |

**IMPORTANT: `num_prototypes` (Output Dimension)**

This is the **dimension of hidden labels** output by each agent!

```yaml
agents:
  num_prototypes: 256  # Each agent outputs 256-dim softmax
```

**Options:**
- `128`: Faster, less expressive
- `256`: **Recommended** (good balance)
- `512`: More expressive, slower
- `1024`: Very expressive, much slower
- `4096+`: Like DINOv2, needs huge batch size

**Rule of thumb:**
- CIFAR-100: 256 is enough
- Tiny-ImageNet: 256-512
- ImageNet: 512-1024

**DINO Training Parameters:**

```yaml
agents:
  dino:
    student_temp: 0.1      # Student temperature (higher = softer)
    teacher_temp: 0.07     # Teacher temperature (lower = sharper)
    center_momentum: 0.9   # EMA for centering
    teacher_momentum: 0.996  # EMA for teacher update
```

**Don't change these unless you know what you're doing!** They're tuned based on DINOv2.

---

### 3. Performance Estimators Configuration 🔍

**Key Parameters:**

| Parameter | Type | Description | Recommended |
|-----------|------|-------------|-------------|
| `type` | str | Estimator type | `'vae'` |

**Options:**
- `'vae'`: VAE-based (reconstruction error)
- `'mlp'`: MLP-based (learned score)
- `'hybrid'`: Combine VAE + MLP

**Example:**

```yaml
estimators:
  type: "vae"
  vae:
    latent_dim: 16
    hidden_dims: [64, 32]
    beta: 1.0
```

**Recommendation:** Start with `'vae'`, it's more interpretable.

---

### 4. Agent Selection Configuration 🎯

**THIS IS WHERE YOU CHOOSE YOUR METHOD!**

**Key Parameters:**

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `strategy` | str | **CRITICAL!** Selection method | See below |
| `k` | int | Top-k agents per slot | `3` |

**Selection Strategies:**

#### a) `'topk_estimator'` - Simple Top-K (Baseline) ⭐ START HERE

```yaml
selection:
  strategy: "topk_estimator"
  k: 3
```

**How it works:**
1. Estimate performance of all 50 agents on a slot (using VAE)
2. Select top-3 agents with best estimated performance
3. No exploration, pure exploitation

**Pros:**
- ✅ Simple and fast
- ✅ Easy to understand
- ✅ No hyperparameters to tune

**Cons:**
- ❌ No exploration (may miss better agents)
- ❌ Gets stuck in local optima

**When to use:** Baseline, debugging, quick experiments

---

#### b) `'bandit_ucb'` - UCB Exploration 🎲

```yaml
selection:
  strategy: "bandit_ucb"
  k: 3
  bandit:
    ucb:
      exploration_constant: 2.0  # Higher = more exploration
```

**How it works:**
- UCB score = mean_reward + c * sqrt(log(t) / count)
- Balances exploitation (high reward) and exploration (low visit count)

**Hyperparameters:**
- `exploration_constant`: 
  - `0.0`: Pure exploitation (same as top-k)
  - `1.0`: Moderate exploration
  - `2.0`: **Recommended** (classic UCB)
  - `5.0+`: Heavy exploration (slower convergence)

**Pros:**
- ✅ Principled exploration-exploitation
- ✅ Provable regret bounds
- ✅ Adapts over time

**Cons:**
- ❌ Slower initially (explores bad agents)
- ❌ Requires reward definition (see below)

**When to use:** After baseline, when you want to explore

---

#### c) `'bandit_thompson'` - Thompson Sampling 🎲🧮

```yaml
selection:
  strategy: "bandit_thompson"
  k: 3
  bandit:
    thompson:
      alpha_init: 1.0  # Prior successes
      beta_init: 1.0   # Prior failures
```

**How it works:**
- Maintains Beta(α, β) distribution for each agent
- Samples from posterior to select agents
- Bayesian approach

**Hyperparameters:**
- `alpha_init`, `beta_init`: Prior parameters
  - `(1, 1)`: Uniform prior (no bias)
  - `(10, 10)`: Stronger prior (slower learning)

**Pros:**
- ✅ Bayesian (more principled than UCB)
- ✅ Naturally handles uncertainty
- ✅ Good for non-stationary rewards

**Cons:**
- ❌ Requires reward in [0, 1] (binary or normalized)
- ❌ More complex than UCB

**When to use:** Advanced experiments, Bayesian approach

---

#### d) `'bandit_epsilon_greedy'` - Epsilon-Greedy 🎲

```yaml
selection:
  strategy: "bandit_epsilon_greedy"
  k: 3
  bandit:
    epsilon_greedy:
      epsilon: 0.1        # 10% random exploration
      epsilon_decay: 0.995  # Decay per step
      min_epsilon: 0.01   # Minimum epsilon
```

**How it works:**
- With prob ε: random agents
- With prob 1-ε: greedy (top-k by estimate)
- ε decays over time

**Pros:**
- ✅ Simple to understand
- ✅ Guaranteed exploration

**Cons:**
- ❌ Random exploration (not intelligent)
- ❌ Hyperparameter sensitive (ε, decay)

**When to use:** Quick & dirty exploration

---

#### e) `'weighted_topk'` - Weighted Baseline

```yaml
selection:
  strategy: "weighted_topk"
  k: 3
```

**How it works:**
- Select top-k by estimate
- Weight by softmax of scores
- No exploration

**When to use:** Same as `topk_estimator`, just with weights

---

### Bandit Rewards ⚠️ IMPORTANT

**If using bandit strategies, you need to define rewards!**

```yaml
training:
  phase2_tree:
    update_bandit: true
    reward_type: "accuracy"  # Options: 'accuracy', 'loss', 'confidence'
```

**Reward Types:**
- `'accuracy'`: 1 if prediction correct, 0 otherwise
- `'loss'`: Negative loss (higher = better)
- `'confidence'`: Tree prediction confidence

**TODO:** Consult professor for optimal reward definition! 📝

---

### 5. Aggregator Configuration 🌳

**Key Parameters:**

| Parameter | Type | Description | Recommended |
|-----------|------|-------------|-------------|
| `type` | str | Tree type | `'hoeffding_adaptive'` |
| `grace_period` | int | Examples before split | `200` |
| `split_confidence` | float | Hoeffding δ | `1e-5` |
| `leaf_prediction` | str | Leaf strategy | `'nba'` |

**Options:**

#### a) `'hoeffding'` - Basic Hoeffding Tree

```yaml
aggregator:
  type: "hoeffding"
  hoeffding:
    grace_period: 200
    split_confidence: 1.0e-5
    leaf_prediction: "nba"
```

**When to use:** Stationary distribution (no concept drift)

---

#### b) `'hoeffding_adaptive'` - Adaptive Hoeffding Tree ⭐ RECOMMENDED

```yaml
aggregator:
  type: "hoeffding_adaptive"
  hoeffding:
    grace_period: 200
    split_confidence: 1.0e-5
    leaf_prediction: "nba"
```

**Difference:** Handles concept drift with ADWIN drift detector

**When to use:** Continual learning (distribution changes over time)

---

#### c) `'ensemble'` - Adaptive Random Forest

```yaml
aggregator:
  type: "ensemble"
  ensemble:
    n_models: 10
    max_features: "sqrt"
    grace_period: 200
    split_confidence: 1.0e-5
```

**Pros:**
- ✅ Better performance (ensemble voting)
- ✅ More robust

**Cons:**
- ❌ Slower (10x inference time)
- ❌ More memory

**When to use:** When accuracy is critical, speed is OK

---

### 6. Training Configuration 🏋️

**Phase 1: Agent Training (DINO SSL)**

```yaml
training:
  phase1_agents:
    enabled: true
    epochs: 10
    batch_size: 32
    learning_rate: 1.0e-3
    weight_decay: 0.04
    clip_grad_norm: 3.0
```

**Key:**
- `epochs`: 10-20 is usually enough
- `batch_size`: 32 for CIFAR, 16 for larger images
- `learning_rate`: 1e-3 is standard, don't change
- `clip_grad_norm`: 3.0 (DINO standard)

**Phase 2: Tree Training (Incremental)**

```yaml
training:
  phase2_tree:
    enabled: true
    eval_every_n_examples: 100
    update_bandit: true  # Only if using bandit strategy
    reward_type: "accuracy"
```

---

## Pretrained Checkpoints 💾

### Loading Pretrained Slot Attention

```yaml
slot_attention:
  adaptive: true
  pretrained:
    enabled: true
    checkpoint_path: "./checkpoints/slot_attention/CLEVR10.ckpt"  # AdaSlot checkpoint
    strict_load: false  # Ignore shape mismatches
    freeze: false       # Fine-tune (set true to freeze)
```

**Available Checkpoints:**
- **AdaSlot CLEVR10**: Pretrained on CLEVR dataset
  - Download: Google Drive (see `Setup/adaslot/`)
  - Path: `./checkpoints/slot_attention/CLEVR10.ckpt`

**When to use:**
- ✅ Transfer learning from CLEVR to your dataset
- ✅ Skip Slot Attention training (faster)
- ✅ Better initialization than random

**Load config:**
```python
cfg = load_config('config_variants/07_pretrained_slot_attention.yaml')
```

---

### Loading Pretrained Agents (Phase 1 → Phase 2)

After training agents with DINO (Phase 1), you can skip Phase 1 and go directly to Phase 2:

```yaml
agents:
  pretrained:
    enabled: true
    checkpoint_path: "./checkpoints/agents/agents_phase1_epoch10.pth"
    load_teacher: true
    freeze: true  # Freeze agents, only train tree

training:
  phase1_agents:
    enabled: false  # Skip Phase 1
  phase2_tree:
    enabled: true   # Only Phase 2
```

**Workflow:**
```bash
# Step 1: Train Phase 1 (agents)
python train_phase1.py --config config.yaml

# Step 2: Load pretrained agents, train Phase 2 (tree)
python train_phase2.py --config config_variants/08_pretrained_agents_phase2.yaml
```

**Load config:**
```python
cfg = load_config('config_variants/08_pretrained_agents_phase2.yaml')
```

---

### Checkpoint Utilities

```python
from src.utils.checkpoint import (
    load_slot_attention_checkpoint,
    load_agent_checkpoint,
    save_agent_checkpoint
)

# Load Slot Attention
from src.models.slot_attention import SlotAttentionAutoEncoder
slot_model = SlotAttentionAutoEncoder(num_slots=7, slot_dim=64)
slot_model = load_slot_attention_checkpoint(
    slot_model,
    './checkpoints/CLEVR10.ckpt',
    strict=False
)

# Load agents
from src.slot_multi_agent import create_agent_pool
students, teachers = create_agent_pool(50, 64, 256)
students, teachers = load_agent_checkpoint(
    students, teachers,
    './checkpoints/agents_phase1_epoch10.pth'
)

# Save agents after Phase 1
save_agent_checkpoint(
    students, teachers, dino_losses,
    './checkpoints/agents_phase1_epoch10.pth',
    epoch=10
)
```

---

## Quick Configs for Different Scenarios 🎯

### Scenario 1: Quick Baseline (Start Here)

```yaml
slot_attention:
  adaptive: true
  min_slots: 3
  max_slots: 10

agents:
  num_agents: 50
  num_prototypes: 256

estimators:
  type: "vae"

selection:
  strategy: "topk_estimator"
  k: 3

aggregator:
  type: "hoeffding_adaptive"
```

**Load:** `config_variants/01_baseline_adaptive_slots.yaml`

---

### Scenario 2: Exploration with UCB

```yaml
selection:
  strategy: "bandit_ucb"
  k: 3
  bandit:
    ucb:
      exploration_constant: 2.0

training:
  phase2_tree:
    update_bandit: true
    reward_type: "accuracy"
```

**Load:** `config_variants/02_ucb_bandit_exploration.yaml`

---

### Scenario 3: High Capacity (More Prototypes)

```yaml
agents:
  num_prototypes: 512
  hidden_dim: 384

training:
  phase1_agents:
    batch_size: 16  # Reduce due to memory
    epochs: 15
```

**Load:** `config_variants/03_large_prototypes.yaml`

---

### Scenario 4: Best Performance (Ensemble)

```yaml
aggregator:
  type: "ensemble"
  ensemble:
    n_models: 10
```

**Load:** `config_variants/04_ensemble_tree.yaml`

---

## Command Line Usage 💻

```bash
# Train with default config
python train.py --config config.yaml

# Train with variant
python train.py --config config_variants/02_ucb_bandit_exploration.yaml

# Override specific values
python train.py --config config.yaml \
    --override selection.strategy=bandit_ucb \
    --override selection.k=5 \
    --override agents.num_prototypes=512
```

---

## Programmatic Usage 🐍

```python
from src.utils.config import load_config, create_experiment_config

# Method 1: Load existing config
cfg = load_config('config_variants/01_baseline_adaptive_slots.yaml')

# Method 2: Create custom config
cfg = create_experiment_config(
    'config.yaml',
    'my_experiment',
    selection__strategy='bandit_ucb',
    agents__num_prototypes=512
)

# Method 3: Load and modify
cfg = load_config('config.yaml')
cfg.selection.strategy = 'bandit_ucb'
cfg.agents.num_prototypes = 512

# Print summary
cfg.print_summary()

# Use in training
from train import train
train(cfg)
```

---

## Validation & Debugging ✅

### Check Config

```python
cfg = load_config('config.yaml')

# Config is automatically validated on load
# Checks:
# - CUDA availability
# - Valid parameter ranges
# - Consistent dimensions (slot_dim matches everywhere)
# - Valid strategy names
```

### Print Current Settings

```python
cfg.print_summary()  # Human-readable summary
```

### Save Modified Config

```python
from src.utils.config import save_config

cfg = load_config('config.yaml')
cfg.agents.num_prototypes = 512

save_config(cfg, 'config_modified.yaml')
```

---

## FAQ ❓

### Q: AdaSlot vs Fixed Slots?

**A:** Use **AdaSlot** (adaptive=true) for:
- Real-world images (varying object counts)
- Better flexibility
- Slightly slower but worth it

Use **Fixed Slots** for:
- Ablation studies
- Faster training/inference
- Controlled experiments

---

### Q: How many prototypes should I use?

**A:**
- CIFAR-100 (32×32): **256** ✅
- Tiny-ImageNet (64×64): **256-512** ✅
- ImageNet (224×224): **512-1024** ✅

More is not always better! 256 is a sweet spot.

---

### Q: Top-k vs Bandit selection?

**A:**
1. **Start with `topk_estimator`** (baseline)
2. If performance plateaus, try **`bandit_ucb`** (exploration)
3. For Bayesian approach, try **`bandit_thompson`**

Bandit is NOT magic! Only helps if:
- Initial estimates are noisy
- Optimal agents change over time
- You want to explore

---

### Q: Single tree vs Ensemble?

**A:**
- **Single tree** (`hoeffding_adaptive`): Fast, good enough ✅
- **Ensemble** (`ensemble`): Better accuracy, 10x slower

Start with single tree.

---

### Q: Reward function for bandits?

**A:** **TODO:** Consult with professor! 📝

Options:
- Accuracy (1/0)
- Negative loss
- Prediction confidence
- Diversity bonus

This is research territory!

---

## Summary Checklist ✅

Before running experiments:

- [ ] Choose adaptive vs fixed slots
- [ ] Set `num_prototypes` (output dim)
- [ ] Choose selection strategy (start with `topk_estimator`)
- [ ] Set `k` (number of agents per slot)
- [ ] Choose aggregator type (`hoeffding_adaptive` recommended)
- [ ] Check data settings (dataset, num_experiences)
- [ ] Verify device settings (CUDA available?)
- [ ] Set logging/checkpointing paths
- [ ] Print config summary (`cfg.print_summary()`)

---

## Next Steps 🚀

1. **Start with baseline:**
   ```python
   cfg = load_config('config_variants/01_baseline_adaptive_slots.yaml')
   ```

2. **Train Phase 1 (agents):**
   ```python
   python train_phase1.py --config config_variants/01_baseline_adaptive_slots.yaml
   ```

3. **Train Phase 2 (tree):**
   ```python
   python train_phase2.py --config config_variants/01_baseline_adaptive_slots.yaml
   ```

4. **Try UCB bandit:**
   ```python
   cfg = load_config('config_variants/02_ucb_bandit_exploration.yaml')
   ```

5. **Ablation studies:**
   - Fixed vs adaptive slots
   - Different prototype counts
   - Bandit vs no bandit
   - Single tree vs ensemble

---

**Questions?** Check `ARCHITECTURE_FINAL.md` for system details!

**Last Updated:** 2026-02-13

