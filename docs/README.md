# Slot-Based Multi-Agent Continual Learning

## 🎯 Core Idea

```
Image
  ↓
Slot Attention → Decompose thành N slots (objects)
  ↓
For EACH slot:
  ├─ Sub-network₁ (VAE/MLP) → score₁
  ├─ Sub-network₂ (VAE/MLP) → score₂
  ├─ ...
  └─ Sub-networkₙ (VAE/MLP) → scoreₙ
  ↓
  Select top-k agents (highest scores)
  ↓
  Run agents → hidden representations
  ↓
Aggregate all slots → Decision Tree
  ↓
Final prediction
```

## 📁 Project Structure

```
Continual-Learning/
├── src/
│   ├── base/
│   │   ├── types.py              # Common types
│   │   └── base_agent.py         # Base agent class
│   │
│   ├── models/
│   │   ├── slot_attention/       # ✅ Slot decomposition
│   │   │   ├── slot_attention.py
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   └── model.py
│   │   │
│   │   └── vae/                  # ✅ VAE estimator
│   │       ├── vae.py
│   │       └── uncertainty.py
│   │
│   ├── data/                     # Data loaders
│   │   ├── continual_cifar100_avalanche.py
│   │   └── continual_tinyimagenet.py
│   │
│   └── slot_multi_agent/         # ✅ COMPLETE
│       ├── estimators.py         # ✅ VAE, MLP estimators
│       ├── selector.py           # ✅ Top-k selection
│       ├── atomic_agent.py       # ✅ 50 agents (ResidualMLP)
│       ├── aggregator.py         # ✅ Hoeffding Tree
│       └── system.py             # ✅ End-to-end pipeline
│
├── tests/
│   └── data/                     # Basic tests
│
├── README.md                     # This file
├── environment.yml               # Conda environment
└── requirements.txt              # Dependencies
```

## 🚀 Architecture Components

### 1. Slot Attention (✅ Done)
Decomposes image into object-centric slots.

```python
from src.models.slot_attention import SlotAttention

slots, attn = slot_attention(features)  # → (B, num_slots, slot_dim)
```

### 2. Sub-network Estimators (✅ Done)
Lightweight networks ước lượng performance của agent trên slot:
- **VAE-based**: Reconstruction error
- **MLP-based**: Direct score prediction

```python
from src.slot_multi_agent import VAEEstimator, MLPEstimator

estimator = VAEEstimator(agent_id=0, slot_dim=64)
score = estimator.estimate_performance(slot)  # [0, 1]
```

### 3. Top-k Selector (✅ Done)
Chọn k agents tốt nhất cho mỗi slot.

```python
from src.slot_multi_agent import TopKAgentSelector

selector = TopKAgentSelector(estimators, k=3)
selected_ids, scores = selector.select_top_k(slot)
```

### 4. Atomic Agents (✅ Done)
Generate hidden representations (not final predictions).

```python
from src.slot_multi_agent import ResidualMLPAgent

agent = ResidualMLPAgent(agent_id=0, slot_dim=64, output_dim=128)
hidden_label = agent(slot)  # 128-dim embedding
```

### 5. Decision Tree Aggregator (✅ Done)
Combines hidden labels, supports continual learning.

```python
from src.slot_multi_agent import HoeffdingTreeAggregator

tree = HoeffdingTreeAggregator()  # Incremental, no retraining
tree.partial_fit(hidden_labels, targets)  # Add new classes
predictions = tree.predict(hidden_labels)
```

## 🎓 Training Strategy

### Phase 1: DINO-style Self-Supervised Learning (Agents)

Agents learn hidden representations via **simplified DINO** (no augmentation, only centering + sharpening):

```python
from src.slot_multi_agent import DINOLoss

# Student and Teacher agents (Teacher updated via EMA)
student_logits = student_agent(slot)  # (B, num_prototypes)
teacher_logits = teacher_agent(slot)  # (B, num_prototypes)

# DINO Loss: Centering + Temperature Sharpening
dino_loss = DINOLoss(
    num_prototypes=256,
    student_temp=0.1,    # Higher temp (softer)
    teacher_temp=0.07,   # Lower temp (sharper, more confident)
    center_momentum=0.9  # EMA for center (prevents collapse)
)
loss = dino_loss(student_logits, teacher_logits)
```

**Key mechanisms:**
- ✅ **Centering**: Subtract running mean from teacher outputs (prevent mode collapse)
- ✅ **Sharpening**: Lower temperature for teacher (more confident predictions)
- ✅ **EMA Teacher**: Teacher updated via exponential moving average (no backprop)
- ❌ **NO multi-crop augmentation** (simplified version)

### Phase 2: Supervised Tree Training

```python
# Get hidden labels from trained agents
hidden_labels = []
for slot in slots:
    selected_agents = selector.select_top_k(slot, k=3)
    labels = [agent(slot) for agent in selected_agents]
    hidden_labels.append(torch.cat(labels))

# Train Hoeffding Tree incrementally
tree.partial_fit(hidden_labels, targets)  # Supports new classes
```

## 📊 Key Features

- ✅ **Slot Attention**: Object-centric decomposition (adaptive slots 3-10)
- ✅ **VAE/MLP Estimators**: Lightweight performance estimation
- ✅ **Top-k Selection**: Efficient agent selection per slot (k=3)
- ✅ **Bandit Strategies**: UCB, Thompson Sampling, Epsilon-Greedy
- ✅ **50 Agents**: ResidualMLP (same architecture, specialized weights)
- ✅ **Hoeffding Tree**: True incremental learning (no retraining)
- ✅ **Hidden Labels**: Softmax probabilities over 256 prototypes
- ✅ **DINO Training**: Centering + sharpening (simplified, no augmentation)
- ✅ **Complete System**: End-to-end pipeline with checkpointing

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load configuration
python -c "
from src.utils import load_config
cfg = load_config('config.yaml')
print(f'Agents: {cfg.agents.num_agents}')
print(f'Prototypes: {cfg.agents.num_prototypes}')
print(f'Slots: {cfg.slot_attention.min_slots}-{cfg.slot_attention.max_slots}')
"

# 3. Quick test
python -c "
from src.slot_multi_agent import create_agent_pool
students, teachers = create_agent_pool(50, 64, 256)
print(f'✓ Created {len(students)} agents')
"
```

See **QUICKSTART.md** for training examples and **CONFIG_GUIDE.md** for all options!

## ⚙️ Configuration

All system parameters are managed via **`config.yaml`**:

```yaml
# Example: config.yaml
slot_attention:
  adaptive: true        # AdaSlot (adaptive slot count)
  min_slots: 3
  max_slots: 10
  slot_dim: 64

agents:
  num_agents: 50
  num_prototypes: 256   # Hidden label dimension
  dino:
    student_temp: 0.1
    teacher_temp: 0.07
    center_momentum: 0.9

selection:
  strategy: "top_k"     # "top_k" | "ucb" | "thompson" | "epsilon_greedy"
  k: 3

aggregator:
  type: "hoeffding_tree"  # Incremental learning
```

**8 Pre-configured Variants:**
- `01_baseline_adaptive_slots.yaml` - Standard setup
- `02_ucb_bandit_exploration.yaml` - UCB agent selection
- `03_large_prototypes.yaml` - 512 prototypes (more expressive)
- `04_ensemble_tree.yaml` - 5 trees for robustness
- `05_thompson_sampling.yaml` - Bayesian agent selection
- `06_fixed_slots.yaml` - Fixed 7 slots (no adaptation)
- `07_pretrained_slot_attention.yaml` - Load AdaSlot checkpoint
- `08_pretrained_agents_phase2.yaml` - Skip Phase 1

See **CONFIG_GUIDE.md** for all options!

## 🔧 Next Steps

1. ✅ ~~Implement all components~~ **DONE!**
2. 🎯 Train Phase 1 (agents with DINO)
3. 🎯 Train Phase 2 (incremental tree)
4. 🎯 Compare selection strategies (top-k vs bandit)
5. 🎯 Experiment with prototype dimensions
6. 🎯 Research better estimation methods (see RESEARCH_PROMPT...md)

## 📚 Documentation

- **ARCHITECTURE_FINAL.md** (651 lines) - Complete system architecture
- **CONFIG_GUIDE.md** (700+ lines) - All configuration options
- **CHECKPOINT_GUIDE.md** (500+ lines) - Loading/saving models
- **DINOV2_TRAINING_DETAILS.md** (376 lines) - DINO mechanism deep dive
- **RESEARCH_PROMPT_PERFORMANCE_ESTIMATION.md** (304 lines) - Research directions
- **QUICKSTART.md** (190 lines) - Step-by-step examples

## 📝 Implementation Notes

- **Agents**: Same architecture (ResidualMLP), specialized via DINO training
- **Estimators**: VAE (reconstruction-based) or MLP (direct prediction)
- **Selection**: Top-k (deterministic) or bandit (exploration)
- **Hidden Labels**: Softmax probabilities (continuous, not discrete)
- **Tree**: Hoeffding Tree (true incremental, no retraining)
- **DINO**: Simplified version (centering + sharpening, no augmentation)
- **Checkpoints**: Full support for resuming training

## 🌟 Key Design Decisions

1. **Why simplified DINO?** 
   - Full DINOv2 requires multi-crop augmentation (expensive)
   - Centering + sharpening capture core mechanism
   - Faster training, still prevents collapse

2. **Why softmax probabilities?**
   - Decision trees handle continuous features well
   - More expressive than discrete IDs
   - Gradual confidence levels

3. **Why Hoeffding Tree?**
   - True online learning (no retraining)
   - Supports new classes dynamically
   - Handles high-dimensional continuous features

---

**Status**: ✅ **COMPLETE** - All components implemented (~10,500+ lines)

**Last Updated**: 2026-02-14
