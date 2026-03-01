# Slot-Based Multi-Agent Continual Learning

A continual learning system combining **Slot Attention** (object-centric decomposition), **DINO-style self-supervised learning** (agent training), **bandit algorithms** (agent selection), and **Hoeffding Trees** (incremental classification) for class-incremental learning on CIFAR-100.

---

## Architecture Overview

```
Input Image (B, C, H, W)
      ↓
 AdaSlotModel / SlotAttentionAutoEncoder
      ↓
 Slots: (B, num_slots, slot_dim)
      ↓  [For each slot]
 Performance Estimators (VAE / MLP / Hybrid)
      ↓
 Agent Selection (TopK / UCB / Thompson / ε-Greedy)
      ↓
 ResidualMLPAgent × k  →  Hidden Labels (softmax over prototypes)
      ↓  [Concatenate across all slots × k agents]
 IncrementalTreeAggregator (Hoeffding Tree)
      ↓
 Final Class Prediction
```

---

## Project Structure

```
Continual-Learning/
├── src/
│   ├── __init__.py                      # Package root (v0.1.0)
│   ├── train.py                         # End-to-end training pipeline (Phase 1→2→3)
│   ├── base/
│   │   ├── types.py
│   │   └── base_agent.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── continual_cifar100.py        # Class-incremental CIFAR-100 pipeline
│   │   ├── continual_cifar100_avalanche.py
│   │   └── continual_tinyimagenet.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── slot_attention/              # Classic Slot Attention (encoder/decoder)
│   │   │   ├── slot_attention.py
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   └── model.py
│   │   ├── adaslot/                     # AdaSlot (adaptive slot count via Gumbel)
│   │   │   ├── model.py                 # AdaSlotModel
│   │   │   ├── perceptual_grouping.py   # SlotAttentionGroupingGumbelV1
│   │   │   ├── feature_extractor.py
│   │   │   ├── conditioning.py          # RandomConditioning
│   │   │   ├── decoder.py
│   │   │   ├── mlp.py
│   │   │   └── positional_embedding.py
│   │   └── vae/                         # Standalone VAE
│   │       ├── vae.py
│   │       └── uncertainty.py
│   ├── slot_multi_agent/                # Core multi-agent system
│   │   ├── __init__.py
│   │   ├── atomic_agent.py              # ResidualMLPAgent, DINOLoss
│   │   ├── estimators.py               # VAEEstimator, MLPEstimator, HybridEstimator
│   │   ├── selector.py                  # TopKAgentSelector, AdaptiveKSelector
│   │   ├── bandit_selector.py          # UCB, Thompson, ε-Greedy, WeightedTopK
│   │   ├── aggregator.py               # IncrementalTreeAggregator, EnsembleTreeAggregator
│   │   └── system.py                   # SlotMultiAgentSystem (end-to-end)
│   └── utils/
├── tests/
├── checkpoints/
├── data/
├── requirements.txt
└── README.md
```

---

## Component Details

### 1. Slot Decomposition

Two implementations available:

#### Classic Slot Attention (`src/models/slot_attention/`)
Standard encoder-decoder with fixed slot count.

#### AdaSlot (`src/models/adaslot/`)
Adaptive slot count using Gumbel-Softmax hard decisions. Can load from pretrained checkpoint.

```python
from src.models.adaslot.model import AdaSlotModel

model = AdaSlotModel(
    resolution=(128, 128),
    num_slots=11,   # Max slots; actual active count decided by Gumbel
    slot_dim=64,
    num_iterations=3,
    feature_dim=64,
    kvq_dim=128,
    low_bound=1     # Minimum active slots
)

# Forward: returns dict with 'slots', 'masks', 'reconstruction', etc.
out = model(image, global_step=step)

# Encode only
slots = model.encode(image)  # (B, num_slots, slot_dim)
```

---

### 2. Atomic Agents (`atomic_agent.py`)

Each agent is a `ResidualMLPAgent` trained with DINO-style SSL.

```python
from src.slot_multi_agent import ResidualMLPAgent, DINOLoss, create_agent_pool

# Create a pool of 50 agent pairs (student + teacher)
student_agents, teacher_agents = create_agent_pool(
    num_agents=50,
    slot_dim=64,
    num_prototypes=256,   # Hidden label dimension
    hidden_dim=256,
    num_blocks=3,
    dropout=0.1,
    device='cuda'
)

# Forward: slot → softmax probabilities (hidden label)
hidden_label = student_agents[i](slot)              # (B, 256) softmax probs
logits       = student_agents[i](slot, return_logits=True, temperature=0.1)

# DINO Loss (centering + temperature sharpening)
criterion = DINOLoss(
    num_prototypes=256,
    student_temp=0.1,
    teacher_temp=0.07,
    center_momentum=0.9   # EMA for center (prevents collapse)
)
loss = criterion(student_logits, teacher_logits)
```

**ResidualMLPAgent architecture:**
```
slot (64) → Linear + LayerNorm → [ResidualBlock × 3] → Linear → prototypes (256) → softmax
```
Each `ResidualBlock`: `x → Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm → Dropout → (+x)`

**EMA teacher update:**
```python
from src.slot_multi_agent import update_teacher, update_all_teachers

# Single pair
update_teacher(student, teacher, momentum=0.996)

# Entire pool
update_all_teachers(student_agents, teacher_agents, momentum=0.996)
```

---

### 3. Performance Estimators (`estimators.py`)

Lightweight networks that predict how well a given agent will perform on a slot *without running the agent*.

```python
from src.slot_multi_agent import VAEEstimator, MLPEstimator, HybridEstimator, create_estimator_pool

# VAE-based (one per agent)
vae = VAEEstimator(agent_id=0, slot_dim=64, latent_dim=16, hidden_dim=64)
score = vae.estimate_performance(slot)   # [0, 1] – lower recon error → higher score

# MLP-based (shared across agents, uses agent embedding)
mlp = MLPEstimator(num_agents=50, slot_dim=64, hidden_dim=128, agent_embed_dim=32)
score = mlp.estimate_performance(slot, agent_id=5)

# Hybrid: weighted combination of VAE + MLP
hybrid = HybridEstimator(agent_id=0, num_agents=50, slot_dim=64, vae_weight=0.5)

# Create a pool of 50 estimators
estimators = create_estimator_pool(num_agents=50, estimator_type='vae', slot_dim=64)
```

---

### 4. Agent Selection

#### Simple Top-K (`selector.py`)

```python
from src.slot_multi_agent import TopKAgentSelector, AdaptiveKSelector, create_selector

selector = TopKAgentSelector(estimators, k=3, temperature=1.0)

# Deterministic top-k
selected_ids, scores = selector.select_top_k(slot)

# Probabilistic (for exploration during training)
selected_ids = selector.select_probabilistic(slot)

# Batch selection
ids_batch, scores_batch = selector.select_batch(slots)  # slots: (B, num_slots, slot_dim)

# Adaptive k (adjusts based on score uncertainty)
adaptive = AdaptiveKSelector(estimators, k_min=2, k_max=5)
selected_ids, scores, k_used = adaptive.select_adaptive(slot)
```

#### Bandit-Based Selection (`bandit_selector.py`)

Exploration-exploitation trade-off using multi-armed bandit algorithms.

| Strategy | Class | Key Parameters |
|---|---|---|
| UCB | `UCBSelector` | `exploration_constant=2.0` |
| Thompson Sampling | `ThompsonSamplingSelector` | `alpha_init=1.0`, `beta_init=1.0` |
| Epsilon-Greedy | `EpsilonGreedySelector` | `epsilon=0.1`, `epsilon_decay=0.995`, `min_epsilon=0.01` |
| Weighted Top-K | `WeightedTopKSelector` | — (pure exploitation baseline) |

```python
from src.slot_multi_agent import create_bandit_selector

bandit = create_bandit_selector(strategy='ucb', num_agents=50, exploration_constant=2.0)

# Select k agents + compute softmax weights
selected_indices, weights = bandit.select_and_weight(
    slot=slot,                  # (B, slot_dim)
    estimated_scores=scores,    # (B, num_agents)
    k=3
)
# selected_indices: (B, 3),  weights: (B, 3) sums to 1

# Update bandit statistics after observing reward
bandit.update(agent_idx=2, slot=slot, reward=0.85)
```

**UCB formula:**
```
UCB(i) = μ_i + c × sqrt(log(t) / n_i)
```

**Thompson Sampling:** maintains Beta(α, β) distribution per agent, samples from posterior.

**Epsilon-Greedy:** ε decays from `epsilon` to `min_epsilon` by factor `epsilon_decay` each step.

---

### 5. Aggregator – Incremental Decision Tree (`aggregator.py`)

Learns online from concatenated hidden labels → class, using Hoeffding Trees from the [`river`](https://riverml.xyz/) library.

```python
from src.slot_multi_agent import IncrementalTreeAggregator, EnsembleTreeAggregator, BatchTreeAggregator, create_aggregator

# Single Hoeffding Tree
tree = IncrementalTreeAggregator(
    grace_period=200,
    split_confidence=1e-5,
    leaf_prediction='nba',   # Naive Bayes Adaptive
    adaptive=True            # HoeffdingAdaptiveTreeClassifier (handles concept drift)
)
tree.learn_one(hidden_labels, label)          # hidden_labels: numpy array
pred   = tree.predict_one(hidden_labels)
proba  = tree.predict_proba_one(hidden_labels)  # Dict[class → prob]

# Ensemble (Adaptive Random Forest from river)
ensemble = EnsembleTreeAggregator(n_models=10, max_features='sqrt')

# Batch wrapper (loops internally one-by-one)
batch_tree = BatchTreeAggregator(tree)
batch_tree.learn_batch(hidden_labels_batch, labels_batch)   # Tensor inputs
preds = batch_tree.predict_batch(hidden_labels_batch)

# Factory
aggregator = create_aggregator('hoeffding_adaptive')  # or 'hoeffding', 'ensemble'
```

**Input feature size**: `num_slots × k × num_prototypes` (e.g. 7 × 3 × 256 = **5376** features)

---

### 6. End-to-End System (`system.py`)

```python
from src.slot_multi_agent.system import SlotMultiAgentSystem

system = SlotMultiAgentSystem(
    num_agents=50,
    num_slots=7,
    slot_dim=64,
    num_prototypes=256,
    k=3,
    estimator_type='vae',          # 'vae' or 'mlp'
    aggregator_type='hoeffding',   # 'hoeffding', 'incremental', 'soft'
    aggregate_mode='concat',       # 'concat' or 'mean'
    device='cuda'
)

# Full forward pass
predictions, metadata = system.forward(images, return_metadata=True)

# Incremental training step
info = system.train_step(images, targets)

# Evaluate
metrics = system.evaluate(images, targets)

# Checkpointing
system.save_checkpoint('checkpoints/system.pt')
system.load_checkpoint('checkpoints/system.pt')
```

---

## Data Pipeline (`src/data/`)

### CIFAR-100 Class-Incremental

```python
from src.data import get_continual_cifar100_loaders

train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
    n_tasks=5,          # Supported: 2, 5, 10, 20, 50, 100
    batch_size=128,
    num_workers=4,
    root='./data',
    seed=42
)

# train_loaders[i] → classes for task i only (20 classes/task for n_tasks=5)
# test_loaders[i]  → all classes seen up to task i
```

- Configurable splits: 2 / 5 / 10 / 20 / 50 / 100 tasks
- Standard CIFAR-100 augmentation (RandomCrop, HorizontalFlip, Normalize)
- Cumulative test evaluation (all seen classes)

Also available:
- `continual_cifar100_avalanche.py` — Avalanche-lib integration
- `continual_tinyimagenet.py` — Tiny-ImageNet support

---

## Training Pipeline

### Phase 1 – DINO SSL (Unsupervised Agent Training)

```python
# Initialize
student_agents, teacher_agents = create_agent_pool(50, 64, 256, device='cuda')
estimators = create_estimator_pool(50, 'vae', 64)
dino_losses = [DINOLoss(256) for _ in range(50)]

for epoch in range(epochs):
    for images in unlabeled_loader:
        slots = slot_model.encode(images)    # (B, 7, 64)
        total_loss = 0

        for slot_idx in range(7):
            slot = slots[:, slot_idx, :]
            scores = torch.stack([est.estimate_performance(slot) for est in estimators], dim=1)
            top_k = torch.topk(scores, k=3).indices

            for agent_idx in top_k[0]:
                student_logits = student_agents[agent_idx](slot, return_logits=True)
                with torch.no_grad():
                    teacher_logits = teacher_agents[agent_idx](slot, return_logits=True)
                total_loss += dino_losses[agent_idx](student_logits, teacher_logits)

        total_loss.backward()
        optimizer.step()
        update_all_teachers(student_agents, teacher_agents, momentum=0.996)
```

**DINO Loss mechanics:**
- Teacher: `softmax((logits - center) / teacher_temp)`, center updated via EMA (momentum 0.9)
- Student: `log_softmax(logits / student_temp)`
- Loss: cross-entropy (teacher || student)
- *Simplified*: no multi-crop augmentation (vs. full DINOv2)

### Phase 2 – Incremental Tree Training (Supervised)

```python
# Freeze agents
for agent in student_agents: agent.eval()

bandit = create_bandit_selector('ucb', num_agents=50)
tree = create_aggregator('hoeffding_adaptive')

for images, labels in labeled_loader:
    with torch.no_grad():
        slots = slot_model.encode(images)

    for i in range(images.size(0)):
        hidden_labels = []
        for slot_idx in range(7):
            slot = slots[i:i+1, slot_idx, :]
            scores = torch.stack([est.estimate_performance(slot) for est in estimators], dim=1)
            top_k_indices, weights = bandit.select_and_weight(slot, scores, k=3)

            for k_idx in range(3):
                agent_idx = top_k_indices[0, k_idx].item()
                prob_dist = student_agents[agent_idx](slot)  # (1, 256)
                hidden_labels.append(prob_dist.cpu().numpy().flatten())

        features = np.concatenate(hidden_labels)   # (5376,)
        tree.learn_one(features, labels[i].item())
```

### Phase 3 – Continual Learning (New Tasks)

```python
# Agents stay frozen; only tree receives new examples
for images, labels in new_task_loader:
    for i in range(images.size(0)):
        features = extract_hidden_labels(images[i:i+1])
        tree.learn_one(features, labels[i].item())
# Tree grows new branches; old knowledge preserved
```

---

## Key Hyperparameters

| Component | Parameter | Default | Notes |
|---|---|---|---|
| **AdaSlot** | `num_slots` | 11 | Max slots (adaptive) |
| | `slot_dim` | 64 | Slot embedding dim |
| | `low_bound` | 1 | Min active slots |
| **Agents** | `num_agents` | 50 | Pool size |
| | `num_prototypes` | 256 | Hidden label dim |
| | `num_blocks` | 3 | Residual blocks |
| | `hidden_dim` | 256 | MLP hidden dim |
| **DINO** | `teacher_temp` | 0.07 | Sharp teacher |
| | `student_temp` | 0.1 | Softer student |
| | `center_momentum` | 0.9 | EMA for centering |
| | `ema_momentum` | 0.996 | Teacher weight update |
| **Selection** | `k` | 3 | Agents per slot |
| **UCB** | `exploration_constant` | 2.0 | Exploration bonus |
| **ε-Greedy** | `epsilon` | 0.1 | Initial exploration |
| | `epsilon_decay` | 0.995 | Decay per step |
| **Tree** | `grace_period` | 200 | Examples before split |
| | `split_confidence` | 1e-5 | Hoeffding bound δ |
| | `leaf_prediction` | `'nba'` | Naive Bayes Adaptive |

---

## Quick Start

```bash
# Run all 3 phases sequentially
python -m src.train --phase all --data_dir data/clevr

# Phase 1 only: train AdaSlot slot decomposition
python -m src.train --phase 1 --data_dir data/clevr --p1_steps 500000

# Phase 2 only: train agents with DINO SSL (requires AdaSlot checkpoint)
python -m src.train --phase 2 --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth

# Phase 3 only: fit Hoeffding Tree incrementally
python -m src.train --phase 3 \
    --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth \
    --agent_ckpt checkpoints/agents/agents_final.pth \
    --num_classes 100
```

---

## Installation

```bash
# PyTorch 2.1 with CUDA 12.1
pip install -r requirements.txt

# Requirements include:
# torch==2.1.0+cu121, torchvision==0.16.0+cu121
# avalanche-lib==0.4.0
# river  (Hoeffding Trees)
# numpy, pandas, scipy
# matplotlib, seaborn
# tensorboard, wandb
# pytest
```

---

## Implementation Status

| Component | Status | File |
|---|---|---|
| ResidualMLPAgent | ✅ Done | `atomic_agent.py` |
| DINOLoss | ✅ Done | `atomic_agent.py` |
| VAEEstimator | ✅ Done | `estimators.py` |
| MLPEstimator | ✅ Done | `estimators.py` |
| HybridEstimator | ✅ Done | `estimators.py` |
| TopKAgentSelector | ✅ Done | `selector.py` |
| AdaptiveKSelector | ✅ Done | `selector.py` |
| UCBSelector | ✅ Done | `bandit_selector.py` |
| ThompsonSamplingSelector | ✅ Done | `bandit_selector.py` |
| EpsilonGreedySelector | ✅ Done | `bandit_selector.py` |
| IncrementalTreeAggregator | ✅ Done | `aggregator.py` |
| EnsembleTreeAggregator | ✅ Done | `aggregator.py` |
| SlotMultiAgentSystem | ✅ Done | `system.py` |
| AdaSlotModel | ✅ Done | `models/adaslot/model.py` |
| ContinualCIFAR100 pipeline | ✅ Done | `data/continual_cifar100.py` |
| Bandit reward definition | ⏳ TODO | — |
| Experiments on CIFAR-100 | ⏳ TODO | — |
| Ablation studies | ⏳ TODO | — |

---

## References

1. **Slot Attention** — Locatello et al. (2020) *Object-Centric Learning with Slot Attention*
2. **DINOv2** — Oquab et al. (2023) *DINOv2: Learning Robust Visual Features without Supervision* · [GitHub](https://github.com/facebookresearch/dinov2)
4. **Hoeffding Tree** — Domingos & Hulten (2000) *Mining High-Speed Data Streams* · [River](https://riverml.xyz/)
5. **UCB Bandits** — Auer et al. (2002) *Finite-time Analysis of the Multiarmed Bandit Problem*
6. **Thompson Sampling** — Thompson (1933)
7. **Avalanche** — [avalanche.continualai.org](https://avalanche.continualai.org/)

---

**Version**: 0.1.0 | **Last Updated**: 2026-03-01
