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

### Supervised + Self-Supervised (DINOv2-style)

```python
# Supervised
hidden_labels = model(images, return_hidden=True)
tree.fit(hidden_labels, targets)

# Self-supervised
aug1, aug2 = augment(images)
h1 = model(aug1, return_hidden=True)
h2 = model(aug2, return_hidden=True)
loss = contrastive_loss(h1, h2)
```

## 📊 Key Features

- ✅ **Slot Attention**: Object-centric decomposition
- ✅ **VAE/MLP Estimators**: Performance estimation
- ✅ **Top-k Selection**: Efficient agent selection (k=3)
- ✅ **50 Agents**: ResidualMLP (same architecture, different weights)
- ✅ **Hoeffding Tree**: True incremental learning
- ✅ **Hidden Labels**: DINO-style embeddings
- ✅ **Complete System**: End-to-end pipeline

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements_slot_agent.txt
pip install river  # For Hoeffding Tree

# Run system
python -c "
from src.slot_multi_agent import SlotMultiAgentSystem
system = SlotMultiAgentSystem(num_agents=50, k=3, device='cuda')
print('System ready!')
"
```

See **QUICKSTART.md** for complete examples!

## 🔧 Next Steps

1. ✅ ~~Implement all components~~ **DONE!**
2. 🎯 Train on CIFAR-100 continual learning
3. 🎯 Experiment with different k values
4. 🎯 Compare VAE vs MLP estimators
5. 🎯 Research better estimation methods (see RESEARCH_PROMPT...md)

## 📝 Notes

- **Agents có cùng architecture**, khác nhau ở weights (specialized)
- **Sub-networks**: VAE hoặc MLP đơn giản
- **Top-k selection**: Không phức tạp, chỉ sort scores
- **Decision tree**: Hỗ trợ học thêm classes mới
- **Aggregation**: Bạn sẽ bàn chi tiết cách kết hợp outputs

---

**Status**: ✅ **COMPLETE** - All 5 components implemented (~1,600 lines)

See **COMPLETE_IMPLEMENTATION.md** for full details!
