# Implementation Status

## ✅ Completed

### 1. **Estimators** (`src/slot_multi_agent/estimators.py`) ✅
- ✅ `VAEEstimator`: Lightweight VAE, reconstruction error → performance score
- ✅ `MLPEstimator`: Direct mapping (slot + agent_id) → score
- ✅ `HybridEstimator`: Combines both VAE and MLP

### 2. **Selector** (`src/slot_multi_agent/selector.py`) ✅
- ✅ `TopKAgentSelector`: Simple top-k selection by scores
- ✅ `AdaptiveKSelector`: Adjusts k based on uncertainty
- ✅ Batch processing support
- ✅ Probabilistic sampling (for exploration)

### 3. **Atomic Agents** (`src/slot_multi_agent/atomic_agent.py`) ✅
- ✅ `ResidualMLPAgent`: Main agent with residual blocks
- ✅ `LightweightMLPAgent`: Lighter variant
- ✅ `AgentEnsemble`: Ensemble with learned weighting
- ✅ `create_agent_pool()`: Factory for 50 agents

### 4. **Aggregator** (`src/slot_multi_agent/aggregator.py`) ✅
- ✅ `HoeffdingTreeAggregator`: True incremental (river library)
- ✅ `IncrementalDecisionTree`: sklearn-based with storage
- ✅ `SoftDecisionTree`: Differentiable tree
- ✅ `create_aggregator()`: Factory function

### 5. **Complete System** (`src/slot_multi_agent/system.py`) ✅
- ✅ `SlotMultiAgentSystem`: End-to-end pipeline
- ✅ Forward pass with all components
- ✅ Training step (incremental)
- ✅ Evaluation metrics
- ✅ Checkpoint save/load

## 🔨 Next Steps

### 3. **Atomic Agent** (`src/slot_multi_agent/atomic_agent.py`)
```python
class ResidualMLPAgent(nn.Module):
    """
    Process slot token → hidden representation (DINO-style).
    
    Input: Slot (64-dim vector)
    Output: Hidden label (128-dim embedding for Decision Tree)
    
    Architecture:
      Input Projection: 64 → 256
      3× Residual Blocks (256-dim)
      Projection Head: 256 → 128
    """
```

### 4. **Aggregator** (`src/slot_multi_agent/aggregator.py`)

**Need to implement Incremental Decision Trees:**

#### Option A: Hoeffding Tree (VFDT)
```python
class HoeffdingTreeAggregator:
    """
    Very Fast Decision Tree (Hoeffding Tree).
    
    Perfect for incremental learning:
    - Learns online, no retraining
    - Adds splits when confident (Hoeffding bound)
    - Naturally supports new classes
    
    Library: river-ml or custom implementation
    """
```

**Advantages**:
- ✅ True incremental learning
- ✅ No need to store all data
- ✅ Adds hyperplanes when有信心
- ✅ Supports new classes naturally

**Disadvantages**:
- ❌ Might need many samples before splitting
- ❌ Less accurate than batch trees initially

#### Option B: iForest + Soft Tree
```python
class IncrementalDecisionTree:
    """
    Hybrid approach:
    1. Use sklearn tree for current classes
    2. When new classes arrive, expand tree structure
    3. Store representative samples for quick retrain
    """
```

**Advantages**:
- ✅ Better accuracy (uses sklearn backend)
- ✅ Flexible expansion strategy

**Disadvantages**:
- ❌ Need to retrain on stored data
- ❌ Memory overhead

### 5. **Complete System** (`src/slot_multi_agent/system.py`)
```python
class SlotMultiAgentSystem(nn.Module):
    """
    End-to-end pipeline:
    
    Image
      ↓
    Slot Attention → 7 slots
      ↓
    For each slot:
      Estimators → scores (50 agents)
      TopK Selector → 3 best agents
      Agents → 3 hidden labels (128-dim each)
      Concat → 384-dim
      ↓
    Aggregate all slots → 7 × 384 = 2688-dim
      ↓
    Incremental Tree → Final prediction
    """
```

## 📋 Configuration

Based on your answers:

1. **Decision Tree**: Hoeffding Tree (incremental, adds hyperplanes)
2. **Aggregation**: Concatenation of agent outputs
3. **Task ID**: Not available (class-incremental, no task boundaries)
4. **Num Agents**: 50
5. **Continual Learning**: Class-incremental (CIFAR-100, 5 experiences, 20 classes each)
6. **Top-k**: k=3 (3 agents per slot)

## 🎯 Next Steps

1. Implement `atomic_agent.py` (ResidualMLPAgent)
2. Research and implement `aggregator.py` (Hoeffding Tree or hybrid)
3. Implement `system.py` (complete pipeline)
4. Create training scripts
5. Test on CIFAR-100

## 💡 Key Design Decisions

### Aggregation Strategy

**Per Slot**:
```
Slot → Top-3 agents → [hidden1, hidden2, hidden3]
→ Concat → 384-dim vector
```

**All Slots**:
```
7 slots × 384-dim = 2688-dim
→ Decision Tree input
```

**Alternative** (if 2688-dim too large):
```
7 slots × 384-dim
→ Mean pooling → 384-dim
→ Decision Tree
```

### Incremental Tree Recommendation

**Use Hoeffding Tree** because:
1. True incremental (không cần retrain)
2. Adds splits dynamically
3. Supports new classes naturally
4. Matches "học thêm hyperplane" requirement

**Library**: `river` (formerly `creme`)
```bash
pip install river
```

**Example**:
```python
from river import tree

model = tree.HoeffdingTreeClassifier()

# Incremental learning
for x, y in stream:
    model.predict_one(x)  # Predict
    model.learn_one(x, y)  # Learn
```

Perfect for class-incremental continual learning! 🎯

## 📊 Architecture Summary

```
CIFAR-100 Image (32×32×3)
  ↓
CNN Encoder → Features
  ↓
Slot Attention (initialized slots)
  ↓
7 Slots (each 64-dim)
  ↓
For EACH of 7 slots:
  ├─ 50 VAE/MLP Estimators → 50 scores
  ├─ TopK Selector → Select 3 best agents
  ├─ 3 ResidualMLPAgents → 3 × 128-dim hidden labels
  └─ Concat → 384-dim
  ↓
Aggregate 7 slots:
  Concat → 7 × 384 = 2688-dim
  (or mean pool → 384-dim)
  ↓
Hoeffding Tree Classifier
  ├─ Online learning
  ├─ Adds splits when confident
  └─ Supports new classes (0-99)
  ↓
Final Prediction
```

## 🔬 Training Pipeline

### Phase 1: Self-Supervised (Optional)
Train agents to generate consistent features (DINO-style).

### Phase 2: Train Estimators
- Collect data: (slot, agent_id, true_performance)
- Train VAE estimators (per-agent)
- Train MLP estimator (shared, conditioned on agent_id)

### Phase 3: Continual Learning
```python
for experience_id in range(5):  # 5 experiences
    for batch in experience_data:
        # Get slots
        slots = slot_attention(images)
        
        # For each slot, select top-k agents
        for slot in slots:
            selected_agents = selector.select_top_k(slot)
            
            # Get hidden labels
            hidden_labels = [agent(slot) for agent in selected_agents]
            hidden_concat = torch.cat(hidden_labels)
        
        # Aggregate all slots
        all_hidden = aggregate(all_slots_hidden)
        
        # Incremental tree learning
        tree.learn_one(all_hidden, target)
```

No retraining needed! Tree learns incrementally! 🚀

---

**Status**: Ready to implement remaining components (atomic_agent, aggregator, system)

**Recommendation**: Use Hoeffding Tree from `river` library for true incremental learning.

