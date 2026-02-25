# Final Architecture: Slot Multi-Agent System for Continual Learning

## Overview

A novel continual learning system that combines:
1. **Slot Attention** - Object-centric decomposition
2. **DINO-style SSL** - Self-supervised agent training
3. **Bandit Theory** - Agent selection with exploration-exploitation
4. **Incremental Trees** - Online learning without catastrophic forgetting

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE (32×32)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                        ┌────────▼────────┐
                        │ Slot Attention  │
                        │  (num_slots=7)  │
                        └────────┬────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   7 Slot Tokens       │
                     │   (batch, 7, 64)      │
                     └───────────┬───────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
         Slot 1               Slot 2        ...    Slot 7
            │                    │                    │
    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │   STEP 1:    │    │   STEP 1:    │    │   STEP 1:    │
    │  Estimate    │    │  Estimate    │    │  Estimate    │
    │ Performance  │    │ Performance  │    │ Performance  │
    │  (50 agents) │    │  (50 agents) │    │  (50 agents) │
    └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
            │                    │                    │
            │ VAE/MLP           │ VAE/MLP           │ VAE/MLP
            │ Estimators        │ Estimators        │ Estimators
            │                    │                    │
            ├─► scores[50]       ├─► scores[50]       ├─► scores[50]
            │                    │                    │
    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │   STEP 2:    │    │   STEP 2:    │    │   STEP 2:    │
    │ Bandit-based │    │ Bandit-based │    │ Bandit-based │
    │  Selection   │    │  Selection   │    │  Selection   │
    │  + Weighting │    │  + Weighting │    │  + Weighting │
    └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
            │                    │                    │
            │ UCB/Thompson      │ UCB/Thompson      │ UCB/Thompson
            │ Sampling          │ Sampling          │ Sampling
            │                    │                    │
            ├─► top-k=3         ├─► top-k=3         ├─► top-k=3
            │   weights          │   weights          │   weights
            │                    │                    │
    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │   STEP 3:    │    │   STEP 3:    │    │   STEP 3:    │
    │ Apply agents │    │ Apply agents │    │ Apply agents │
    │ to get       │    │ to get       │    │ to get       │
    │ hidden labels│    │ hidden labels│    │ hidden labels│
    └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
            │                    │                    │
            │ Agent_i(slot)     │ Agent_j(slot)     │ Agent_k(slot)
            │ → softmax(256)    │ → softmax(256)    │ → softmax(256)
            │                    │                    │
            ├─► [p1, p2, p3]    ├─► [p1, p2, p3]    ├─► [p1, p2, p3]
            │   (3 × 256)       │   (3 × 256)       │   (3 × 256)
            │                    │                    │
            └────────────────────┴────────────────────┴─────────┐
                                                                 │
                                                       ┌─────────▼─────────┐
                                                       │   STEP 4:         │
                                                       │  Concatenate      │
                                                       │  All Hidden       │
                                                       │  Labels           │
                                                       └─────────┬─────────┘
                                                                 │
                                                    (7 slots × 3 agents × 256)
                                                    = 5376 features
                                                                 │
                                                       ┌─────────▼─────────┐
                                                       │   STEP 5:         │
                                                       │ Hoeffding Tree    │
                                                       │  (Incremental)    │
                                                       │                   │
                                                       │ Learn online:     │
                                                       │ features → class  │
                                                       └─────────┬─────────┘
                                                                 │
                                                       ┌─────────▼─────────┐
                                                       │   FINAL CLASS     │
                                                       │   PREDICTION      │
                                                       └───────────────────┘
```

---

## Component Details

### 1. Slot Attention (Object-Centric Decomposition)

**Purpose:** Decompose image into semantically meaningful slots (object-centric tokens).

**Architecture:**
```python
SlotAttentionAutoEncoder(
    num_slots=7,
    slot_dim=64,
    num_iterations=3
)
```

**Output:**
- `slots`: (batch_size, 7, 64) - 7 slot tokens per image
- Each slot roughly corresponds to an object or semantic region

**Why?**
- Disentangles objects in the scene
- Makes agent specialization meaningful (per-object processing)
- Provides structured input to agents

---

### 2. Atomic Agents (DINO-Trained, Output Hidden Labels)

**Purpose:** Each agent learns to extract discrete concepts from slots via self-supervised learning.

**Architecture:**
```python
ResidualMLPAgent(
    slot_dim=64,
    hidden_dim=256,
    num_prototypes=256,  # 256 discrete concepts
    num_blocks=3
)
```

**Training (Phase 1): DINO-Style SSL**

```python
# For each slot:
student_logits = student_agent(slot)  # (batch, 256)
teacher_logits = teacher_agent(slot)  # (batch, 256) - no grad

# Teacher: centering + sharpening
teacher_probs = softmax((teacher_logits - center) / temp_teacher)

# Student: sharpening
student_log_probs = log_softmax(student_logits / temp_student)

# Cross-entropy loss
loss = -sum(teacher_probs * student_log_probs)

# Update teacher via EMA
θ_teacher = 0.996 * θ_teacher + 0.004 * θ_student
```

**Output (Phase 2): Hidden Labels**
```python
agent(slot) → logits (256,) → softmax → probabilities (256,)
```
- NOT argmax! Tree handles continuous features.
- Each probability distribution = "hidden label" (concept representation)

**Key Hyperparameters:**
- `num_prototypes=256`: Number of discrete concepts per agent
- `teacher_temp=0.07`: Sharp teacher (confident)
- `student_temp=0.1`: Less sharp student (learning)
- `momentum=0.996`: Slow EMA for teacher

**Why DINO?**
- Self-supervised: No labels needed for agent training
- Prevents collapse: Centering mechanism
- Rich representations: Learns diverse concepts
- Proven at scale: DINOv2 uses this

---

### 3. Performance Estimators (VAE/MLP)

**Purpose:** Estimate how well each agent would perform on a given slot (without actually running the agent).

**VAE Estimator:**
```python
VAEEstimator(
    slot_dim=64,
    latent_dim=16
)

# Returns reconstruction error as performance estimate
score = -vae.reconstruction_error(slot)  # Lower error = higher score
```

**MLP Estimator:**
```python
MLPEstimator(
    slot_dim=64,
    hidden_dim=128
)

# Directly predicts performance score
score = mlp(slot)  # (batch,) - scalar score
```

**Why?**
- Running all 50 agents is expensive
- Estimators are lightweight (VAE: ~100K params, MLP: ~50K params)
- Agent is heavy (~1M params)
- Amortized inference: Learn to predict performance

---

### 4. Bandit-Based Agent Selection + Weighting ⭐ NEW

**Purpose:** Select top-k agents while balancing exploration-exploitation.

**Problem:**
- Pure greedy (always top-k by estimate) → No exploration
- May miss better agents that are under-explored
- Bandit theory solves this!

**Supported Strategies:**

#### a) UCB (Upper Confidence Bound)
```python
UCBSelector(
    num_agents=50,
    exploration_constant=2.0
)

# UCB score for each agent:
ucb[i] = mean_reward[i] + c * sqrt(log(t) / count[i])
         \_____________/   \_________________________/
          Exploitation         Exploration bonus

# Agents with high reward OR low visit count get bonus
```

#### b) Thompson Sampling
```python
ThompsonSamplingSelector(num_agents=50)

# For each agent, maintain Beta(α, β) distribution
# Sample from posterior: p ~ Beta(α, β)
# Select agents with highest sampled probabilities
```

#### c) Epsilon-Greedy
```python
EpsilonGreedySelector(
    num_agents=50,
    epsilon=0.1  # 10% exploration
)

# With prob ε: random selection
# With prob 1-ε: greedy (top-k by estimate)
```

#### d) Weighted Top-K (Baseline, No Exploration)
```python
WeightedTopKSelector(num_agents=50)

# Simple softmax weighting of top-k agents
# No exploration, pure exploitation
```

**Output:**
```python
selected_indices, weights = bandit_selector.select_and_weight(
    slot=slot,
    estimated_scores=scores,
    k=3
)
# selected_indices: (batch, 3) - which agents
# weights: (batch, 3) - how to weight them (sum to 1)
```

**Why Bandit Theory?**
- Principled exploration-exploitation trade-off
- Provably optimal in certain settings (UCB has regret bounds)
- Adapts over time (learns which agents are best)
- Handles non-stationary rewards (continual learning!)

**TODO:** Specific methodology to be determined by professor.

---

### 5. Hoeffding Tree Aggregator (Incremental Learning)

**Purpose:** Learn mapping from hidden labels → final class, incrementally (one example at a time).

**Architecture:**
```python
IncrementalTreeAggregator(
    grace_period=200,
    split_confidence=1e-5,
    leaf_prediction='nba',  # Naive Bayes Adaptive
    adaptive=True  # Handles concept drift
)
```

**Input:** Concatenated hidden labels
```python
# For each image:
hidden_labels = []
for slot in slots:
    top_k_agents = bandit_selector.select(slot)
    for agent_idx in top_k_agents:
        prob_dist = agents[agent_idx](slot)  # (256,) softmax
        hidden_labels.append(prob_dist)

# Concatenate: (7 slots × 3 agents × 256 prototypes) = 5376 features
final_features = concatenate(hidden_labels)  # (5376,)
```

**Training (Phase 2):**
```python
# Online learning (one example at a time)
for image, label in dataloader:
    hidden_labels = extract_hidden_labels(image)
    tree.learn_one(hidden_labels, label)  # Incremental update
```

**Prediction:**
```python
hidden_labels = extract_hidden_labels(image)
predicted_class = tree.predict_one(hidden_labels)
```

**Why Hoeffding Tree?**
- ✅ Incremental: Learns one example at a time
- ✅ Handles continuous features: Softmax probabilities work directly
- ✅ New classes: Can learn new classes without retraining
- ✅ No task ID: Decision based on features alone
- ✅ No catastrophic forgetting: Tree grows, doesn't overwrite
- ✅ Concept drift: Adaptive variant handles distribution shift

**Hoeffding Bound:**
```
With probability 1-δ, true best split is within ε of observed best split:
ε = sqrt(R² * log(1/δ) / (2 * n))

where:
    R = range of split criterion
    n = number of examples seen
    δ = confidence (default: 1e-5)
```

---

## Training Pipeline

### Phase 1: Train Agents (DINO SSL, Unsupervised) 🚀

```python
# Initialize
student_agents, teacher_agents = create_agent_pool(
    num_agents=50,
    slot_dim=64,
    num_prototypes=256
)

estimators = create_estimator_pool(num_agents=50, estimator_type='vae')
dino_losses = [DINOLoss(256) for _ in range(50)]

# Training loop
for epoch in range(10):
    for images in unlabeled_dataloader:
        _, slots, _ = slot_attention(images)  # (batch, 7, 64)
        
        total_loss = 0
        for slot_idx in range(7):
            slot = slots[:, slot_idx, :]
            
            # Estimate performance
            scores = [estimators[i](slot) for i in range(50)]
            top_k = torch.topk(scores, k=3).indices
            
            # Train selected agents
            for agent_idx in top_k:
                # Student forward
                student_logits = student_agents[agent_idx](slot, return_logits=True)
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_logits = teacher_agents[agent_idx](slot, return_logits=True)
                
                # DINO loss
                loss = dino_losses[agent_idx](student_logits, teacher_logits)
                total_loss += loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        # Update teachers (EMA)
        for student, teacher in zip(student_agents, teacher_agents):
            update_teacher(student, teacher, momentum=0.996)
```

**Result:** Agents learn rich, diverse representations without labels!

---

### Phase 2: Train Tree (Incremental, Supervised) 🌳

```python
# Freeze agents
for agent in student_agents:
    agent.eval()

# Initialize tree
tree = IncrementalTreeAggregator(adaptive=True)

# Bandit selector
bandit_selector = create_bandit_selector('ucb', num_agents=50)

# Training loop (incremental)
for images, labels in labeled_dataloader:
    for i in range(images.size(0)):
        image = images[i:i+1]
        label = labels[i].item()
        
        # Extract hidden labels
        with torch.no_grad():
            _, slots, _ = slot_attention(image)
            
            hidden_labels = []
            for slot_idx in range(7):
                slot = slots[:, slot_idx, :]
                
                # Estimate performance
                scores = torch.stack([
                    -estimators[j](slot)
                    for j in range(50)
                ], dim=1)
                
                # Bandit-based selection + weighting
                top_k_indices, weights = bandit_selector.select_and_weight(
                    slot, scores, k=3
                )
                
                # Get hidden labels from selected agents
                for k_idx in range(3):
                    agent_idx = top_k_indices[0, k_idx].item()
                    prob_dist = student_agents[agent_idx](slot)  # (1, 256)
                    
                    # Optionally weight by bandit weight
                    # prob_dist = prob_dist * weights[0, k_idx]
                    
                    hidden_labels.append(prob_dist.cpu().numpy().flatten())
            
            # Concatenate
            final_features = np.concatenate(hidden_labels)  # (5376,)
        
        # Online learning
        tree.learn_one(final_features, label)
        
        # Optional: Update bandit statistics
        # (requires computing reward, e.g., prediction accuracy)
```

**Result:** Tree learns incrementally, no catastrophic forgetting!

---

### Phase 3: Continual Learning (New Tasks) 🔄

```python
# Task 1: Classes 0-9 (already trained)

# Task 2: Classes 10-19 (new classes!)
for images, labels in new_task_dataloader:
    for i in range(images.size(0)):
        hidden_labels = extract_hidden_labels(images[i:i+1])
        tree.learn_one(hidden_labels, labels[i].item())

# Agents are frozen, only tree is updated
# Tree grows new branches for new classes
# Old knowledge preserved (no overwriting)
```

---

## Key Hyperparameters

| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| **Slot Attention** | `num_slots` | 7 | Number of object slots |
| | `slot_dim` | 64 | Slot embedding dimension |
| | `num_iterations` | 3 | Attention iterations |
| **Agents** | `num_agents` | 50 | Pool size |
| | `hidden_dim` | 256 | MLP hidden dimension |
| | `num_prototypes` | 256 | Discrete concepts per agent |
| | `num_blocks` | 3 | Residual blocks |
| **DINO Training** | `teacher_temp` | 0.07 | Sharp teacher |
| | `student_temp` | 0.1 | Less sharp student |
| | `momentum` | 0.996 | EMA for teacher |
| | `center_momentum` | 0.9 | EMA for centering |
| **Selection** | `k` | 3 | Top-k agents per slot |
| **Bandit (UCB)** | `exploration_constant` | 2.0 | Exploration bonus |
| **Tree** | `grace_period` | 200 | Examples before split |
| | `split_confidence` | 1e-5 | Hoeffding bound |
| | `leaf_prediction` | 'nba' | Naive Bayes Adaptive |

---

## Ablation Studies (TODO)

To validate design choices:

1. **Slot Attention vs. No Slots**
   - Baseline: Apply agents directly to full image features
   - Expected: Slots improve interpretability and specialization

2. **DINO SSL vs. Random Init vs. Supervised**
   - Baseline 1: Random agent initialization
   - Baseline 2: Train agents end-to-end with supervised loss
   - Expected: DINO gives better representations

3. **Bandit Selection vs. Fixed Top-K**
   - Baseline: Always select same top-k agents (no exploration)
   - Expected: Bandit improves over time by exploring

4. **Hoeffding Tree vs. Standard Tree vs. MLP**
   - Baseline 1: Retrain standard tree for each task
   - Baseline 2: MLP classifier (prone to forgetting)
   - Expected: Hoeffding Tree handles continual learning best

5. **Hidden Labels (Softmax) vs. Argmax vs. Embeddings**
   - Baseline 1: Use argmax (discrete ID only)
   - Baseline 2: Use continuous embeddings (before softmax)
   - Expected: Softmax probabilities balance discrete + continuous

---

## Limitations & Future Work

### Current Limitations

1. **Fixed Slot Count**
   - `num_slots=7` is fixed, may not fit all images
   - Solution: AdaSlot (adaptive slot count)

2. **No Agent Specialization Enforcement**
   - Agents may learn redundant concepts
   - Solution: Add diversity loss (e.g., orthogonality constraint)

3. **Bandit Rewards Not Yet Defined**
   - How to compute reward for agent selection?
   - Options: Accuracy, confidence, diversity
   - TODO: Consult with professor

4. **Large Input to Tree**
   - 5376 features (7 × 3 × 256) is high-dimensional
   - May slow down tree learning
   - Solution: Dimensionality reduction (PCA, autoencoder)

5. **No Task Boundaries**
   - Agents trained on all data at once (Phase 1)
   - What if new tasks have very different images?
   - Solution: Continual agent training (difficult!)

### Future Directions

1. **Adaptive Architecture**
   - Dynamic slot count (AdaSlot)
   - Dynamic agent count (add new agents for new tasks)
   - Dynamic k (adaptive top-k selection)

2. **Improved Agent Training**
   - Multi-crop augmentation (like DINOv2)
   - iBOT masking for local features
   - Larger prototype count (1024, 4096)

3. **Better Bandit Rewards**
   - Incorporate confidence and diversity
   - Meta-learning to learn reward function
   - Contextual bandits (slot-dependent strategies)

4. **Hierarchical Trees**
   - Ensemble of trees (Random Forest)
   - Hierarchical tree structure (coarse → fine classes)
   - Mixture of experts at leaves

5. **Theoretical Analysis**
   - Regret bounds for bandit selection
   - Generalization bounds for incremental tree
   - Sample complexity for continual learning

---

## Implementation Status

✅ **Completed:**
- Atomic agents with DINO training (`atomic_agent.py`)
- Performance estimators (`estimators.py`)
- Bandit-based selection (`bandit_selector.py`)
- Incremental tree aggregator (`aggregator.py`)

⏳ **In Progress:**
- Full system integration (`system.py`)
- Training scripts

❌ **TODO:**
- Experiments on CIFAR-100 / Tiny-ImageNet
- Ablation studies
- Bandit reward definition (待教授確認)
- Hyperparameter tuning

---

## References

1. **Slot Attention:**
   - Locatello et al. (2020) "Object-Centric Learning with Slot Attention"

2. **DINOv2:**
   - Oquab et al. (2023) "DINOv2: Learning Robust Visual Features without Supervision"
   - GitHub: https://github.com/facebookresearch/dinov2

3. **Hoeffding Tree:**
   - Domingos & Hulten (2000) "Mining High-Speed Data Streams"
   - River Library: https://riverml.xyz/

4. **Multi-Armed Bandits:**
   - Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem" (UCB)
   - Thompson (1933) "On the Likelihood..." (Thompson Sampling)

5. **Continual Learning:**
   - Avalanche Library: https://avalanche.continualai.org/
   - van de Ven & Tolias (2019) "Three scenarios for continual learning"

---

## Contact & Questions

For methodology questions (especially bandit rewards):
→ Consult with professor

For implementation questions:
→ See code in `src/slot_multi_agent/`

---

**Last Updated:** 2026-02-13


