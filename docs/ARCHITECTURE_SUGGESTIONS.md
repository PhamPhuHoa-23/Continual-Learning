# Architecture Suggestions for Token-Processing Agents

## 🎯 Context

**Input**: Slots từ Slot Attention = Tokens (vectors, e.g., 64-dim)  
**Output**: Hidden labels (DINO-style embeddings) → Decision Tree  
**NOT**: Raw images (không cần CNN!)

---

## 💡 Agent Architecture Options

### Option 1: Simple MLP (Lightweight) ⭐⭐⭐⭐⭐

**Best cho**: Fast, simple, effective với vector inputs

```python
class MLPAgent(nn.Module):
    """
    Process token → hidden representation.
    Lightweight, fast, suitable for vectors.
    """
    def __init__(self, slot_dim=64, hidden_dims=[128, 256, 128], output_dim=128):
        super().__init__()
        
        layers = []
        in_dim = slot_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),  # Stabilize training
                nn.GELU(),            # Smooth activation
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        
        # Projection head (DINO-style)
        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(in_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, slot):
        """
        Args:
            slot: (slot_dim,) - token from Slot Attention
        Returns:
            hidden_label: (output_dim,) - embedding for Decision Tree
        """
        features = self.backbone(slot)
        hidden_label = self.projection(features)
        return hidden_label
```

**Pros**:
- ✅ Very fast (~0.1ms)
- ✅ Lightweight (<1M params)
- ✅ Suitable for vector inputs
- ✅ Easy to train

**Cons**:
- ❌ Limited capacity
- ❌ No temporal/sequential modeling

---

### Option 2: Transformer Encoder (More Powerful) ⭐⭐⭐⭐

**Best cho**: If need richer representations, self-attention

```python
class TransformerAgent(nn.Module):
    """
    Use Transformer encoder to process token.
    Can attend to different dimensions of slot.
    """
    def __init__(self, slot_dim=64, num_heads=4, num_layers=2, output_dim=128):
        super().__init__()
        
        # Project slot to higher dim for transformer
        self.input_proj = nn.Linear(slot_dim, 256)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (DINO-style)
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, slot):
        """
        Args:
            slot: (slot_dim,)
        Returns:
            hidden_label: (output_dim,)
        """
        # Add sequence dimension (treat slot as 1-token sequence)
        x = slot.unsqueeze(0)  # (1, slot_dim)
        
        # Project
        x = self.input_proj(x)  # (1, 256)
        
        # Transformer
        x = self.transformer(x)  # (1, 256)
        
        # Project to hidden label
        hidden_label = self.projection(x.squeeze(0))  # (output_dim,)
        
        return hidden_label
```

**Pros**:
- ✅ More expressive
- ✅ Self-attention on slot dimensions
- ✅ Better for complex patterns

**Cons**:
- ❌ Slower (~1-2ms)
- ❌ More parameters (~5-10M)
- ❌ Might be overkill for simple vectors

---

### Option 3: Residual MLP (Balanced) ⭐⭐⭐⭐⭐

**Best cho**: Balance between simplicity and capacity

```python
class ResidualMLPAgent(nn.Module):
    """
    MLP with residual connections.
    Good balance: more capacity than simple MLP, faster than Transformer.
    """
    def __init__(self, slot_dim=64, hidden_dim=256, num_blocks=3, output_dim=128):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(slot_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output projection (DINO-style)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, slot):
        x = self.input_proj(slot)
        
        for block in self.blocks:
            x = block(x)
        
        hidden_label = self.projection(x)
        return hidden_label


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return x + self.net(x)  # Residual connection
```

**Pros**:
- ✅ Good capacity
- ✅ Fast (~0.5ms)
- ✅ Stable training (residuals)
- ✅ Moderate params (~2-3M)

**Cons**:
- ❌ More complex than simple MLP

---

## 🔄 Training Strategy

### Phase 1: Self-Supervised (DINO-style)

```python
def train_self_supervised(agents, slot_attention, images):
    """
    Train agents to generate good hidden representations.
    Similar to DINO: consistency between augmented views.
    """
    # Two augmented views
    aug1 = augment(images)
    aug2 = augment(images)
    
    # Get slots from both views
    slots1 = slot_attention(aug1)  # (B, num_slots, slot_dim)
    slots2 = slot_attention(aug2)  # (B, num_slots, slot_dim)
    
    loss = 0
    for slot_idx in range(num_slots):
        s1 = slots1[:, slot_idx]  # (B, slot_dim)
        s2 = slots2[:, slot_idx]  # (B, slot_dim)
        
        # Random agent processes both views
        agent = random.choice(agents)
        
        h1 = agent(s1)  # (B, output_dim)
        h2 = agent(s2)  # (B, output_dim)
        
        # Consistency loss (MSE or cosine similarity)
        loss += F.mse_loss(h1, h2)
        # Or: loss += 1 - F.cosine_similarity(h1, h2).mean()
    
    return loss / num_slots
```

**Goal**: Agents learn to extract **consistent** features from slots, regardless of augmentations.

---

### Phase 2: Supervised (With Decision Tree)

```python
def train_supervised(agents, selector, slot_attention, decision_tree, 
                     images, targets):
    """
    Train end-to-end:
    1. Get slots
    2. Select top-k agents per slot
    3. Generate hidden labels
    4. Train decision tree
    """
    # Get slots
    slots = slot_attention(images)  # (B, num_slots, slot_dim)
    
    all_hidden_labels = []
    
    for b in range(B):
        sample_hidden = []
        
        for slot_idx in range(num_slots):
            slot = slots[b, slot_idx]
            
            # Select top-k agents
            selected_agents, scores = selector.select_top_k(slot, k=3)
            
            # Get hidden labels from selected agents
            slot_hidden = []
            for agent in selected_agents:
                h = agent(slot)  # (output_dim,)
                slot_hidden.append(h)
            
            # Aggregate (mean, weighted, etc.)
            slot_aggregated = torch.stack(slot_hidden).mean(dim=0)
            sample_hidden.append(slot_aggregated)
        
        # Aggregate across slots
        sample_aggregated = torch.stack(sample_hidden).mean(dim=0)
        all_hidden_labels.append(sample_aggregated)
    
    hidden_labels = torch.stack(all_hidden_labels)  # (B, output_dim)
    
    # Train decision tree (sklearn or custom differentiable tree)
    decision_tree.fit(hidden_labels.detach().numpy(), targets.numpy())
    
    # Or use soft decision tree (differentiable)
    # logits = decision_tree(hidden_labels)
    # loss = F.cross_entropy(logits, targets)
    
    return hidden_labels
```

---

### Phase 3: Train Sub-Networks (Performance Estimation)

```python
def train_estimators(agents, selector, slot_attention, train_loader):
    """
    Train sub-networks to estimate agent performance.
    
    Ground truth: Run agents and measure actual performance.
    """
    for images, targets in train_loader:
        slots = slot_attention(images)
        
        for b in range(B):
            for slot_idx in range(num_slots):
                slot = slots[b, slot_idx]
                
                # Get ground truth: run all agents and measure
                true_performances = []
                for agent in agents:
                    # Run agent
                    hidden = agent(slot)
                    
                    # Measure performance (e.g., prototype distance)
                    perf = measure_quality(hidden, targets[b])
                    true_performances.append(perf)
                
                # Train estimators to predict these performances
                for agent_id, (agent, estimator) in enumerate(zip(agents, estimators)):
                    predicted_perf = estimator(slot)
                    true_perf = true_performances[agent_id]
                    
                    loss = F.mse_loss(predicted_perf, true_perf)
                    # Update estimator...
```

---

## 🎯 Recommendation

### For Your Use Case:

**Option: Residual MLP Agent** ⭐⭐⭐⭐⭐

**Reasons**:
1. ✅ **Slots are vectors**: Don't need spatial processing (CNN) or heavy attention (Transformer)
2. ✅ **Fast**: Need to process many slot-agent pairs
3. ✅ **Good capacity**: Residual connections provide depth without instability
4. ✅ **DINO-compatible**: MLP architectures work well with self-supervised learning
5. ✅ **Continual learning**: Easier to expand/adapt than complex architectures

### Architecture:

```python
Agent Architecture:
Input: Slot (64-dim token)
  ↓
Input Projection: 64 → 256
  ↓
3× Residual Blocks (256-dim)
  ├─ Linear 256 → 512
  ├─ LayerNorm + GELU
  ├─ Linear 512 → 256
  └─ Residual connection
  ↓
Projection Head (DINO-style):
  ├─ Linear 256 → 256
  ├─ GELU
  └─ Linear 256 → 128
  ↓
Output: Hidden label (128-dim embedding)
```

---

## 📊 Decision Tree for Continual Learning

### Options:

#### 1. Incremental Decision Tree (sklearn-based)

```python
from sklearn.tree import DecisionTreeClassifier

class IncrementalTree:
    def __init__(self):
        self.tree = DecisionTreeClassifier(max_depth=10)
        self.seen_classes = set()
        self.all_data = []
        self.all_targets = []
    
    def partial_fit(self, X, y):
        # Accumulate data
        self.all_data.append(X)
        self.all_targets.append(y)
        self.seen_classes.update(y)
        
        # Retrain on all data
        X_all = np.concatenate(self.all_data)
        y_all = np.concatenate(self.all_targets)
        
        self.tree.fit(X_all, y_all)
```

**Pros**: Simple, supports new classes  
**Cons**: Not differentiable, retrains on all data

---

#### 2. Soft Decision Tree (Differentiable)

```python
class SoftDecisionTree(nn.Module):
    """
    Differentiable decision tree.
    Can train end-to-end with agents.
    """
    def __init__(self, input_dim=128, depth=5, num_classes=100):
        super().__init__()
        self.depth = depth
        num_nodes = 2 ** depth - 1
        
        # Each node has a decision hyperplane
        self.node_weights = nn.Parameter(torch.randn(num_nodes, input_dim))
        self.node_biases = nn.Parameter(torch.zeros(num_nodes))
        
        # Leaf predictions
        num_leaves = 2 ** depth
        self.leaf_logits = nn.Parameter(torch.randn(num_leaves, num_classes))
    
    def forward(self, x):
        # Soft routing through tree
        # ... (implementation)
        return logits
```

**Pros**: Differentiable, end-to-end training  
**Cons**: More complex, limited depth

---

#### 3. Gradient Boosting (Continual-friendly)

```python
from sklearn.ensemble import GradientBoostingClassifier

class ContinualGradientBoosting:
    def __init__(self):
        self.models = []  # One model per task
        self.task_classifiers = []
    
    def add_task(self, X, y):
        # Train new boosting model for this task
        model = GradientBoostingClassifier(n_estimators=100)
        model.fit(X, y)
        self.models.append(model)
    
    def predict(self, X, task_id=None):
        if task_id is None:
            # Ensemble all models
            predictions = [m.predict_proba(X) for m in self.models]
            return np.mean(predictions, axis=0).argmax(axis=1)
        else:
            return self.models[task_id].predict(X)
```

**Pros**: Strong performance, task-specific models  
**Cons**: Need task ID at inference (or task classifier)

---

## 🎓 Complete Training Pipeline

```python
# Phase 1: Self-supervised (weeks 1-2)
for epoch in range(50):
    loss = train_self_supervised(agents, slot_attention, unlabeled_images)
    # Agents learn to extract consistent features

# Phase 2: Train estimators (week 3)
for epoch in range(20):
    loss = train_estimators(agents, estimators, slot_attention, labeled_data)
    # Sub-networks learn to predict agent performance

# Phase 3: End-to-end with Decision Tree (weeks 4-5)
for epoch in range(50):
    # Get hidden labels
    hidden_labels = get_hidden_labels(agents, selector, slot_attention, images)
    
    # Train decision tree
    decision_tree.partial_fit(hidden_labels, targets)
    
    # Optional: Fine-tune agents with tree feedback
    # (if using soft tree)

# Phase 4: Continual learning (ongoing)
for task_id, (task_data, task_targets) in enumerate(tasks):
    # New task arrives
    hidden_labels = get_hidden_labels(..., task_data)
    
    # Expand decision tree
    decision_tree.add_task(task_id, hidden_labels, task_targets)
    
    # Optionally: Fine-tune specific agents for this task
```

---

## 💡 Key Insights

1. **Slots = Tokens**: Use MLP/Residual architectures, NOT CNNs
2. **Hidden Labels**: Like DINO embeddings, feed to Decision Tree
3. **Continual Learning**: Decision Tree easier to expand than neural classifier
4. **Training**: Self-supervised → Supervised → Continual
5. **Performance Estimation**: Train sub-networks after agents are stable

---

## ❓ Questions to Clarify

1. **Decision Tree**: Sklearn (simple) or differentiable (end-to-end)?
2. **Aggregation**: How to combine outputs from multiple agents? (mean, weighted, attention?)
3. **Task ID**: At inference, do we know task ID? Or need to infer it?
4. **Number of agents**: How many specialized agents? (10? 50?)
5. **Continual learning strategy**: Task-incremental, class-incremental, or domain-incremental?

Let me know và tôi sẽ implement chi tiết! 🚀

