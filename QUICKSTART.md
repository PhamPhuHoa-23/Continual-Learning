# Quick Start: Slot-based Multi-Agent System

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements_slot_agent.txt

# Key package: river (for Hoeffding Tree)
pip install river
```

## 📦 Basic Usage

### 1. Create System

```python
from src.slot_multi_agent import SlotMultiAgentSystem

# Create system
system = SlotMultiAgentSystem(
    num_agents=50,        # 50 specialized agents
    num_slots=7,          # 7 object slots
    slot_dim=64,          # 64-dim slot tokens
    k=3,                  # Top-3 agents per slot
    num_classes=100,      # CIFAR-100
    input_channels=3,
    input_size=32,
    estimator_type='vae', # or 'mlp'
    aggregator_type='hoeffding',  # Incremental tree
    aggregate_mode='concat',  # or 'mean'
    device='cuda'
)
```

### 2. Training (Continual Learning)

```python
from src.data import get_avalanche_cifar100_benchmark

# Get CIFAR-100 benchmark (5 experiences, 20 classes each)
benchmark = get_avalanche_cifar100_benchmark(n_experiences=5, seed=42)

# Continual learning loop
for exp_id, train_exp in enumerate(benchmark.train_stream):
    print(f"\n=== Experience {exp_id} ===")
    print(f"Classes: {train_exp.classes_in_this_experience}")
    
    # Training
    for images, labels, _ in train_exp.dataset:
        images = images.unsqueeze(0)  # Add batch dim if needed
        labels = torch.tensor([labels])
        
        # Incremental learning (one step)
        info = system.train_step(images, labels)
        
        if len(images) % 100 == 0:
            print(f"Accuracy: {info['accuracy']:.4f}")
    
    # Evaluation on all seen classes
    print(f"\nEvaluating on all {exp_id+1} experiences...")
    for test_exp_id in range(exp_id + 1):
        test_exp = benchmark.test_stream[test_exp_id]
        
        all_images = []
        all_labels = []
        for images, labels, _ in test_exp.dataset:
            all_images.append(images)
            all_labels.append(labels)
        
        test_images = torch.stack(all_images)
        test_labels = torch.tensor(all_labels)
        
        metrics = system.evaluate(test_images, test_labels)
        print(f"Experience {test_exp_id} Accuracy: {metrics['accuracy']:.4f}")
```

### 3. Simple Example (Single Batch)

```python
import torch

# Create system
system = SlotMultiAgentSystem(
    num_agents=50,
    num_slots=7,
    k=3,
    num_classes=100,
    device='cuda'
)

# Random batch
images = torch.randn(8, 3, 32, 32).cuda()
labels = torch.randint(0, 20, (8,)).cuda()

# Train step (incremental)
info = system.train_step(images, labels)
print(f"Accuracy: {info['accuracy']:.4f}")

# Predict
predictions = system(images)
print(f"Predictions: {predictions}")

# System info
info = system.get_system_info()
print(f"Tree classes seen: {info['tree']['classes']}")
print(f"Num samples processed: {info['tree']['num_samples_seen']}")
```

## 🎯 Architecture Flow

```
CIFAR-100 Image (8, 3, 32, 32)
  ↓
CNN Encoder → Features (8, 256, H, W)
  ↓
Slot Attention → 7 Slots (8, 7, 64)
  ↓
For EACH of 7 slots:
  ├─ 50 Estimators → 50 scores
  ├─ TopK Selector → 3 best agents
  ├─ 3 Agents → 3 × 128 = 384-dim
  └─ (per slot)
  ↓
Aggregate 7 slots:
  Concat → 7 × 384 = 2688-dim
  ↓
Hoeffding Tree (incremental)
  ├─ Learns online, no retraining
  ├─ Adds splits when confident
  └─ Supports new classes
  ↓
Predictions (8,)
```

## 🔧 Configuration Options

### Estimator Types

```python
# VAE-based (reconstruction error)
system = SlotMultiAgentSystem(estimator_type='vae', ...)

# MLP-based (learned mapping)
system = SlotMultiAgentSystem(estimator_type='mlp', ...)
```

### Aggregator Types

```python
# Hoeffding Tree (recommended - true incremental)
system = SlotMultiAgentSystem(aggregator_type='hoeffding', ...)

# Incremental sklearn tree (retrain with storage)
system = SlotMultiAgentSystem(aggregator_type='incremental', ...)

# Soft Decision Tree (differentiable)
system = SlotMultiAgentSystem(aggregator_type='soft', ...)
```

### Aggregation Modes

```python
# Concatenate all slots (2688-dim input to tree)
system = SlotMultiAgentSystem(aggregate_mode='concat', ...)

# Mean pool over slots (384-dim input to tree)
system = SlotMultiAgentSystem(aggregate_mode='mean', ...)
```

## 📊 Monitoring

```python
# Get system info
info = system.get_system_info()

print(f"Num agents: {info['num_agents']}")
print(f"Num slots: {info['num_slots']}")
print(f"Top-k: {info['k']}")
print(f"Tree info: {info['tree']}")
print(f"Avg FLOPs: {info['avg_flops_per_sample']:,.0f}")
```

## 💾 Save/Load

```python
# Save checkpoint
system.save_checkpoint('checkpoints/system.pth')

# Load checkpoint
system.load_checkpoint('checkpoints/system.pth')
```

## 🎓 Advanced: Training Estimators

```python
# If you want to pre-train estimators before main training
from src.slot_multi_agent.estimators import VAEEstimator

# Example: Train VAE estimator
for agent_id, estimator in enumerate(system.estimators):
    if isinstance(estimator, VAEEstimator):
        # Collect slots
        all_slots = []
        for images, _ in train_loader:
            _, metadata = system.forward(images, return_metadata=True)
            slots = metadata['slots']  # (B, num_slots, slot_dim)
            all_slots.append(slots.reshape(-1, system.slot_dim))
        
        all_slots = torch.cat(all_slots)
        
        # Train VAE
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
        for epoch in range(10):
            loss, recon_loss, kl_loss = estimator.compute_loss(all_slots)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Agent {agent_id} Epoch {epoch}: Loss={loss:.4f}")
```

## 📈 Evaluation

```python
# Full evaluation
metrics = system.evaluate(test_images, test_labels)

print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
print(f"Per-class Accuracy:")
for cls, acc in metrics['per_class_accuracy'].items():
    print(f"  Class {cls}: {acc:.4f}")
```

## 🐛 Troubleshooting

### Issue: "river not installed"
```bash
pip install river
```

### Issue: "Tree not fitted"
Make sure to call `train_step()` before `evaluate()`:
```python
# Train first
system.train_step(images, labels)

# Then evaluate
predictions = system(images)
```

### Issue: Out of memory
Reduce batch size or use `aggregate_mode='mean'`:
```python
system = SlotMultiAgentSystem(
    aggregate_mode='mean',  # 384-dim instead of 2688-dim
    ...
)
```

## 🎯 Next Steps

1. **Train on CIFAR-100**: Use provided continual learning script
2. **Experiment with k**: Try different top-k values (2, 3, 5)
3. **Compare estimators**: VAE vs MLP performance
4. **Visualize slots**: Use attention maps to see what each slot captures
5. **Analyze agents**: See which agents are selected most often

---

**Ready to start! 🚀**

