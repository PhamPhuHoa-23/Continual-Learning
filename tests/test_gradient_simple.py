"""Simple test to isolate gradient flow issue"""
import torch
import torch.nn as nn
from src.compositional.pipeline import CompositionalRoutingPipeline

print("=== Creating pipeline ===")
pipeline = CompositionalRoutingPipeline(
    slot_dim=64,
    d_h=128,
    latent_dim=32,
    device='cpu'
)

# Spawn 5 agents like Task 1
print("\n=== Spawning 5 agents manually ===")
for i in range(5):
    aid = f"agent_{i}"
    pipeline._spawn_agent(aid)
    pipeline.new_agent_ids.append(aid)
    # Create fake cluster center
    if pipeline.task1_cluster_centers is None:
        pipeline.task1_cluster_centers = {}
    pipeline.task1_cluster_centers[aid] = torch.randn(64)

print(f"Agents: {list(pipeline.agents.keys())}")
print(f"Keys: {list(pipeline.aggregator.keys.keys())}")
print(f"Cluster centers: {len(pipeline.task1_cluster_centers)}")

# Check trainability
pipeline.train_mode()
trainable = pipeline.trainable_params()
print(f"Trainable params: {len(trainable)}")

# Create fake batch
print("\n=== Testing forward pass ===")
B, K = 4, 7
slots = torch.randn(B, K, 64)
labels = torch.randint(0, 10, (B,))

print(f"Input: slots {slots.shape}, labels {labels.shape}")

# Forward
losses = pipeline.compute_losses(slots, labels, task_id=1)

print("\n=== Loss results ===")
for k, v in losses.items():
    print(f"{k}: {v.item():.6f}")
    print(f"  requires_grad: {v.requires_grad}")
    print(f"  grad_fn: {v.grad_fn}")

# Test backward
print("\n=== Testing backward ===")
total_loss = losses["total"]

if total_loss.grad_fn is None:
    print("❌ FAILURE: Loss has no grad_fn!")
else:
    print("✅ SUCCESS: Loss has grad_fn")
    try:
        optimizer = torch.optim.Adam(trainable, lr=0.001)
        optimizer.zero_grad()
        total_loss.backward()
        print("✅ Backward pass completed!")
        
        # Check gradients
        grads = [p for p in trainable if p.grad is not None]
        print(f"Parameters with gradients: {len(grads)}/{len(trainable)}")
    except Exception as e:
        print(f"❌ Backward failed: {e}")
