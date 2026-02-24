"""
Debug Task 1 initialization and routing
"""
import torch
import numpy as np
from src.compositional.pipeline import CompositionalRoutingPipeline

print("=== Creating Pipeline ===")
pipeline = CompositionalRoutingPipeline(
    slot_dim=64,
    d_h=128,
    latent_dim=32,
    device='cpu'
)

print("\n=== Simulating Task 1 Clustering ===")
# Simulate what init_agents_from_clustering does
num_samples = 100
num_slots_per_sample = 7
all_slots = torch.randn(num_samples * num_slots_per_sample, 64)

print(f"Generated {all_slots.shape[0]} slots for clustering")

# Create a generator like encode_slots would return
def fake_encoded_iter():
    batch_size = 32
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        X = torch.randn(end - i, 3, 32, 32)
        y = torch.randint(0, 10, (end - i,))
        slots = torch.randn(end - i, num_slots_per_sample, 64)
        yield X, y, slots

# Initialize agents
M0 = 5
print(f"\n=== Initializing {M0} agents via clustering ===")
pipeline.init_agents_from_clustering(fake_encoded_iter(), M0=M0)

print(f"Created {len(pipeline.agents)} agents")
print(f"Created {len(pipeline.aggregator.keys)} keys")
print(f"Cluster centers: {len(pipeline.task1_cluster_centers) if pipeline.task1_cluster_centers else 0}")

if pipeline.task1_cluster_centers:
    for aid, center in pipeline.task1_cluster_centers.items():
        print(f"  {aid}: center shape={center.shape}, mean={center.mean():.4f}, std={center.std():.4f}")

print("\n=== Testing Routing with Cluster Centers ===")
test_slots = torch.randn(4, 7, 64)  # 4 samples, 7 slots
slots_flat = test_slots.view(-1, 64)

with torch.no_grad():
    sigma_flat = pipeline._route_slots(slots_flat)

assigned_count = sum(1 for s in sigma_flat if s != "unassigned")
print(f"Assigned slots: {assigned_count}/{len(sigma_flat)}")

# Show assignments
assignments = {}
for s in sigma_flat:
    assignments[s] = assignments.get(s, 0) + 1
print(f"Assignment distribution: {assignments}")

print("\n=== Testing compute_losses ===")
pipeline.train_mode()

test_labels = torch.randint(0, 10, (4,))
losses = pipeline.compute_losses(test_slots, test_labels, task_id=1)

print(f"Loss results:")
for k, v in losses.items():
    has_grad_fn = v.grad_fn is not None
    print(f"  {k}: {v.item():.6f}, has_grad_fn={has_grad_fn}")

print("\n=== Testing Backward ===")
total_loss = losses["total"]
print(f"Loss requires_grad: {total_loss.requires_grad}")
print(f"Loss grad_fn: {total_loss.grad_fn}")

if total_loss.grad_fn is None:
    print("❌ ERROR: No grad_fn!")
    
    # Debug the dummy loss creation
    print("\n=== Debugging Dummy Loss ===")
    print(f"Number of trainable keys: {sum(1 for k in pipeline.aggregator.keys.values() if k.requires_grad)}")
    print(f"Number of trainable agents: {sum(1 for a in pipeline.agents.values() if any(p.requires_grad for p in a.parameters()))}")
    
    # Manually create dummy loss the way the code does
    dummy = None
    for key in pipeline.aggregator.keys.values():
        if key.requires_grad:
            if dummy is None:
                dummy = key.sum() * 0.0
                print(f"Started dummy from key: requires_grad={dummy.requires_grad}, grad_fn={dummy.grad_fn}")
            else:
                dummy = dummy + key.sum() * 0.0
    
    if dummy is not None:
        print(f"Final dummy: requires_grad={dummy.requires_grad}, grad_fn={dummy.grad_fn}")
    else:
        print("❌ Could not create dummy loss - no trainable parameters!")
else:
    print("✅ Loss has grad_fn!")
    try:
        optimizer = torch.optim.Adam(pipeline.trainable_params(), lr=0.001)
        optimizer.zero_grad()
        total_loss.backward()
        print("✅ Backward successful!")
    except Exception as e:
        print(f"❌ Backward failed: {e}")

print("\n=== Done ===")
