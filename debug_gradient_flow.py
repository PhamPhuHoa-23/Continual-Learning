"""
Debug script to check gradient flow in the pipeline
"""
import torch
import torch.nn.functional as F
from src.compositional.pipeline import CompositionalRoutingPipeline

# Create a simple pipeline
pipeline = CompositionalRoutingPipeline(
    slot_dim=64,
    d_h=128,
    latent_dim=32,
    device='cpu'
)

# Spawn a few agents manually
for i in range(3):
    aid = f"agent_{i}"
    pipeline._spawn_agent(aid)
    pipeline.new_agent_ids.append(aid)

print(f"Created {len(pipeline.agents)} agents")
print(f"Created {len(pipeline.aggregator.keys)} keys")

# Check if agents are trainable
print("\n=== Agent Parameters ===")
for aid, agent in pipeline.agents.items():
    n_params = sum(p.numel() for p in agent.parameters())
    n_trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"{aid}: {n_trainable}/{n_params} trainable params, training={agent.training}")

# Check if keys are trainable
print("\n=== Aggregator Keys ===")
for aid, key in pipeline.aggregator.keys.items():
    print(f"{aid}: requires_grad={key.requires_grad}, shape={key.shape}")

# Get trainable params
trainable_params = pipeline.trainable_params()
print(f"\n=== Total Trainable Parameters: {len(trainable_params)} ===")

# Create dummy input
batch_size = 4
num_slots = 7
slots = torch.randn(batch_size, num_slots, 64)
labels = torch.randint(0, 10, (batch_size,))

print(f"\n=== Testing compute_losses ===")
print(f"Input: slots={slots.shape}, labels={labels.shape}")

# Set train mode
pipeline.train_mode()

# Compute losses
losses = pipeline.compute_losses(slots, labels, task_id=1)

print(f"\n=== Loss Results ===")
for k, v in losses.items():
    has_grad = v.requires_grad
    has_grad_fn = v.grad_fn is not None
    print(f"{k}: {v.item():.6f}, requires_grad={has_grad}, has_grad_fn={has_grad_fn}")

# Try backward
print(f"\n=== Testing Backward ===")
total_loss = losses["total"]
print(f"Loss value: {total_loss.item():.6f}")
print(f"Loss requires_grad: {total_loss.requires_grad}")
print(f"Loss grad_fn: {total_loss.grad_fn}")

if total_loss.grad_fn is None:
    print("\n❌ ERROR: Loss has no grad_fn!")
    print("This means the loss is not connected to any trainable parameters")
    
    # Check what went wrong
    print("\n=== Debugging ===")
    
    # Check if any slots were assigned
    slots_flat = slots.view(-1, 64)
    with torch.no_grad():
        sigma_flat = pipeline._route_slots(slots_flat)
    
    assigned_count = sum(1 for s in sigma_flat if s != "unassigned")
    print(f"Assigned slots: {assigned_count}/{len(sigma_flat)}")
    
    # Check if routing is working
    if assigned_count == 0:
        print("\n⚠️  NO SLOTS WERE ASSIGNED!")
        print("This is the problem - all slots are 'unassigned'")
        print("Let's check the KMeans cluster centers...")
        
        if pipeline.task1_cluster_centers is not None:
            print(f"Cluster centers exist: {len(pipeline.task1_cluster_centers)}")
            for aid, center in pipeline.task1_cluster_centers.items():
                print(f"  {aid}: center shape={center.shape}")
        else:
            print("No cluster centers found - routing will use VAEs (which aren't trained yet)")
    
else:
    print("\n✅ Loss has grad_fn, backward should work!")
    try:
        # Create optimizer
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)
        optimizer.zero_grad()
        total_loss.backward()
        print("✅ Backward pass successful!")
        
        # Check gradients
        n_grads = sum(1 for p in trainable_params if p.grad is not None)
        print(f"Parameters with gradients: {n_grads}/{len(trainable_params)}")
        
    except Exception as e:
        print(f"❌ Backward failed: {e}")

print("\n=== Debug Complete ===")
