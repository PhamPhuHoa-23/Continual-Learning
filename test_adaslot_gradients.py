"""Test if gradients are updating during AdaSlot training"""
import sys
import torch
import torch.nn.functional as F

# Import directly to avoid avalanche dependency
sys.path.insert(0, 'src')
from models.adaslot.model import AdaSlotModel
from data.continual_cifar100 import get_continual_cifar100_loaders

print("="*60)
print("Testing AdaSlot Gradient Flow")
print("="*60)

# Load data
train_loaders, _, _ = get_continual_cifar100_loaders(
    n_tasks=10, batch_size=8, num_workers=0, max_samples_per_task=50
)

# Create model
model = AdaSlotModel(num_slots=7, slot_dim=64)
model.cuda()

# Get one batch
for X, _ in train_loaders[0]:
    X = X.cuda()
    
    # Resize to 128x128
    X = F.interpolate(X, size=(128, 128), mode="bilinear", align_corners=False)
    
    print(f"\n📊 Input:")
    print(f"  Shape: {X.shape}")
    print(f"  Range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Forward
    print(f"\n🔄 Forward pass...")
    out = model(X, global_step=0)
    
    print(f"\n📤 Output:")
    print(f"  Reconstruction shape: {out['reconstruction'].shape}")
    print(f"  Reconstruction range: [{out['reconstruction'].min():.3f}, {out['reconstruction'].max():.3f}]")
    print(f"  Slots kept: {out['hard_keep_decision'].sum().item():.0f} / {out['hard_keep_decision'].shape[1]}")
    
    # Compute loss
    loss_recon = F.mse_loss(out["reconstruction"], X, reduction="sum") / X.size(0)
    loss_sparse = 10.0 * out["hard_keep_decision"].float().mean()
    loss = loss_recon + loss_sparse
    
    print(f"\n💰 Loss:")
    print(f"  L_recon: {loss_recon.item():.2f}")
    print(f"  L_sparse: {loss_sparse.item():.2f}")
    print(f"  Total: {loss.item():.2f}")
    
    # Check gradient
    print(f"\n🔍 Checking gradients...")
    loss.backward()
    
    total_grad_norm = 0.0
    params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            params_with_grad += 1
            if "encoder" in name or "decoder" in name:
                print(f"  {name[:50]:50s}: grad_norm={grad_norm:.6f}")
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"\n📈 Gradient Statistics:")
    print(f"  Params with gradients: {params_with_grad}")
    print(f"  Total grad norm: {total_grad_norm:.4f}")
    
    if total_grad_norm < 1e-6:
        print(f"\n❌ PROBLEM: Gradients are ZERO or near-zero!")
        print(f"  → Model is NOT learning!")
    elif total_grad_norm > 1000:
        print(f"\n⚠️  WARNING: Gradients are VERY large (exploding)")
        print(f"  → May need gradient clipping or lower LR")
    else:
        print(f"\n✅ Gradients look healthy!")
    
    break

print("\n" + "="*60)
