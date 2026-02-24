"""Quick test: AdaSlot loss and gradient on CIFAR-100"""
import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, 'src')
from models.adaslot.model import AdaSlotModel

print("="*60)
print("Quick AdaSlot Test")
print("="*60)

# Load small batch of CIFAR-100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761])
])

dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

# Get one batch
X, y = next(iter(loader))
X = X.cuda()

print(f"\n📊 Input (CIFAR-100):")
print(f"  Shape: {X.shape}")
print(f"  Range: [{X.min():.3f}, {X.max():.3f}]")
print(f"  Mean: {X.mean():.3f}")

# Resize to 128x128 for AdaSlot
X_resized = F.interpolate(X, size=(128, 128), mode="bilinear", align_corners=False)
print(f"\n🔄 After resize to 128x128:")
print(f"  Shape: {X_resized.shape}")
print(f"  Range: [{X_resized.min():.3f}, {X_resized.max():.3f}]")

# Create AdaSlot model
print(f"\n🏗️  Creating AdaSlot (slots=7, dim=64)...")
model = AdaSlotModel(num_slots=7, slot_dim=64)
model.cuda()
model.train()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")

# Forward pass
print(f"\n🔄 Forward pass...")
out = model(X_resized, global_step=0)

print(f"\n📤 Output:")
print(f"  Reconstruction shape: {out['reconstruction'].shape}")
print(f"  Reconstruction range: [{out['reconstruction'].min():.3f}, {out['reconstruction'].max():.3f}]")
print(f"  Slots kept: {out['hard_keep_decision'].sum().item():.0f} / {out['hard_keep_decision'].shape[1]}")

# Compute loss
loss_recon = F.mse_loss(out["reconstruction"], X_resized, reduction="sum") / X_resized.size(0)
loss_sparse = 10.0 * out["hard_keep_decision"].float().mean()
loss = loss_recon + loss_sparse

print(f"\n💰 Loss (per sample):")
print(f"  L_recon: {loss_recon.item():.2f}")
print(f"  L_sparse: {loss_sparse.item():.2f}")
print(f"  Total: {loss.item():.2f}")

# Expected loss calculation
pixels_per_sample = 3 * 128 * 128
avg_mse_per_pixel = loss_recon.item() / pixels_per_sample
print(f"\n📐 Loss breakdown:")
print(f"  Total pixels: {pixels_per_sample:,}")
print(f"  MSE per pixel: {avg_mse_per_pixel:.6f}")

# Check gradients
print(f"\n🔍 Testing backward...")
loss.backward()

grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None and param.requires_grad:
        grad_norm = param.grad.norm().item()
        if "encoder" in name or "decoder" in name:
            grad_norms[name] = grad_norm

total_grad_norm = sum(v**2 for v in grad_norms.values()) ** 0.5

print(f"\n📈 Gradient norms (sample):")
for name, norm in list(grad_norms.items())[:5]:
    print(f"  {name[:60]:60s}: {norm:.6f}")

print(f"\n📊 Summary:")
print(f"  Total gradient norm: {total_grad_norm:.4f}")
print(f"  Params with gradients: {len(grad_norms)}")

if total_grad_norm < 1e-6:
    print(f"\n❌ PROBLEM: Gradients are ZERO!")
elif total_grad_norm > 10000:
    print(f"\n⚠️  WARNING: Gradients are very large (exploding)")
else:
    print(f"\n✅ Gradients look OK!")

print(f"\n🎯 Expected behavior:")
print(f"  • Random init loss: 10,000-50,000")
print(f"  • After training: 2,000-5,000")
print(f"  • Your loss {loss.item():.0f} is: ", end="")
if loss.item() > 30000:
    print("HIGH (expected for random init)")
elif loss.item() > 10000:
    print("REASONABLE (early training)")
elif loss.item() > 5000:
    print("OK (mid training)")
else:
    print("GOOD (converging)")

print("\n" + "="*60)
