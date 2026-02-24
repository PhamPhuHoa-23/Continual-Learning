"""Quick debug: Check data range and loss calculation"""
import sys
import torch
import torch.nn.functional as F

# Import directly to avoid avalanche dependency
sys.path.insert(0, 'src')
from data.continual_cifar100 import get_continual_cifar100_loaders

print("Loading data...")
train_loaders, _, _ = get_continual_cifar100_loaders(
    n_tasks=10, batch_size=32, num_workers=0, max_samples_per_task=50
)

print("\nChecking first batch of Task 1:")
for X, y in train_loaders[0]:
    print(f"Batch shape: {X.shape}")
    print(f"Data range: [{X.min().item():.4f}, {X.max().item():.4f}]")
    print(f"Data mean: {X.mean().item():.4f}")
    print(f"Data std: {X.std().item():.4f}")
    
    # Simulate random reconstruction
    recon_random = torch.randn_like(X) * 0.5  # Small random values
    loss_random = F.mse_loss(recon_random, X, reduction='sum') / X.size(0)
    print(f"\nLoss with random reconstruction: {loss_random.item():.2f}")
    
    # Simulate zero reconstruction
    recon_zero = torch.zeros_like(X)
    loss_zero = F.mse_loss(recon_zero, X, reduction='sum') / X.size(0)
    print(f"Loss with zero reconstruction: {loss_zero.item():.2f}")
    
    break

print("\n✅ If data range is ~[-2, 2]: Data IS normalized")
print("✅ Expected loss with random init: 10,000-50,000")
print("✅ Expected loss after convergence: 2,000-5,000")
