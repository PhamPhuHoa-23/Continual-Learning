"""
Tiny-ImageNet Continual Learning using Avalanche

Tiny-ImageNet is a subset of ImageNet with:
- Image size: 64x64 (2x larger than CIFAR-100)
- Classes: 200
- Training: 100,000 images (500 per class)
- Test: 10,000 images

This is a good balance between CIFAR-100 (too small) and full ImageNet (too large).

Usage:
------
    from src.data import get_tinyimagenet_benchmark
    
    # Get benchmark with 10 experiences (20 classes each)
    benchmark = get_tinyimagenet_benchmark(
        n_experiences=10,
        seed=42
    )
    
    # Training loop
    for exp_id, train_exp in enumerate(benchmark.train_stream):
        for images, labels, task_labels in DataLoader(train_exp.dataset):
            # images: [batch, 3, 64, 64]  <- 2x larger than CIFAR
            ...

Author: Your Team
Date: 2026-02-12
"""

from typing import Optional
from avalanche.benchmarks.classic import SplitTinyImageNet
from avalanche.benchmarks import CLScenario
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tinyimagenet_benchmark(
    n_experiences: int = 10,
    seed: Optional[int] = 42,
    train_transform=None,
    eval_transform=None,
    dataset_root: str = "./data"
) -> CLScenario:
    """
    Create Tiny-ImageNet continual learning benchmark using Avalanche.
    
    Tiny-ImageNet advantages over CIFAR-100:
    - Larger images (64x64 vs 32x32)
    - More classes (200 vs 100)
    - More challenging
    
    Args:
        n_experiences: Number of sequential experiences (2, 4, 5, 10, 20, 40)
            - 2: 100 classes per experience
            - 5: 40 classes per experience
            - 10: 20 classes per experience (recommended)
            - 20: 10 classes per experience
        seed: Random seed for reproducibility
        train_transform: Custom transform for training data
        eval_transform: Custom transform for evaluation data
        dataset_root: Root directory for data storage
    
    Returns:
        CLScenario: Avalanche benchmark object
    
    Example:
        >>> benchmark = get_tinyimagenet_benchmark(n_experiences=10, seed=42)
        >>> 
        >>> # Training loop
        >>> for exp_id, train_exp in enumerate(benchmark.train_stream):
        ...     print(f"Experience {exp_id}")
        ...     print(f"Classes: {train_exp.classes_in_this_experience}")
        ...     print(f"Samples: {len(train_exp.dataset)}")
        ...     
        ...     for images, labels, task_labels in DataLoader(train_exp.dataset):
        ...         # images: [batch, 3, 64, 64]
        ...         pass
    
    Notes:
        - Dataset will be downloaded to {dataset_root}/tiny-imagenet-200/
        - Download size: ~250MB
        - Images are 64x64 RGB (2x larger than CIFAR-100)
        - Good for testing slot attention with larger images
    """
    logger.info(f"Creating Tiny-ImageNet benchmark:")
    logger.info(f"  - Number of experiences: {n_experiences}")
    logger.info(f"  - Classes per experience: {200 // n_experiences}")
    logger.info(f"  - Image size: 64x64 (2x larger than CIFAR-100)")
    logger.info(f"  - Seed: {seed}")
    logger.info(f"  - Dataset root: {dataset_root}")
    
    # Create SplitTinyImageNet benchmark
    benchmark = SplitTinyImageNet(
        n_experiences=n_experiences,
        return_task_id=False,  # Class-incremental learning
        seed=seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
        dataset_root=dataset_root
    )
    
    # Log information
    logger.info(f"\nCreated Tiny-ImageNet benchmark with {n_experiences} experiences")
    logger.info(f"   - Train stream: {len(benchmark.train_stream)} experiences")
    logger.info(f"   - Test stream: {len(benchmark.test_stream)} experiences")
    
    # Show classes per experience
    for exp_id in range(min(3, n_experiences)):
        train_exp = benchmark.train_stream[exp_id]
        classes = train_exp.classes_in_this_experience
        logger.info(f"   - Experience {exp_id}: {len(classes)} classes")
    
    if n_experiences > 3:
        logger.info(f"   ... ({n_experiences - 3} more experiences)")
    
    return benchmark


if __name__ == "__main__":
    """Demo usage of Tiny-ImageNet benchmark."""
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    print("="*70)
    print("Tiny-ImageNet Continual Learning Demo")
    print("="*70)
    
    # Create transforms for 64x64 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create benchmark
    print("\n[1] Creating benchmark with 10 experiences...")
    benchmark = get_tinyimagenet_benchmark(
        n_experiences=10,
        seed=42,
        train_transform=transform,
        eval_transform=transform
    )
    
    # Inspect first experience
    print("\n[2] Inspecting first training experience...")
    train_exp = benchmark.train_stream[0]
    
    print(f"   - Experience ID: {train_exp.current_experience}")
    print(f"   - Classes: {len(train_exp.classes_in_this_experience)} classes")
    print(f"   - Dataset size: {len(train_exp.dataset)} samples")
    
    # Create DataLoader and test
    print("\n[3] Testing DataLoader...")
    train_loader = DataLoader(
        train_exp.dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    for batch_idx, (images, labels, task_labels) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"   - Batch shape: {images.shape}")  # [64, 3, 64, 64]
            print(f"   - Labels shape: {labels.shape}")
            print(f"   - Unique labels: {labels.unique().tolist()[:5]}...")
            break
    
    print("\n[4] Comparing with CIFAR-100...")
    print(f"   CIFAR-100:      32x32, 100 classes")
    print(f"   Tiny-ImageNet:  64x64, 200 classes  <- 2x larger images!")
    
    print("\n" + "="*70)
    print("[SUCCESS] Tiny-ImageNet demo completed!")
    print("="*70)
    print("\nKey points:")
    print("  - Images are 64x64 (2x larger than CIFAR-100)")
    print("  - 200 classes (good for continual learning)")
    print("  - Good balance between CIFAR-100 and full ImageNet")
    print("  - Perfect for testing slot attention with larger images")
    print()
    print("Usage in your code:")
    print("""
    from src.data import get_tinyimagenet_benchmark
    
    benchmark = get_tinyimagenet_benchmark(n_experiences=10, seed=42)
    
    for exp_id, train_exp in enumerate(benchmark.train_stream):
        # Your training code
        ...
    """)

