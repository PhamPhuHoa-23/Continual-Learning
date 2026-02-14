"""
CIFAR-100 Continual Learning using Avalanche's SplitCIFAR100

This module provides a wrapper around Avalanche's built-in SplitCIFAR100 benchmark
for class-incremental learning.

Advantages of using Avalanche:
-------------------------------
1. Well-tested and maintained
2. Consistent with continual learning literature
3. Easy integration with Avalanche strategies
4. Built-in evaluation metrics

Usage:
------
    from src.data import get_avalanche_cifar100_benchmark
    
    # Get Avalanche benchmark
    benchmark = get_avalanche_cifar100_benchmark(
        n_experiences=5,
        seed=42
    )
    
    # Access train and test streams
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    
    # Training loop
    for task_id, train_exp in enumerate(train_stream):
        # Train on this experience
        for images, labels, task_labels in train_exp.dataset:
            ...
        
        # Evaluate on all experiences so far
        for test_exp in test_stream[:task_id+1]:
            for images, labels, task_labels in test_exp.dataset:
                ...

Author: Your Team
Date: 2026-02-12
"""

from typing import Optional
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks import CLScenario
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_avalanche_cifar100_benchmark(
    n_experiences: int = 5,
    seed: Optional[int] = 42,
    train_transform=None,
    eval_transform=None,
    dataset_root: str = "./data"
) -> CLScenario:
    """
    Create CIFAR-100 continual learning benchmark using Avalanche.
    
    This function wraps Avalanche's SplitCIFAR100 benchmark, which implements
    class-incremental learning where:
    1. Each experience contains NEW classes
    2. At test time, model must classify among ALL seen classes
    3. No task identity is provided at test time
    
    Args:
        n_experiences: Number of sequential experiences/tasks (must divide 100)
            - 2: 50 classes per experience
            - 5: 20 classes per experience (recommended)
            - 10: 10 classes per experience
            - 20: 5 classes per experience
        seed: Random seed for reproducibility
        train_transform: Custom transform for training data (None = Avalanche default)
        eval_transform: Custom transform for evaluation data (None = Avalanche default)
        dataset_root: Root directory for CIFAR-100 data
    
    Returns:
        CLScenario: Avalanche benchmark object with:
            - benchmark.train_stream: Training experiences
            - benchmark.test_stream: Test experiences
            - benchmark.classes_in_experience: Classes per experience
            - benchmark.n_experiences: Number of experiences
    
    Example:
        >>> benchmark = get_avalanche_cifar100_benchmark(n_experiences=5, seed=42)
        >>> 
        >>> # Training loop
        >>> for exp_id, train_exp in enumerate(benchmark.train_stream):
        ...     print(f"Training on experience {exp_id}")
        ...     print(f"Classes: {train_exp.classes_in_this_experience}")
        ...     
        ...     # Get dataset
        ...     train_dataset = train_exp.dataset
        ...     
        ...     # Iterate
        ...     for x, y, t in train_dataset:
        ...         # x: image tensor
        ...         # y: class label
        ...         # t: task label (not used in class-incremental)
        ...         pass
        >>> 
        >>> # Evaluation on all seen experiences
        >>> for exp_id in range(len(benchmark.test_stream)):
        ...     test_exp = benchmark.test_stream[exp_id]
        ...     print(f"Test experience {exp_id}")
        ...     print(f"Classes: {test_exp.classes_in_this_experience}")
    
    Notes:
        - Avalanche automatically handles:
            - Data splitting
            - Reproducibility with seed
            - Default transforms (RandomCrop, RandomHorizontalFlip for train)
            - Proper test set (cumulative classes)
        
        - Each experience object has:
            - .dataset: PyTorch dataset
            - .classes_in_this_experience: List of classes
            - .task_label: Task ID
        
        - For class-incremental learning, set return_task_id=False
          (model should not use task labels)
    
    References:
        - Avalanche Docs: https://avalanche.continualai.org/
        - SplitCIFAR100: https://avalanche-api.continualai.org/en/latest/benchmarks.html#split-cifar-100
    """
    logger.info(f"Creating Avalanche CIFAR-100 benchmark:")
    logger.info(f"  - Number of experiences: {n_experiences}")
    logger.info(f"  - Classes per experience: {100 // n_experiences}")
    logger.info(f"  - Seed: {seed}")
    logger.info(f"  - Dataset root: {dataset_root}")
    
    # Create SplitCIFAR100 benchmark
    benchmark = SplitCIFAR100(
        n_experiences=n_experiences,
        return_task_id=False,  # Class-incremental: no task ID at test time
        seed=seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
        dataset_root=dataset_root
    )
    
    # Log information
    logger.info(f"\n✅ Created Avalanche benchmark with {n_experiences} experiences")
    logger.info(f"   - Train stream: {len(benchmark.train_stream)} experiences")
    logger.info(f"   - Test stream: {len(benchmark.test_stream)} experiences")
    
    # Show classes per experience
    for exp_id in range(min(3, n_experiences)):  # Show first 3
        train_exp = benchmark.train_stream[exp_id]
        classes = train_exp.classes_in_this_experience
        logger.info(f"   - Experience {exp_id}: {len(classes)} classes {classes[:5]}...")
    
    if n_experiences > 3:
        logger.info(f"   ... ({n_experiences - 3} more experiences)")
    
    return benchmark


def get_avalanche_loaders_from_benchmark(
    benchmark: CLScenario,
    batch_size: int = 128,
    num_workers: int = 4
):
    """
    Create DataLoaders from Avalanche benchmark.
    
    This is a convenience function to convert Avalanche experiences
    into PyTorch DataLoaders.
    
    Args:
        benchmark: Avalanche benchmark
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loaders, test_loaders)
    
    Example:
        >>> benchmark = get_avalanche_cifar100_benchmark(n_experiences=5)
        >>> train_loaders, test_loaders = get_avalanche_loaders_from_benchmark(
        ...     benchmark, batch_size=128
        ... )
        >>> 
        >>> # Use like normal DataLoaders
        >>> for images, labels, task_labels in train_loaders[0]:
        ...     # Training code
        ...     pass
    """
    import torch.utils.data as data
    
    train_loaders = []
    test_loaders = []
    
    # Create train loaders
    for exp in benchmark.train_stream:
        loader = data.DataLoader(
            exp.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        train_loaders.append(loader)
    
    # Create test loaders
    for exp in benchmark.test_stream:
        loader = data.DataLoader(
            exp.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loaders.append(loader)
    
    logger.info(f"Created DataLoaders:")
    logger.info(f"  - {len(train_loaders)} training loaders")
    logger.info(f"  - {len(test_loaders)} test loaders")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Num workers: {num_workers}")
    
    return train_loaders, test_loaders


if __name__ == "__main__":
    """Demo usage of Avalanche CIFAR-100 benchmark."""
    from torchvision import transforms
    
    print("="*70)
    print("Avalanche CIFAR-100 Benchmark Demo")
    print("="*70)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    # Create benchmark
    print("\n[1] Creating benchmark with 5 experiences...")
    benchmark = get_avalanche_cifar100_benchmark(
        n_experiences=5,
        seed=42,
        train_transform=transform,
        eval_transform=transform
    )
    
    # Inspect first experience
    print("\n[2] Inspecting first training experience...")
    train_exp = benchmark.train_stream[0]
    
    print(f"   - Experience ID: {train_exp.current_experience}")
    print(f"   - Classes in this experience: {train_exp.classes_in_this_experience}")
    print(f"   - Dataset size: {len(train_exp.dataset)}")
    
    # Get one sample (note: x is PIL Image if no transform applied)
    print(f"   - Note: Dataset returns (image, label, task_label) tuples")
    print(f"   - Images are PIL Images (apply transforms for tensors)")
    
    # Inspect test stream
    print("\n[3] Inspecting test stream...")
    for exp_id in range(5):
        test_exp = benchmark.test_stream[exp_id]
        print(f"   - Test exp {exp_id}: {len(test_exp.dataset)} samples, "
              f"{len(test_exp.classes_in_this_experience)} classes")
    
    # Create DataLoaders
    print("\n[4] Creating DataLoaders...")
    train_loaders, test_loaders = get_avalanche_loaders_from_benchmark(
        benchmark,
        batch_size=128,
        num_workers=0  # Use 0 for demo
    )
    
    # Test iteration
    print("\n[5] Testing DataLoader iteration...")
    import torch
    for batch_idx, batch in enumerate(train_loaders[0]):
        if batch_idx == 0:
            # Avalanche returns different format depending on setup
            if len(batch) == 3:
                images, labels, task_labels = batch
                print(f"   - Batch shape: {images.shape}")
                print(f"   - Labels shape: {labels.shape}")
                if isinstance(labels, torch.Tensor):
                    print(f"   - Unique labels: {labels.unique().tolist()[:10]}...")
            else:
                print(f"   - Batch has {len(batch)} elements")
            break
    
    print("\n" + "="*70)
    print("[SUCCESS] Demo completed successfully!")
    print("="*70)
    print("\nKey differences from custom implementation:")
    print("  1. Avalanche returns (image, label, task_label) tuples")
    print("  2. Use benchmark.train_stream and benchmark.test_stream")
    print("  3. Each experience has .classes_in_this_experience attribute")
    print("  4. More features like metrics, strategies, etc.")
    print("\nSee Avalanche docs: https://avalanche.continualai.org/")

