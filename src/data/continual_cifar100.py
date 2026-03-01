"""
CIFAR-100 Continual Learning Data Pipeline

This module implements a class-incremental learning data pipeline for CIFAR-100.

Class-Incremental Learning Setting:
------------------------------------
In class-incremental learning, the model learns new classes sequentially without
access to previous task data. The model must:
1. Learn to classify new classes in each task
2. Retain ability to classify old classes (catastrophic forgetting challenge)
3. NOT have access to task identity at test time

Example:
    Task 0: Classes [0-19]   -> Train on 20 classes
    Task 1: Classes [20-39]  -> Train on 20 NEW classes, test on ALL 40 classes
    Task 2: Classes [40-59]  -> Train on 20 NEW classes, test on ALL 60 classes
    ...

Key Features:
-------------
- Configurable number of tasks (2, 5, 10, 20)
- Data augmentation for training
- Proper train/test split per task
- Support for memory replay (optional)
- Reproducible with seed

Usage:
------
    from src.data import get_continual_cifar100_loaders
    
    # Get data loaders for 5 tasks (20 classes per task)
    train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
        n_tasks=5,
        batch_size=128,
        num_workers=4,
        seed=42
    )
    
    # Training loop
    for task_id in range(5):
        train_loader = train_loaders[task_id]
        test_loader = test_loaders[task_id]  # Tests on ALL classes seen so far
        
        # Train on current task
        for images, labels in train_loader:
            # labels are in range [task_id*20, (task_id+1)*20)
            ...

Author: Your Team
Date: 2026-02-12
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassIncrementalSplit:
    """
    Split CIFAR-100 classes into sequential tasks for class-incremental learning.
    
    This class handles the splitting logic to ensure:
    1. Each task has a disjoint set of classes
    2. Classes are balanced across tasks
    3. Reproducible with seed
    
    Attributes:
        n_classes (int): Total number of classes (100 for CIFAR-100)
        n_tasks (int): Number of sequential tasks
        classes_per_task (int): Number of classes per task
        class_order (np.ndarray): Permutation of classes across tasks
        task_classes (List[List[int]]): Classes for each task
    
    Example:
        >>> splitter = ClassIncrementalSplit(n_tasks=5, seed=42)
        >>> splitter.get_task_classes(task_id=0)
        [83, 71, 45, 12, ...]  # 20 classes for task 0
        >>> splitter.get_seen_classes_up_to_task(task_id=2)
        [83, 71, ..., 23, 15, ...]  # 60 classes (tasks 0, 1, 2)
    """
    
    def __init__(
        self,
        n_tasks: int = 5,
        n_classes: int = 100,
        seed: Optional[int] = 42
    ):
        """
        Initialize class splitting.
        
        Args:
            n_tasks: Number of sequential tasks (2, 5, 10, 20, 50, or 100)
            n_classes: Total number of classes (default: 100 for CIFAR-100)
            seed: Random seed for reproducibility
        
        Raises:
            ValueError: If n_classes is not divisible by n_tasks
        """
        if n_classes % n_tasks != 0:
            raise ValueError(
                f"Number of classes ({n_classes}) must be divisible by "
                f"number of tasks ({n_tasks})"
            )
        
        self.n_classes = n_classes
        self.n_tasks = n_tasks
        self.classes_per_task = n_classes // n_tasks
        self.seed = seed
        
        # Generate class order (permutation of classes)
        if seed is not None:
            np.random.seed(seed)
        self.class_order = np.random.permutation(n_classes)
        
        # Split into tasks
        self.task_classes = [
            self.class_order[
                i * self.classes_per_task:(i + 1) * self.classes_per_task
            ].tolist()
            for i in range(n_tasks)
        ]
        
        logger.info(f"Created class-incremental split:")
        logger.info(f"  - Total classes: {n_classes}")
        logger.info(f"  - Number of tasks: {n_tasks}")
        logger.info(f"  - Classes per task: {self.classes_per_task}")
        logger.info(f"  - Random seed: {seed}")
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """
        Get classes for a specific task.
        
        Args:
            task_id: Task ID (0-indexed)
        
        Returns:
            List of class indices for this task
        
        Raises:
            ValueError: If task_id is out of range
        """
        if task_id < 0 or task_id >= self.n_tasks:
            raise ValueError(
                f"task_id must be in [0, {self.n_tasks-1}], got {task_id}"
            )
        return self.task_classes[task_id]
    
    def get_seen_classes_up_to_task(self, task_id: int) -> List[int]:
        """
        Get all classes seen from task 0 to task_id (inclusive).
        
        This is used for evaluation - at test time after training task_id,
        the model should be able to classify all classes from tasks 0 to task_id.
        
        Args:
            task_id: Current task ID (0-indexed)
        
        Returns:
            List of all class indices seen so far
        """
        if task_id < 0 or task_id >= self.n_tasks:
            raise ValueError(
                f"task_id must be in [0, {self.n_tasks-1}], got {task_id}"
            )
        
        seen_classes = []
        for i in range(task_id + 1):
            seen_classes.extend(self.task_classes[i])
        return seen_classes
    
    def get_task_id_for_class(self, class_id: int) -> int:
        """
        Get which task a class belongs to.
        
        Args:
            class_id: Class index (0-99 for CIFAR-100)
        
        Returns:
            Task ID that this class belongs to
        """
        for task_id, classes in enumerate(self.task_classes):
            if class_id in classes:
                return task_id
        raise ValueError(f"Class {class_id} not found in any task")
    
    def __repr__(self) -> str:
        return (
            f"ClassIncrementalSplit(n_tasks={self.n_tasks}, "
            f"classes_per_task={self.classes_per_task}, seed={self.seed})"
        )


class ContinualCIFAR100Dataset(Dataset):
    """
    Wrapper around CIFAR-100 for a specific task in continual learning.
    
    This dataset filters CIFAR-100 to only include samples from specific classes
    (i.e., classes in the current task).
    
    Attributes:
        base_dataset: Original CIFAR-100 dataset
        task_classes: List of classes for this task
        indices: Indices of samples belonging to task_classes
        class_mapping: Mapping from original labels to task-specific labels
    
    Example:
        >>> # Task 0 with classes [83, 71, 45, ...]
        >>> dataset = ContinualCIFAR100Dataset(
        ...     root='./data',
        ...     train=True,
        ...     task_classes=[83, 71, 45],
        ...     transform=transform
        ... )
        >>> image, label = dataset[0]
        >>> # label will be in [83, 71, 45]
    """
    
    def __init__(
        self,
        root: str,
        train: bool,
        task_classes: List[int],
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize task-specific CIFAR-100 dataset.
        
        Args:
            root: Root directory for data storage
            train: If True, use training set; else use test set
            task_classes: List of class indices for this task
            transform: Torchvision transforms to apply
            download: If True, download CIFAR-100 if not present
            max_samples: If set, limit to this many samples (for fast testing)
        """
        self.base_dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        
        self.task_classes = task_classes
        self.task_classes_set = set(task_classes)
        
        # Find indices of samples belonging to task_classes
        # If max_samples is set, limit to max_samples PER CLASS (not total)
        self.indices = []
        
        if max_samples is not None:
            # Track samples per class to ensure balanced sampling
            class_counts = {cls: 0 for cls in task_classes}
            max_per_class = max_samples
            
            for idx, (_, label) in enumerate(self.base_dataset):
                if label in self.task_classes_set:
                    if class_counts[label] < max_per_class:
                        self.indices.append(idx)
                        class_counts[label] += 1
                    
                    # Stop when all classes have max_samples
                    if all(cnt >= max_per_class for cnt in class_counts.values()):
                        break
        else:
            # No limit - collect all samples
            for idx, (_, label) in enumerate(self.base_dataset):
                if label in self.task_classes_set:
                    self.indices.append(idx)
        
        # We keep original labels (not remapped)
        # This is important for class-incremental learning
        # where labels should be consistent across tasks
        
        logger.info(
            f"Created {'train' if train else 'test'} dataset for "
            f"{len(task_classes)} classes with {len(self.indices)} samples"
            f"{f' ({max_samples} per class)' if max_samples else ''}"
        )
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Index
        
        Returns:
            (image, label) where label is the original CIFAR-100 class index
        """
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        return image, label


def get_cifar100_transforms(train: bool = True) -> transforms.Compose:
    """
    Get standard data augmentation transforms for CIFAR-100.
    
    Training transforms:
        - RandomCrop(32, padding=4)
        - RandomHorizontalFlip
        - ToTensor
        - Normalize (mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    
    Test transforms:
        - ToTensor
        - Normalize (same as train)
    
    Args:
        train: If True, return training transforms; else test transforms
    
    Returns:
        Composition of transforms
    """
    # CIFAR-100 normalization statistics
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_continual_cifar100_loaders(
    n_tasks: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
    root: str = './data',
    seed: Optional[int] = 42,
    pin_memory: bool = True,
    max_samples_per_task: Optional[int] = None
) -> Tuple[List[DataLoader], List[DataLoader], np.ndarray]:
    """
    Create data loaders for class-incremental learning on CIFAR-100.
    
    This function creates:
    1. Training loaders: One per task, containing only NEW classes for that task
    2. Test loaders: One per task, containing ALL classes seen up to that task
    
    Args:
        n_tasks: Number of sequential tasks (must divide 100)
            - 2 tasks: 50 classes per task
            - 5 tasks: 20 classes per task (recommended)
            - 10 tasks: 10 classes per task
            - 20 tasks: 5 classes per task
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        root: Root directory for data storage
        seed: Random seed for reproducibility
        pin_memory: If True, pin memory for faster GPU transfer
        max_samples_per_task: If set, limit each task to this many samples (for fast testing)
    
    Returns:
        Tuple of (train_loaders, test_loaders, class_order):
            - train_loaders: List of training DataLoaders (one per task)
            - test_loaders: List of test DataLoaders (one per task)
            - class_order: Class order permutation used
    
    Example:
        >>> train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
        ...     n_tasks=5,
        ...     batch_size=128,
        ...     seed=42
        ... )
        >>> 
        >>> # Training on Task 0
        >>> for images, labels in train_loaders[0]:
        ...     # labels are 20 NEW classes for task 0
        ...     ...
        >>> 
        >>> # Testing after Task 0 - should classify 20 classes
        >>> for images, labels in test_loaders[0]:
        ...     # labels are same 20 classes as training
        ...     ...
        >>> 
        >>> # Training on Task 1
        >>> for images, labels in train_loaders[1]:
        ...     # labels are 20 NEW classes for task 1
        ...     ...
        >>> 
        >>> # Testing after Task 1 - should classify 40 classes (tasks 0+1)
        >>> for images, labels in test_loaders[1]:
        ...     # labels are ALL 40 classes seen so far
        ...     ...
    
    Notes:
        - Train loaders use data augmentation (RandomCrop, RandomHorizontalFlip)
        - Test loaders use no augmentation
        - Labels are NOT remapped - they keep original CIFAR-100 class indices
        - At test time, model should output probabilities over all 100 classes
          (or dynamically grow the classifier head)
    """
    # Create class split
    splitter = ClassIncrementalSplit(n_tasks=n_tasks, n_classes=100, seed=seed)
    
    # Get transforms
    train_transform = get_cifar100_transforms(train=True)
    test_transform = get_cifar100_transforms(train=False)
    
    train_loaders = []
    test_loaders = []
    
    # Create loaders for each task
    for task_id in range(n_tasks):
        # Get classes for this task
        task_classes = splitter.get_task_classes(task_id)
        seen_classes = splitter.get_seen_classes_up_to_task(task_id)
        
        logger.info(f"\n=== Task {task_id} ===")
        logger.info(f"Training classes: {task_classes[:5]}... ({len(task_classes)} total)")
        logger.info(f"Test classes: {seen_classes[:5]}... ({len(seen_classes)} total)")
        
        # Training dataset: Only NEW classes for this task
        train_dataset = ContinualCIFAR100Dataset(
            root=root,
            train=True,
            task_classes=task_classes,
            transform=train_transform,
            download=True,
            max_samples=max_samples_per_task
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        train_loaders.append(train_loader)
        
        # Test dataset: ALL classes seen so far
        test_dataset = ContinualCIFAR100Dataset(
            root=root,
            train=False,
            task_classes=seen_classes,
            transform=test_transform,
            download=True,
            max_samples=max_samples_per_task
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loaders.append(test_loader)
    
    logger.info(f"\n✅ Created data loaders for {n_tasks} tasks")
    logger.info(f"   - Classes per task: {100 // n_tasks}")
    logger.info(f"   - Batch size: {batch_size}")
    logger.info(f"   - Seed: {seed}")
    
    return train_loaders, test_loaders, splitter.class_order


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("CIFAR-100 Continual Learning Data Pipeline Demo")
    print("="*60)
    
    # Create loaders for 5 tasks
    train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
        n_tasks=5,
        batch_size=128,
        num_workers=0,  # Use 0 for demo to avoid multiprocessing issues
        seed=42
    )
    
    print("\n" + "="*60)
    print("Inspecting Task 0:")
    print("="*60)
    
    # Check first task
    train_loader = train_loaders[0]
    test_loader = test_loaders[0]
    
    # Get one batch from training
    images, labels = next(iter(train_loader))
    print(f"\nTraining batch:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Unique labels: {torch.unique(labels).tolist()}")
    print(f"  - Number of unique classes: {len(torch.unique(labels))}")
    
    # Get one batch from test
    images, labels = next(iter(test_loader))
    print(f"\nTest batch (after task 0):")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Unique labels in batch: {torch.unique(labels).tolist()[:10]}...")
    
    print("\n" + "="*60)
    print("Inspecting Task 1:")
    print("="*60)
    
    # Check second task
    train_loader = train_loaders[1]
    test_loader = test_loaders[1]
    
    # Training on task 1
    images, labels = next(iter(train_loader))
    print(f"\nTraining batch (task 1 - NEW classes):")
    print(f"  - Unique labels: {torch.unique(labels).tolist()}")
    
    # Test after task 1 (should have classes from task 0 AND task 1)
    images, labels = next(iter(test_loader))
    unique_test_labels = []
    for images, labels in test_loader:
        unique_test_labels.extend(labels.tolist())
    unique_test_labels = sorted(list(set(unique_test_labels)))
    
    print(f"\nTest set (after task 1 - should have 40 classes):")
    print(f"  - Number of unique classes: {len(unique_test_labels)}")
    print(f"  - Classes: {unique_test_labels[:10]}... (showing first 10)")
    
    print("\n" + "="*60)
    print("✅ Demo completed successfully!")
    print("="*60)

