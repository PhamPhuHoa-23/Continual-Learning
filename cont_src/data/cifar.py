"""
CIFAR-10 and CIFAR-100 Continual Learning Datasets

Provides class-incremental learning datasets for CIFAR-10 and CIFAR-100
using Avalanche framework.
"""

from typing import Optional
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100
from avalanche.benchmarks import CLScenario

from cont_src.core.registry import DATASET_REGISTRY
from cont_src.data.base_dataset import AvalancheDatasetWrapper, BaseContinualDataset


@DATASET_REGISTRY.register("cifar10")
class CIFAR10ContinualDataset(AvalancheDatasetWrapper):
    """
    CIFAR-10 continual learning dataset.

    Class-incremental learning on CIFAR-10:
    - 10 classes total
    - Typical splits: 2, 5, or 10 tasks
    - Uses Avalanche's SplitCIFAR10

    Example:
        >>> from cont_src.core.registry import DATASET_REGISTRY
        >>> dataset = DATASET_REGISTRY.build(
        ...     "cifar10",
        ...     n_tasks=5,
        ...     batch_size=64
        ... )
        >>> 
        >>> # Get first task
        >>> task_0 = dataset.get_task_data(0)
        >>> print(f"Classes in task 0: {task_0.classes}")
        >>> 
        >>> # Training
        >>> for images, labels in task_0.train_loader:
        ...     # Train on 2 classes
        ...     pass
    """

    def __init__(
        self,
        n_tasks: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: Optional[int] = 42,
        data_root: str = "./data",
        return_task_id: bool = False,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Initialize CIFAR-10 continual learning dataset.

        Args:
            n_tasks: Number of tasks (2, 5, or 10)
            batch_size: Batch size
            num_workers: Number of data loading workers
            seed: Random seed for class order
            data_root: Root directory for CIFAR-10 data
            return_task_id: If True, return (image, label, task_id)
            shuffle: Shuffle classes order
        """
        if 10 % n_tasks != 0:
            raise ValueError(f"n_tasks={n_tasks} must divide 10 evenly")

        # Create Avalanche benchmark
        benchmark = SplitCIFAR10(
            n_experiences=n_tasks,
            seed=seed,
            return_task_id=return_task_id,
            shuffle=shuffle,
            dataset_root=data_root,
        )

        # Initialize wrapper
        super().__init__(
            benchmark=benchmark,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )


@DATASET_REGISTRY.register("cifar100")
class CIFAR100ContinualDataset(AvalancheDatasetWrapper):
    """
    CIFAR-100 continual learning dataset.

    Class-incremental learning on CIFAR-100:
    - 100 classes total
    - Typical splits: 5, 10, or 20 tasks
    - Uses Avalanche's SplitCIFAR100

    Task Setup Examples:
    - n_tasks=5:  20 classes per task
    - n_tasks=10: 10 classes per task
    - n_tasks=20: 5 classes per task

    Example:
        >>> from cont_src.core.registry import DATASET_REGISTRY
        >>> dataset = DATASET_REGISTRY.build(
        ...     "cifar100",
        ...     n_tasks=10,
        ...     batch_size=64
        ... )
        >>> 
        >>> # Iterate over all tasks
        >>> for task_id in range(len(dataset)):
        ...     task = dataset[task_id]
        ...     print(f"Task {task_id}: {len(task.classes)} classes")
        ...     
        ...     for images, labels in task.train_loader:
        ...         # Training code
        ...         pass
        ...     
        ...     # Evaluate on all classes seen so far
        ...     for images, labels in task.test_loader:
        ...         # Test on classes [0, task_id*classes_per_task]
        ...         pass
    """

    def __init__(
        self,
        n_tasks: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: Optional[int] = 42,
        data_root: str = "./data",
        return_task_id: bool = False,
        shuffle: bool = True,
        fixed_class_order: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize CIFAR-100 continual learning dataset.

        Args:
            n_tasks: Number of tasks (2, 5, 10, 20, 25, 50, or 100)
            batch_size: Batch size
            num_workers: Number of data loading workers
            seed: Random seed for class order (ignored if fixed_class_order is set)
            data_root: Root directory for CIFAR-100 data
            return_task_id: If True, return (image, label, task_id)
            shuffle: Shuffle classes order (ignored if fixed_class_order is set)
            fixed_class_order: Fixed order of classes (overrides seed and shuffle)
        """
        if 100 % n_tasks != 0:
            raise ValueError(f"n_tasks={n_tasks} must divide 100 evenly")

        # Create Avalanche benchmark
        benchmark = SplitCIFAR100(
            n_experiences=n_tasks,
            seed=seed,
            return_task_id=return_task_id,
            shuffle=shuffle,
            fixed_class_order=fixed_class_order,
            dataset_root=data_root,
        )

        # Initialize wrapper
        super().__init__(
            benchmark=benchmark,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )


# Convenience functions for backward compatibility
def get_cifar10_continual(
    n_tasks: int = 5,
    batch_size: int = 64,
    **kwargs
) -> CIFAR10ContinualDataset:
    """
    Get CIFAR-10 continual learning dataset.

    Args:
        n_tasks: Number of tasks
        batch_size: Batch size
        **kwargs: Additional arguments

    Returns:
        CIFAR10ContinualDataset instance
    """
    return CIFAR10ContinualDataset(
        n_tasks=n_tasks,
        batch_size=batch_size,
        **kwargs
    )


def get_cifar100_continual(
    n_tasks: int = 10,
    batch_size: int = 64,
    **kwargs
) -> CIFAR100ContinualDataset:
    """
    Get CIFAR-100 continual learning dataset.

    Args:
        n_tasks: Number of tasks
        batch_size: Batch size
        **kwargs: Additional arguments

    Returns:
        CIFAR100ContinualDataset instance
    """
    return CIFAR100ContinualDataset(
        n_tasks=n_tasks,
        batch_size=batch_size,
        **kwargs
    )
