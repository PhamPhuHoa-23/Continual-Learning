"""
MNIST Continual Learning Dataset

Provides class-incremental learning dataset for MNIST using Avalanche.
"""

from typing import Optional
from avalanche.benchmarks.classic import SplitMNIST

from cont_src.core.registry import DATASET_REGISTRY
from cont_src.data.base_dataset import AvalancheDatasetWrapper


@DATASET_REGISTRY.register("mnist")
class MNISTContinualDataset(AvalancheDatasetWrapper):
    """
    MNIST continual learning dataset.

    Class-incremental learning on MNIST:
    - 10 classes (digits 0-9)
    - Typical splits: 2, 5, or 10 tasks
    - Uses Avalanche's SplitMNIST

    Example:
        >>> from cont_src.core.registry import DATASET_REGISTRY
        >>> dataset = DATASET_REGISTRY.build(
        ...     "mnist",
        ...     n_tasks=5,
        ...     batch_size=128
        ... )
        >>> 
        >>> # Each task has 2 classes
        >>> task_0 = dataset.get_task_data(0)
        >>> print(f"Task 0 classes: {task_0.classes}")  # e.g., [3, 7]
        >>> 
        >>> for images, labels in task_0.train_loader:
        ...     # images: (B, 1, 28, 28)
        ...     # labels: class indices
        ...     pass
    """

    def __init__(
        self,
        n_tasks: int = 5,
        batch_size: int = 128,
        num_workers: int = 4,
        seed: Optional[int] = 42,
        data_root: str = "./data",
        return_task_id: bool = False,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Initialize MNIST continual learning dataset.

        Args:
            n_tasks: Number of tasks (2, 5, or 10)
            batch_size: Batch size
            num_workers: Number of data loading workers
            seed: Random seed for class order
            data_root: Root directory for MNIST data
            return_task_id: If True, return (image, label, task_id)
            shuffle: Shuffle classes order
        """
        if 10 % n_tasks != 0:
            raise ValueError(f"n_tasks={n_tasks} must divide 10 evenly")

        # Create Avalanche benchmark
        benchmark = SplitMNIST(
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


def get_mnist_continual(
    n_tasks: int = 5,
    batch_size: int = 128,
    **kwargs
) -> MNISTContinualDataset:
    """
    Get MNIST continual learning dataset.

    Args:
        n_tasks: Number of tasks
        batch_size: Batch size
        **kwargs: Additional arguments

    Returns:
        MNISTContinualDataset instance
    """
    return MNISTContinualDataset(
        n_tasks=n_tasks,
        batch_size=batch_size,
        **kwargs
    )
