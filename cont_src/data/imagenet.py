"""
ImageNet Continual Learning Datasets

Provides continual learning datasets for ImageNet variants.
Currently supports:
- Tiny-ImageNet (200 classes, 64x64)
"""

from typing import Optional
from avalanche.benchmarks.classic import SplitTinyImageNet

from cont_src.core.registry import DATASET_REGISTRY
from cont_src.data.base_dataset import AvalancheDatasetWrapper


@DATASET_REGISTRY.register("tiny_imagenet")
class TinyImageNetContinualDataset(AvalancheDatasetWrapper):
    """
    Tiny-ImageNet continual learning dataset.

    Class-incremental learning on Tiny-ImageNet:
    - 200 classes from ImageNet
    - 64x64 resolution images
    - 500 training images per class
    - Typical splits: 10, 20, 40 tasks
    - Uses Avalanche's SplitTinyImageNet

    Example:
        >>> from cont_src.core.registry import DATASET_REGISTRY
        >>> dataset = DATASET_REGISTRY.build(
        ...     "tiny_imagenet",
        ...     n_tasks=20,
        ...     batch_size=64
        ... )
        >>> 
        >>> # Each task has 10 classes
        >>> for task_id in range(20):
        ...     task = dataset[task_id]
        ...     print(f"Task {task_id}: {len(task.classes)} classes")
        ...     
        ...     for images, labels in task.train_loader:
        ...         # images: (B, 3, 64, 64)
        ...         # labels: class indices
        ...         pass
    """

    def __init__(
        self,
        n_tasks: int = 20,
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
        Initialize Tiny-ImageNet continual learning dataset.

        Args:
            n_tasks: Number of tasks (2, 4, 5, 8, 10, 20, 25, 40, 50, 100, or 200)
            batch_size: Batch size
            num_workers: Number of data loading workers
            seed: Random seed for class order
            data_root: Root directory for Tiny-ImageNet data
            return_task_id: If True, return (image, label, task_id)
            shuffle: Shuffle classes order
            fixed_class_order: Fixed order of classes
        """
        if 200 % n_tasks != 0:
            raise ValueError(f"n_tasks={n_tasks} must divide 200 evenly")

        # Create Avalanche benchmark
        benchmark = SplitTinyImageNet(
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


def get_tiny_imagenet_continual(
    n_tasks: int = 20,
    batch_size: int = 64,
    **kwargs
) -> TinyImageNetContinualDataset:
    """
    Get Tiny-ImageNet continual learning dataset.

    Args:
        n_tasks: Number of tasks
        batch_size: Batch size
        **kwargs: Additional arguments

    Returns:
        TinyImageNetContinualDataset instance
    """
    return TinyImageNetContinualDataset(
        n_tasks=n_tasks,
        batch_size=batch_size,
        **kwargs
    )


# Placeholder for future ImageNet support
@DATASET_REGISTRY.register("imagenet")
class ImageNetContinualDataset(AvalancheDatasetWrapper):
    """
    Full ImageNet continual learning dataset (placeholder).

    TODO: Implement when Avalanche adds full ImageNet support.
    For now, use Tiny-ImageNet instead.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Full ImageNet continual learning not yet implemented. "
            "Use 'tiny_imagenet' instead."
        )
