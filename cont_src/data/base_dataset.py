"""
Base Dataset Interfaces for Continual Learning

Provides abstract classes and utilities for continual learning datasets.
All datasets should inherit from these base classes and register with the registry.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from avalanche.benchmarks import CLScenario


@dataclass
class TaskData:
    """
    Data container for a single continual learning task.
    
    Attributes:
        task_id: Task identifier (0-indexed)
        train_loader: Training data loader
        test_loader: Test data loader (tests on all classes seen so far)
        classes: List of class indices in this task
        n_classes_total: Total number of classes seen so far
    """
    task_id: int
    train_loader: DataLoader
    test_loader: DataLoader
    classes: List[int]
    n_classes_total: int


class BaseContinualDataset(ABC):
    """
    Base class for continual learning datasets.
    
    All dataset implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(
        self,
        n_tasks: int,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: Optional[int] = 42,
        data_root: str = "./data",
        **kwargs
    ):
        """
        Initialize continual learning dataset.
        
        Args:
            n_tasks: Number of sequential tasks
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
            data_root: Root directory for dataset storage
            **kwargs: Additional dataset-specific arguments
        """
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.data_root = data_root
        self.kwargs = kwargs
        
        # To be set by subclasses
        self.n_classes_total = None
        self.classes_per_task = None
        self.class_order = None
        self.task_data = {}
    
    @abstractmethod
    def get_task_data(self, task_id: int) -> TaskData:
        """
        Get data for a specific task.
        
        Args:
            task_id: Task identifier (0-indexed)
        
        Returns:
            TaskData object with train/test loaders
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_task_classes(self, task_id: int) -> List[int]:
        """
        Get class indices for a specific task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            List of class indices
        """
        raise NotImplementedError
    
    def get_all_task_data(self) -> List[TaskData]:
        """
        Get data for all tasks.
        
        Returns:
            List of TaskData objects
        """
        return [self.get_task_data(i) for i in range(self.n_tasks)]
    
    def __len__(self) -> int:
        """Return number of tasks."""
        return self.n_tasks
    
    def __getitem__(self, task_id: int) -> TaskData:
        """Get task data by index."""
        return self.get_task_data(task_id)


class AvalancheDatasetWrapper(BaseContinualDataset):
    """
    Wrapper for Avalanche benchmarks.
    
    Provides unified interface for Avalanche's built-in benchmarks.
    """
    
    def __init__(
        self,
        benchmark: CLScenario,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Initialize Avalanche dataset wrapper.
        
        Args:
            benchmark: Avalanche CLScenario benchmark
            batch_size: Batch size for data loaders
            num_workers: Number of workers
            pin_memory: Pin memory for faster GPU transfer
        """
        # Initialize base class
        n_tasks = benchmark.n_experiences
        seed = kwargs.get("seed", 42)
        data_root = kwargs.get("data_root", "./data")
        
        super().__init__(
            n_tasks=n_tasks,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            data_root=data_root,
            **kwargs
        )
        
        self.benchmark = benchmark
        self.pin_memory = pin_memory
        
        # Extract metadata
        self.train_stream = benchmark.train_stream
        self.test_stream = benchmark.test_stream
        
        # Get class information
        if hasattr(benchmark, 'n_classes'):
            self.n_classes_total = benchmark.n_classes
        else:
            # Infer from experiences
            all_classes = set()
            for exp in self.train_stream:
                all_classes.update(exp.classes_in_this_experience)
            self.n_classes_total = len(all_classes)
    
    def get_task_data(self, task_id: int) -> TaskData:
        """Get data for a specific task."""
        if task_id < 0 or task_id >= self.n_tasks:
            raise ValueError(f"Task ID {task_id} out of range [0, {self.n_tasks})")
        
        # Cache task data
        if task_id in self.task_data:
            return self.task_data[task_id]
        
        # Get train experience
        train_exp = self.train_stream[task_id]
        train_dataset = train_exp.dataset
        
        # Create train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        # Get test experience (tests on all classes seen so far)
        test_exp = self.test_stream[task_id]
        test_dataset = test_exp.dataset
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        # Get classes in this task
        classes = list(train_exp.classes_in_this_experience)
        
        # Count total classes seen so far
        n_classes_so_far = 0
        for i in range(task_id + 1):
            n_classes_so_far += len(self.train_stream[i].classes_in_this_experience)
        
        task_data = TaskData(
            task_id=task_id,
            train_loader=train_loader,
            test_loader=test_loader,
            classes=classes,
            n_classes_total=n_classes_so_far,
        )
        
        self.task_data[task_id] = task_data
        return task_data
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """Get class indices for a task."""
        return list(self.train_stream[task_id].classes_in_this_experience)


class DatasetInfo:
    """
    Dataset information container.
    
    Stores metadata about a dataset for easy reference.
    """
    
    def __init__(
        self,
        name: str,
        n_classes: int,
        image_shape: Tuple[int, int, int],
        n_train_samples: int,
        n_test_samples: int,
        description: str = "",
    ):
        self.name = name
        self.n_classes = n_classes
        self.image_shape = image_shape
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.description = description
    
    def __repr__(self) -> str:
        return (
            f"DatasetInfo(name='{self.name}', "
            f"classes={self.n_classes}, "
            f"shape={self.image_shape}, "
            f"train={self.n_train_samples}, test={self.n_test_samples})"
        )


# Dataset metadata
DATASET_INFO = {
    "cifar10": DatasetInfo(
        name="CIFAR-10",
        n_classes=10,
        image_shape=(3, 32, 32),
        n_train_samples=50000,
        n_test_samples=10000,
        description="10 classes of natural images",
    ),
    "cifar100": DatasetInfo(
        name="CIFAR-100",
        n_classes=100,
        image_shape=(3, 32, 32),
        n_train_samples=50000,
        n_test_samples=10000,
        description="100 classes of natural images",
    ),
    "mnist": DatasetInfo(
        name="MNIST",
        n_classes=10,
        image_shape=(1, 28, 28),
        n_train_samples=60000,
        n_test_samples=10000,
        description="Handwritten digits 0-9",
    ),
    "tiny_imagenet": DatasetInfo(
        name="Tiny-ImageNet",
        n_classes=200,
        image_shape=(3, 64, 64),
        n_train_samples=100000,
        n_test_samples=10000,
        description="200 classes from ImageNet, 64x64 resolution",
    ),
}
