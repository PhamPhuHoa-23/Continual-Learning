"""
Custom Dataset Template

Example template for creating custom continual learning datasets
that are not available in Avalanche.

Use this template to create your own datasets:
1. Inherit from BaseContinualDataset
2. Implement get_task_data() and get_task_classes()
3. Register with @DATASET_REGISTRY.register()
"""

from typing import List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from cont_src.core.registry import DATASET_REGISTRY
from cont_src.data.base_dataset import BaseContinualDataset, TaskData


class CustomTaskDataset(Dataset):
    """
    Custom PyTorch Dataset for a single task.
    
    This is a simple example - replace with your own data loading logic.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        transform=None,
    ):
        """
        Args:
            data: Array of shape (N, C, H, W) or (N, H, W, C)
            targets: Array of shape (N,) with class labels
            transform: Optional transforms
        """
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


# Example: Custom continual learning dataset
@DATASET_REGISTRY.register("custom_example")
class CustomContinualDataset(BaseContinualDataset):
    """
    Example custom continual learning dataset.
    
    Replace this with your own dataset implementation.
    
    Steps to create your own dataset:
    1. Load your data (images, labels)
    2. Split into tasks
    3. Create DataLoaders
    4. Return TaskData objects
    
    Example:
        >>> from cont_src.core.registry import DATASET_REGISTRY
        >>> dataset = DATASET_REGISTRY.build(
        ...     "custom_example",
        ...     n_tasks=5,
        ...     batch_size=32
        ... )
    """
    
    def __init__(
        self,
        n_tasks: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: Optional[int] = 42,
        data_root: str = "./data",
        **kwargs
    ):
        super().__init__(
            n_tasks=n_tasks,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            data_root=data_root,
            **kwargs
        )
        
        # Example: 50 classes, 5 tasks = 10 classes per task
        self.n_classes_total = 50
        self.classes_per_task = self.n_classes_total // n_tasks
        
        # Example: Random class order
        rng = np.random.RandomState(seed)
        self.class_order = rng.permutation(self.n_classes_total).tolist()
        
        # Load your data here
        # For this example, we'll create dummy data
        self._load_data()
    
    def _load_data(self):
        """
        Load your dataset.
        
        Replace this with actual data loading logic:
        - Load from files
        - Download from URLs
        - Load from HuggingFace datasets
        - etc.
        """
        # Example: Create dummy data
        # Replace with your actual data loading
        self.train_data = {}
        self.test_data = {}
        
        for class_idx in range(self.n_classes_total):
            # Example: 500 training samples per class
            self.train_data[class_idx] = {
                "images": np.random.randn(500, 3, 32, 32).astype(np.float32),
                "labels": np.full(500, class_idx, dtype=np.int64),
            }
            
            # Example: 100 test samples per class
            self.test_data[class_idx] = {
                "images": np.random.randn(100, 3, 32, 32).astype(np.float32),
                "labels": np.full(100, class_idx, dtype=np.int64),
            }
    
    def get_task_data(self, task_id: int) -> TaskData:
        """Get data for a specific task."""
        if task_id < 0 or task_id >= self.n_tasks:
            raise ValueError(f"Task ID {task_id} out of range")
        
        # Cache task data
        if task_id in self.task_data:
            return self.task_data[task_id]
        
        # Get classes for this task
        task_classes = self.get_task_classes(task_id)
        
        # Collect training data for this task
        train_images_list = []
        train_labels_list = []
        
        for class_idx in task_classes:
            train_images_list.append(self.train_data[class_idx]["images"])
            train_labels_list.append(self.train_data[class_idx]["labels"])
        
        train_images = np.concatenate(train_images_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        
        # Create train dataset and loader
        train_dataset = CustomTaskDataset(train_images, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        # Collect test data for all classes seen so far
        test_images_list = []
        test_labels_list = []
        
        classes_so_far = []
        for t in range(task_id + 1):
            classes_so_far.extend(self.get_task_classes(t))
        
        for class_idx in classes_so_far:
            test_images_list.append(self.test_data[class_idx]["images"])
            test_labels_list.append(self.test_data[class_idx]["labels"])
        
        test_images = np.concatenate(test_images_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
        
        # Create test dataset and loader
        test_dataset = CustomTaskDataset(test_images, test_labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        # Create TaskData
        task_data = TaskData(
            task_id=task_id,
            train_loader=train_loader,
            test_loader=test_loader,
            classes=task_classes,
            n_classes_total=len(classes_so_far),
        )
        
        self.task_data[task_id] = task_data
        return task_data
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """Get class indices for a task."""
        start_idx = task_id * self.classes_per_task
        end_idx = start_idx + self.classes_per_task
        return self.class_order[start_idx:end_idx]


# Template for dataset with custom data format
class YourCustomDataset(BaseContinualDataset):
    """
    Template for your own dataset.
    
    Steps:
    1. Copy this class
    2. Rename to your dataset name
    3. Add @DATASET_REGISTRY.register("your_name") decorator
    4. Implement _load_data() with your data loading logic
    5. Implement get_task_data() and get_task_classes()
    6. Test with your data
    
    Example usage after implementation:
        >>> dataset = DATASET_REGISTRY.build(
        ...     "your_name",
        ...     n_tasks=10,
        ...     batch_size=64
        ... )
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
        super().__init__(
            n_tasks=n_tasks,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            data_root=data_root,
            **kwargs
        )
        
        # TODO: Set your dataset metadata
        self.n_classes_total = None  # Total number of classes
        self.classes_per_task = None  # Classes per task
        self.class_order = None  # Order of classes
        
        # TODO: Load your data
        self._load_data()
    
    def _load_data(self):
        """TODO: Implement your data loading logic."""
        raise NotImplementedError("Implement data loading for your dataset")
    
    def get_task_data(self, task_id: int) -> TaskData:
        """TODO: Implement task data creation."""
        raise NotImplementedError("Implement get_task_data for your dataset")
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """TODO: Implement task class retrieval."""
        raise NotImplementedError("Implement get_task_classes for your dataset")
