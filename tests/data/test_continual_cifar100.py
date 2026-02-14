"""
Unit Tests for CIFAR-100 Continual Learning Data Pipeline

Tests cover:
1. ClassIncrementalSplit: Class splitting logic
2. ContinualCIFAR100Dataset: Dataset filtering
3. get_continual_cifar100_loaders: Data loader creation
4. Data integrity: Labels, shapes, class distribution
5. Reproducibility: Seed consistency

Run tests with:
    pytest tests/data/test_continual_cifar100.py -v
    pytest tests/data/test_continual_cifar100.py -v --cov=src/data

Author: Your Team
Date: 2026-02-12
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.continual_cifar100 import (
    ClassIncrementalSplit,
    ContinualCIFAR100Dataset,
    get_continual_cifar100_loaders,
    get_cifar100_transforms,
)


class TestClassIncrementalSplit:
    """Test ClassIncrementalSplit class."""
    
    def test_initialization_valid(self):
        """Test initialization with valid parameters."""
        splitter = ClassIncrementalSplit(n_tasks=5, n_classes=100, seed=42)
        
        assert splitter.n_classes == 100
        assert splitter.n_tasks == 5
        assert splitter.classes_per_task == 20
        assert len(splitter.class_order) == 100
        assert len(splitter.task_classes) == 5
    
    def test_initialization_invalid_split(self):
        """Test that invalid splits raise ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            ClassIncrementalSplit(n_tasks=7, n_classes=100)
    
    def test_class_order_is_permutation(self):
        """Test that class_order is a valid permutation."""
        splitter = ClassIncrementalSplit(n_tasks=5, seed=42)
        
        # Should contain all classes 0-99
        assert set(splitter.class_order) == set(range(100))
        
        # Should have no duplicates
        assert len(splitter.class_order) == len(set(splitter.class_order))
    
    def test_task_classes_no_overlap(self):
        """Test that task classes don't overlap."""
        splitter = ClassIncrementalSplit(n_tasks=5, seed=42)
        
        all_classes = []
        for task_classes in splitter.task_classes:
            # Check no overlap with previous tasks
            assert len(set(task_classes) & set(all_classes)) == 0
            all_classes.extend(task_classes)
        
        # All tasks together should cover all classes
        assert set(all_classes) == set(range(100))
    
    def test_task_classes_correct_size(self):
        """Test that each task has correct number of classes."""
        splitter = ClassIncrementalSplit(n_tasks=5, seed=42)
        
        for task_classes in splitter.task_classes:
            assert len(task_classes) == 20
    
    def test_get_task_classes(self):
        """Test get_task_classes method."""
        splitter = ClassIncrementalSplit(n_tasks=5, seed=42)
        
        # Valid task IDs
        for task_id in range(5):
            classes = splitter.get_task_classes(task_id)
            assert len(classes) == 20
            assert all(isinstance(c, (int, np.integer)) for c in classes)
        
        # Invalid task IDs
        with pytest.raises(ValueError):
            splitter.get_task_classes(-1)
        
        with pytest.raises(ValueError):
            splitter.get_task_classes(5)
    
    def test_get_seen_classes_up_to_task(self):
        """Test get_seen_classes_up_to_task method."""
        splitter = ClassIncrementalSplit(n_tasks=5, seed=42)
        
        # Task 0: 20 classes
        seen = splitter.get_seen_classes_up_to_task(0)
        assert len(seen) == 20
        
        # Task 1: 40 classes
        seen = splitter.get_seen_classes_up_to_task(1)
        assert len(seen) == 40
        
        # Task 2: 60 classes
        seen = splitter.get_seen_classes_up_to_task(2)
        assert len(seen) == 60
        
        # Task 4: 100 classes
        seen = splitter.get_seen_classes_up_to_task(4)
        assert len(seen) == 100
        assert set(seen) == set(range(100))
    
    def test_get_task_id_for_class(self):
        """Test get_task_id_for_class method."""
        splitter = ClassIncrementalSplit(n_tasks=5, seed=42)
        
        # Each class should belong to exactly one task
        for class_id in range(100):
            task_id = splitter.get_task_id_for_class(class_id)
            assert 0 <= task_id < 5
            assert class_id in splitter.task_classes[task_id]
    
    def test_reproducibility(self):
        """Test that same seed produces same split."""
        splitter1 = ClassIncrementalSplit(n_tasks=5, seed=42)
        splitter2 = ClassIncrementalSplit(n_tasks=5, seed=42)
        
        np.testing.assert_array_equal(splitter1.class_order, splitter2.class_order)
        
        for i in range(5):
            assert splitter1.task_classes[i] == splitter2.task_classes[i]
    
    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        splitter1 = ClassIncrementalSplit(n_tasks=5, seed=42)
        splitter2 = ClassIncrementalSplit(n_tasks=5, seed=123)
        
        # Should be different
        assert not np.array_equal(splitter1.class_order, splitter2.class_order)
    
    def test_different_n_tasks(self):
        """Test different number of tasks."""
        for n_tasks in [2, 5, 10, 20]:
            splitter = ClassIncrementalSplit(n_tasks=n_tasks, seed=42)
            assert splitter.classes_per_task == 100 // n_tasks
            assert len(splitter.task_classes) == n_tasks


class TestContinualCIFAR100Dataset:
    """Test ContinualCIFAR100Dataset class."""
    
    @pytest.fixture
    def task_classes(self):
        """Fixture for task classes."""
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    def test_dataset_creation(self, task_classes, tmp_path):
        """Test dataset creation."""
        dataset = ContinualCIFAR100Dataset(
            root=str(tmp_path),
            train=True,
            task_classes=task_classes,
            download=True
        )
        
        assert len(dataset) > 0
        assert dataset.task_classes == task_classes
    
    def test_dataset_filtering(self, task_classes, tmp_path):
        """Test that dataset only contains specified classes."""
        dataset = ContinualCIFAR100Dataset(
            root=str(tmp_path),
            train=True,
            task_classes=task_classes,
            download=True
        )
        
        # Check all samples
        all_labels = []
        for i in range(min(len(dataset), 100)):  # Check first 100 samples
            _, label = dataset[i]
            all_labels.append(label)
        
        # All labels should be in task_classes
        assert all(label in task_classes for label in all_labels)
    
    def test_dataset_getitem(self, task_classes, tmp_path):
        """Test __getitem__ returns correct format."""
        dataset = ContinualCIFAR100Dataset(
            root=str(tmp_path),
            train=True,
            task_classes=task_classes,
            download=True
        )
        
        image, label = dataset[0]
        
        # Check types
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        
        # Check label in task_classes
        assert label in task_classes
    
    def test_train_vs_test_split(self, task_classes, tmp_path):
        """Test that train and test datasets are different sizes."""
        train_dataset = ContinualCIFAR100Dataset(
            root=str(tmp_path),
            train=True,
            task_classes=task_classes,
            download=True
        )
        
        test_dataset = ContinualCIFAR100Dataset(
            root=str(tmp_path),
            train=False,
            task_classes=task_classes,
            download=True
        )
        
        # Train should be larger than test
        assert len(train_dataset) > len(test_dataset)
        
        # Rough ratio check (CIFAR-100 is 50k train, 10k test)
        # For 10 classes out of 100, expect ~5k train, ~1k test
        assert len(train_dataset) > 1000
        assert len(test_dataset) > 100


class TestTransforms:
    """Test transform functions."""
    
    def test_train_transforms(self):
        """Test training transforms."""
        transform = get_cifar100_transforms(train=True)
        
        # Check it's a Compose
        assert isinstance(transform, torch.nn.Sequential) or hasattr(transform, 'transforms')
    
    def test_test_transforms(self):
        """Test test transforms."""
        transform = get_cifar100_transforms(train=False)
        
        assert isinstance(transform, torch.nn.Sequential) or hasattr(transform, 'transforms')
    
    def test_transforms_output_shape(self, tmp_path):
        """Test that transforms produce correct output shape."""
        from torchvision import datasets
        
        # Load one sample
        dataset = datasets.CIFAR100(root=str(tmp_path), train=True, download=True)
        image, _ = dataset[0]
        
        # Apply transforms
        train_transform = get_cifar100_transforms(train=True)
        test_transform = get_cifar100_transforms(train=False)
        
        train_image = train_transform(image)
        test_image = test_transform(image)
        
        # Both should be [3, 32, 32]
        assert train_image.shape == (3, 32, 32)
        assert test_image.shape == (3, 32, 32)


class TestDataLoaders:
    """Test data loader creation."""
    
    def test_loader_creation(self, tmp_path):
        """Test basic loader creation."""
        train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
            n_tasks=5,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        assert len(train_loaders) == 5
        assert len(test_loaders) == 5
        assert len(class_order) == 100
    
    def test_loader_types(self, tmp_path):
        """Test loader types."""
        train_loaders, test_loaders, _ = get_continual_cifar100_loaders(
            n_tasks=5,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        for loader in train_loaders:
            assert isinstance(loader, DataLoader)
        
        for loader in test_loaders:
            assert isinstance(loader, DataLoader)
    
    def test_batch_iteration(self, tmp_path):
        """Test iterating through batches."""
        train_loaders, test_loaders, _ = get_continual_cifar100_loaders(
            n_tasks=2,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        # Get one batch from first task
        images, labels = next(iter(train_loaders[0]))
        
        assert images.shape[0] <= 64  # Batch size
        assert images.shape[1:] == (3, 32, 32)  # Image shape
        assert len(labels) == images.shape[0]
    
    def test_class_incremental_property(self, tmp_path):
        """Test class-incremental property: test set grows over tasks."""
        train_loaders, test_loaders, _ = get_continual_cifar100_loaders(
            n_tasks=5,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        # Count unique classes in each test loader
        test_classes_per_task = []
        
        for task_id in range(5):
            all_labels = []
            for images, labels in test_loaders[task_id]:
                all_labels.extend(labels.tolist())
            
            unique_classes = len(set(all_labels))
            test_classes_per_task.append(unique_classes)
        
        # Test sets should grow: 20, 40, 60, 80, 100
        expected = [20, 40, 60, 80, 100]
        assert test_classes_per_task == expected
    
    def test_train_classes_per_task(self, tmp_path):
        """Test that each training task has correct number of classes."""
        train_loaders, _, _ = get_continual_cifar100_loaders(
            n_tasks=5,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        for task_id in range(5):
            all_labels = []
            for images, labels in train_loaders[task_id]:
                all_labels.extend(labels.tolist())
            
            unique_classes = len(set(all_labels))
            
            # Each training task should have exactly 20 classes
            assert unique_classes == 20, f"Task {task_id} has {unique_classes} classes, expected 20"
    
    def test_no_class_overlap_across_train_tasks(self, tmp_path):
        """Test that training tasks have no overlapping classes."""
        train_loaders, _, _ = get_continual_cifar100_loaders(
            n_tasks=5,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        all_train_classes = []
        
        for task_id in range(5):
            task_labels = []
            for images, labels in train_loaders[task_id]:
                task_labels.extend(labels.tolist())
            
            task_classes = set(task_labels)
            
            # No overlap with previous tasks
            assert len(set(task_classes) & set(all_train_classes)) == 0
            
            all_train_classes.extend(task_classes)
        
        # All tasks together should cover all 100 classes
        assert len(set(all_train_classes)) == 100
    
    def test_reproducibility_with_seed(self, tmp_path):
        """Test that same seed produces same data loaders."""
        train_loaders1, test_loaders1, class_order1 = get_continual_cifar100_loaders(
            n_tasks=5,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        train_loaders2, test_loaders2, class_order2 = get_continual_cifar100_loaders(
            n_tasks=5,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        # Class order should be the same
        np.testing.assert_array_equal(class_order1, class_order2)
        
        # First batch should be the same
        images1, labels1 = next(iter(train_loaders1[0]))
        images2, labels2 = next(iter(train_loaders2[0]))
        
        # Note: This might fail due to data augmentation randomness
        # Just check labels
        torch.testing.assert_close(labels1, labels2)
    
    def test_different_n_tasks(self, tmp_path):
        """Test with different number of tasks."""
        for n_tasks in [2, 5, 10]:
            train_loaders, test_loaders, _ = get_continual_cifar100_loaders(
                n_tasks=n_tasks,
                batch_size=64,
                num_workers=0,
                root=str(tmp_path),
                seed=42
            )
            
            assert len(train_loaders) == n_tasks
            assert len(test_loaders) == n_tasks
            
            # Check classes per task
            classes_per_task = 100 // n_tasks
            
            for task_id in range(n_tasks):
                all_labels = []
                for images, labels in train_loaders[task_id]:
                    all_labels.extend(labels.tolist())
                
                unique_classes = len(set(all_labels))
                assert unique_classes == classes_per_task


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_task(self, tmp_path):
        """Test with single task (all classes at once)."""
        train_loaders, test_loaders, _ = get_continual_cifar100_loaders(
            n_tasks=1,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        # Should have 100 classes in single task
        all_labels = []
        for images, labels in train_loaders[0]:
            all_labels.extend(labels.tolist())
        
        assert len(set(all_labels)) == 100
    
    def test_many_tasks(self, tmp_path):
        """Test with many tasks (one class per task)."""
        train_loaders, test_loaders, _ = get_continual_cifar100_loaders(
            n_tasks=100,
            batch_size=64,
            num_workers=0,
            root=str(tmp_path),
            seed=42
        )
        
        assert len(train_loaders) == 100
        
        # Each task should have 1 class
        for task_id in range(5):  # Check first 5 tasks
            all_labels = []
            for images, labels in train_loaders[task_id]:
                all_labels.extend(labels.tolist())
            
            assert len(set(all_labels)) == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

