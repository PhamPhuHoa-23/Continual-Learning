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
):
    """
    Create Tiny-ImageNet continual learning benchmark using Avalanche.
    Requires `avalanche-lib` to be installed.
    
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
    from avalanche.benchmarks.classic import SplitTinyImageNet  # lazy import
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



# ─────────────────────────────────────────────────────────────────────────────
#  Standalone loader (no Avalanche required)
#  Interface matches get_continual_cifar100_loaders exactly:
#    train_loaders, test_loaders, class_order = get_continual_tinyimagenet_loaders(...)
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import List, Tuple, Optional


_TINYIMAGENET_URL = (
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
)
_TINYIMAGENET_CLASSES = 200


def _download_tinyimagenet(root: str) -> str:
    """Download and extract Tiny-ImageNet to `root/tiny-imagenet-200/`.

    Returns path to the extracted root directory.
    """
    extract_dir = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(extract_dir):
        return extract_dir   # already there

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")

    if not os.path.isfile(zip_path):
        print(f"[INFO] Downloading Tiny-ImageNet-200 (~250 MB) …")
        urllib.request.urlretrieve(_TINYIMAGENET_URL, zip_path)
        print("[INFO] Download complete.")

    print("[INFO] Extracting Tiny-ImageNet-200 …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    print("[INFO] Extraction complete.")
    return extract_dir


def _prepare_val_dir(tiny_root: str) -> None:
    """Reorganise the val/ directory so ImageFolder can read it.

    Tiny-ImageNet's val/images/ is flat with a val_annotations.txt file
    that maps filenames → wnids.  This function moves images into
    per-class sub-directories (idempotent — skips if already done).
    """
    val_dir   = os.path.join(tiny_root, "val")
    ann_file  = os.path.join(val_dir, "val_annotations.txt")
    img_dir   = os.path.join(val_dir, "images")

    if not os.path.isfile(ann_file):
        return  # already reorganised or missing

    with open(ann_file) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split("\t")
        fname, wnid = parts[0], parts[1]
        src_path = os.path.join(img_dir, fname)
        dst_dir  = os.path.join(val_dir, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, fname)
        if os.path.isfile(src_path) and not os.path.isfile(dst_path):
            shutil.move(src_path, dst_path)

    # Remove the now-empty images/ dir and annotation file
    try:
        shutil.rmtree(img_dir, ignore_errors=True)
        os.remove(ann_file)
    except OSError:
        pass


class _TinyImageNetTaskDataset(Dataset):
    """Wraps an ImageFolder subset filtered to a specific set of original class ids."""

    def __init__(
        self,
        base_dataset: ImageFolder,
        task_classes: List[int],  # original class indices (0–199)
    ):
        self.base_dataset = base_dataset
        self.task_classes_set = set(task_classes)

        # ImageFolder stores .targets as a list of ints matching folder order
        self.indices = [
            i for i, t in enumerate(base_dataset.targets)
            if t in self.task_classes_set
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]
        return img, label


def get_continual_tinyimagenet_loaders(
    n_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    resolution: int = 64,
    root: str = "./data",
    pin_memory: bool = True,
) -> Tuple[List[DataLoader], List[DataLoader], np.ndarray]:
    """Return per-task DataLoaders for Split Tiny-ImageNet (200 classes).

    Interface is identical to ``get_continual_cifar100_loaders``:
        train_loaders, test_loaders, class_order = get_continual_tinyimagenet_loaders(...)

    Args:
        n_tasks:     Number of sequential tasks (must divide 200 evenly).
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        seed:        Random seed for class permutation.
        resolution:  Spatial resolution to resize images to (default 64).
        root:        Directory where Tiny-ImageNet will be downloaded.
        pin_memory:  Pin DataLoader memory.

    Returns:
        train_loaders: List[DataLoader] — one per task (current-task classes only).
        test_loaders:  List[DataLoader] — one per task (all classes seen so far).
        class_order:   np.ndarray of shape (200,) — permuted class indices.
    """
    if _TINYIMAGENET_CLASSES % n_tasks != 0:
        raise ValueError(
            f"200 classes must be divisible by n_tasks ({n_tasks}). "
            f"Use n_tasks in {{1,2,4,5,8,10,20,25,40,50,100,200}}."
        )

    # ── Download / prepare dataset ────────────────────────────────────
    tiny_root = _download_tinyimagenet(root)
    _prepare_val_dir(tiny_root)

    train_dir = os.path.join(tiny_root, "train")
    val_dir   = os.path.join(tiny_root, "val")

    # ── Transforms ───────────────────────────────────────────────────
    _mean = (0.485, 0.456, 0.406)
    _std  = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

    # ── Build base ImageFolder datasets ──────────────────────────────
    base_train = ImageFolder(train_dir, transform=train_transform)
    base_val   = ImageFolder(val_dir,   transform=test_transform)

    n_classes = len(base_train.classes)   # should be 200

    # ── Class permutation (reproducible) ─────────────────────────────
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(n_classes)
    classes_per_task = n_classes // n_tasks
    task_classes_list = [
        class_order[i * classes_per_task : (i + 1) * classes_per_task].tolist()
        for i in range(n_tasks)
    ]

    # ── Build per-task loaders ────────────────────────────────────────
    train_loaders: List[DataLoader] = []
    test_loaders:  List[DataLoader] = []

    for task_id in range(n_tasks):
        current_classes = task_classes_list[task_id]
        seen_classes    = [c for sub in task_classes_list[: task_id + 1] for c in sub]

        # Training: only current-task classes
        train_ds = _TinyImageNetTaskDataset(base_train, current_classes)
        train_loaders.append(DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        ))

        # Evaluation: all classes seen so far
        test_ds = _TinyImageNetTaskDataset(base_val, seen_classes)
        test_loaders.append(DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ))

        logger.info(
            f"Task {task_id:>2d}: train={len(train_ds):>6d} samples "
            f"({len(current_classes)} classes) | "
            f"test={len(test_ds):>6d} samples ({len(seen_classes)} classes)"
        )

    return train_loaders, test_loaders, class_order


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

