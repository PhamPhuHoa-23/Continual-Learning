"""
Visualize AdaSlot - Simple Slot Visualization Script

Load trained AdaSlot checkpoint and visualize slot attention masks.
"""

import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.models.adaslot.model import AdaSlotModel
from src.data.continual_cifar100 import get_continual_cifar100_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AdaSlot slots")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to AdaSlot checkpoint")
    parser.add_argument("--num_images", type=int, default=8,
                        help="Number of images to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def visualize_slots(model, images, labels, args, save_path):
    """Visualize slot attention masks for a batch of images."""
    model.eval()
    
    with torch.no_grad():
        # Resize to AdaSlot resolution
        X = F.interpolate(
            images, size=(128, 128),
            mode="bilinear", align_corners=False
        )
        
        # Forward pass
        out = model(X, global_step=0)
        
        slots = out["slots"]  # (B, K, D)
        recon = out["reconstruction"]  # (B, 3, H, W)
        masks = out.get("masks", None)  # (B, K, H, W) if available
        
        B, K, D = slots.shape
        
        # Create figure
        fig, axes = plt.subplots(B, K + 2, figsize=(3 * (K + 2), 3 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        
        for b in range(B):
            # Original image
            img = images[b].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[b, 0].imshow(img)
            axes[b, 0].set_title(f"Original (Class {labels[b].item()})")
            axes[b, 0].axis('off')
            
            # Reconstruction
            recon_img = recon[b].cpu().permute(1, 2, 0).numpy()
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
            axes[b, 1].imshow(recon_img)
            axes[b, 1].set_title("Reconstruction")
            axes[b, 1].axis('off')
            
            # Slot masks
            if masks is not None:
                for k in range(K):
                    mask = masks[b, k].cpu().numpy()
                    axes[b, k + 2].imshow(mask, cmap='hot')
                    axes[b, k + 2].set_title(f"Slot {k}")
                    axes[b, k + 2].axis('off')
            else:
                # If no masks, just show slot index
                for k in range(K):
                    axes[b, k + 2].text(0.5, 0.5, f"Slot {k}\n(no mask)",
                                       ha='center', va='center',
                                       transform=axes[b, k + 2].transAxes)
                    axes[b, k + 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")


def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract args from checkpoint
    if 'args' in ckpt:
        ckpt_args = ckpt['args']
        num_slots = ckpt_args['num_slots']
        slot_dim = ckpt_args['slot_dim']
        n_tasks = ckpt_args.get('n_tasks', 10)
        batch_size = min(args.num_images, 32)
    else:
        # Default values
        num_slots = 7
        slot_dim = 64
        n_tasks = 10
        batch_size = args.num_images
    
    # Initialize model
    print(f"Initializing AdaSlot (slots={num_slots}, dim={slot_dim})")
    model = AdaSlotModel(num_slots=num_slots, slot_dim=slot_dim)
    
    # Load weights
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(args.device)
    model.eval()
    
    # Load data
    print("Loading CIFAR-100 data...")
    train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
        n_tasks=n_tasks,
        batch_size=batch_size,
        num_workers=0,
        seed=42
    )
    
    # Get a batch from task 1
    test_loader = test_loaders[0]
    images, labels = next(iter(test_loader))
    images = images[:args.num_images].to(args.device)
    labels = labels[:args.num_images]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize
    save_path = output_dir / "slot_visualization.png"
    print(f"\nVisualizing {args.num_images} images...")
    visualize_slots(model, images, labels, args, save_path)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
