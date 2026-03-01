"""
Visualize AdaSlot outputs using the full pretrained model (56/56 checkpoint keys).

Pipeline:
  CIFAR-100 (32x32) → resize 128x128 → AdaSlotModel (pretrained CLEVR10) →
  reconstruction (128x128) + per-slot decomposition + attention masks
"""
from torch.utils.data import DataLoader
from avalanche.benchmarks.classic import SplitCIFAR100
from src.models.adaslot.model import AdaSlotModel
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


CHECKPOINT = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Load CIFAR-100 ----------------------------------------------------
    benchmark = SplitCIFAR100(n_experiences=10, return_task_id=False, seed=42)
    dataloader = DataLoader(
        benchmark.train_stream[0].dataset, batch_size=6, shuffle=True)
    batch = next(iter(dataloader))
    images_small, labels = batch[0].to(device), batch[1]

    # AdaSlot was trained on 128×128 (CLEVR10) – resize CIFAR-100 32×32 to 128×128
    images = F.interpolate(images_small, size=(
        128, 128), mode='bilinear', align_corners=False)
    print(f"Images resized: {images_small.shape} -> {images.shape}")
    print(f"Labels: {labels.tolist()}")

    # ── 2. Build AdaSlotModel (CLEVR10 config: 11 slots, 128×128) -----------
    model = AdaSlotModel(
        resolution=(128, 128),
        num_slots=11,
        slot_dim=64,
        num_iterations=3,
        feature_dim=64,
        kvq_dim=128,
        low_bound=1,
    ).to(device)

    # Load ALL 56 pretrained weights
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {total_params:,} params, 56/56 checkpoint keys loaded (strict)")

    # ── 3. Forward pass ------------------------------------------------------
    with torch.no_grad():
        out = model(images, global_step=0)

    reconstruction = out['reconstruction'].cpu()        # (B, 3, 128, 128)
    slot_recons = out['object_reconstructions'].cpu()  # (B, K, 3, 128, 128)
    masks = out['masks'].cpu()                  # (B, K, 128, 128)
    slots = out['slots'].cpu()                  # (B, K, 64)
    hard_keep = out['hard_keep_decision'].cpu()     # (B, K)
    slots_keep_prob = out['slots_keep_prob'].cpu()        # (B, K)

    B, K = hard_keep.shape
    print(
        f"\nOutputs:  recon {reconstruction.shape}  masks {masks.shape}  slots {slots.shape}")
    print(f"Active slots/sample: {hard_keep.sum(dim=1).int().tolist()}")

    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    # ── Helpers --------------------------------------------------------------
    def to_img(t):
        t = t.permute(1, 2, 0).numpy()
        return np.clip((t - t.min()) / (t.max() - t.min() + 1e-8), 0, 1)

    # ── 4. Original vs Reconstruction ----------------------------------------
    n_show = min(B, 6)
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.5, 5))

    for b in range(n_show):
        # Row 0: input (resized)
        ax = axes[0, b]
        ax.imshow(to_img(images[b].cpu()))
        ax.set_title(f"Input\nCls {labels[b].item()}", fontsize=9)
        ax.axis('off')

        # Row 1: reconstruction
        ax = axes[1, b]
        mse = ((images[b].cpu() - reconstruction[b])**2).mean().item()
        ax.imshow(to_img(reconstruction[b]))
        ax.set_title(f"Recon\nMSE={mse:.4f}", fontsize=9)
        ax.axis('off')

    plt.suptitle('Input (32→128px) vs Full-Image Reconstruction', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved: visualizations/reconstruction.png")
    plt.close()

    # ── 5. Per-slot decomposition (sample 0) ---------------------------------
    b = 0
    active = hard_keep[b].bool()   # (K,) bool mask
    n_active = active.sum().item()

    # Show up to 11 cols: input | recon | slot0..slotK-1
    n_cols = 2 + K
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2, 2.5))

    axes[0].imshow(to_img(images[b].cpu()))
    axes[0].set_title('Input')
    axes[0].axis('off')
    axes[1].imshow(to_img(reconstruction[b]))
    axes[1].set_title('Recon')
    axes[1].axis('off')

    for k in range(K):
        ax = axes[2 + k]
        is_active = hard_keep[b, k].item() > 0.5
        # Show slot reconstruction blended with its alpha mask
        slot_rgb = to_img(slot_recons[b, k])   # (H, W, 3)
        alpha_k = masks[b, k].numpy()         # (H, W)
        alpha_k = (alpha_k - alpha_k.min()) / \
            (alpha_k.max() - alpha_k.min() + 1e-8)
        ax.imshow(slot_rgb, alpha=0.85)
        ax.imshow(alpha_k, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        color = 'green' if is_active else 'gray'
        prob = slots_keep_prob[b, k].item()
        ax.set_title(f"S{k}\n{'ON' if is_active else 'OFF'} ({prob:.2f})",
                     fontsize=8, color=color)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.suptitle(f'Per-Slot Decomposition – Sample 0 (class {labels[0].item()}) │ '
                 f'{n_active}/{K} slots active', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'slot_decomposition.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved: visualizations/slot_decomposition.png")
    plt.close()

    # ── 6. Attention masks for sample 0 -------------------------------------
    fig, axes = plt.subplots(1, K, figsize=(K * 2, 2))
    for k in range(K):
        ax = axes[k]
        ax.imshow(masks[b, k].numpy(), cmap='viridis',
                  vmin=0, vmax=masks[b].max().item())
        is_active = hard_keep[b, k].item() > 0.5
        ax.set_title(f"S{k}\n{'ON' if is_active else 'OFF'}", fontsize=8,
                     color='green' if is_active else 'gray')
        ax.axis('off')
    plt.suptitle(
        'Spatial Attention Masks (alpha after softmax + Gumbel mask)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_maps.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved: visualizations/attention_maps.png")
    plt.close()

    # ── 7. Slot selection across batch ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for b_i in range(B):
        ax.bar(np.arange(K) + b_i * 0.1,
               slots_keep_prob[b_i].numpy(), width=0.09,
               label=f'cls {labels[b_i].item()}', alpha=0.7)
    ax.set_xlabel('Slot index')
    ax.set_ylabel('Keep probability')
    ax.set_title('Gumbel Keep Probabilities per Sample')
    ax.axhline(0.5, color='red', ls='--', alpha=0.5)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(hard_keep.numpy().T, cmap='RdYlGn',
                   vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Slot')
    ax.set_title('Hard Keep Decision (1=active)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'slot_selection.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved: visualizations/slot_selection.png")
    plt.close()

    # ── Summary --------------------------------------------------------------
    mean_mse = ((images.cpu() - reconstruction)**2).mean().item()
    print(f"\n{'='*72}")
    print(f"SUMMARY  |  {B} CIFAR-100 samples, resized 32->128px")
    print(f"  Reconstruction MSE : {mean_mse:.6f}")
    print(f"  Active slots/sample: {hard_keep.sum(dim=1).int().tolist()}")
    print(f"  Mean keep prob     : {slots_keep_prob.mean():.3f}")
    print(f"\nAll visualizations saved to: visualizations/")


if __name__ == '__main__':
    main()
