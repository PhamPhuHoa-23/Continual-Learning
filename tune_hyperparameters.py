"""
Hyperparameter Tuning Script with Visualization
Tests different HDBSCAN and routing parameters to find optimal configuration
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.continual_cifar100 import get_continual_cifar100_loaders
from src.models.adaslot.model import AdaSlotModel

try:
    from hdbscan import HDBSCAN
except ImportError:
    logger.warning("HDBSCAN not found, using DBSCAN fallback")
    from sklearn.cluster import DBSCAN as HDBSCAN


def load_adaslot(checkpoint_path, device):
    """Load pre-trained AdaSlot model"""
    logger.info(f"Loading AdaSlot from {checkpoint_path}")
    adaslot = AdaSlotModel(
        num_slots=7,
        slot_dim=64
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        adaslot.load_state_dict(ckpt['model_state_dict'])
    else:
        adaslot.load_state_dict(ckpt)
    
    adaslot.eval()
    for param in adaslot.parameters():
        param.requires_grad = False
    
    return adaslot


def extract_slot_embeddings(adaslot, dataloader, device, max_batches=10):
    """Extract slot embeddings from Task 1 data"""
    logger.info("Extracting slot embeddings...")
    all_slots = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= max_batches:
                break
            
            images = images.to(device)
            slots = adaslot.encode(images)  # [B, num_slots, slot_dim]
            
            # Flatten: [B*num_slots, slot_dim]
            slots_flat = slots.view(-1, slots.size(-1))
            all_slots.append(slots_flat.cpu().numpy())
            
            # Repeat labels for each slot
            labels_repeated = labels.repeat_interleave(slots.size(1))  # Use slots.size(1) for num_slots
            all_labels.append(labels_repeated.numpy())
    
    slots_array = np.concatenate(all_slots, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    logger.info(f"Extracted {slots_array.shape[0]} slot embeddings")
    return slots_array, labels_array


def test_hdbscan_params(slots, labels, param_grid):
    """Test different HDBSCAN parameter combinations"""
    logger.info("Testing HDBSCAN parameter combinations...")
    results = []
    
    for min_cluster_size in param_grid['min_cluster_size']:
        for min_samples in param_grid['min_samples']:
            logger.info(f"Testing min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            cluster_labels = clusterer.fit_predict(slots)
            
            # Count clusters (excluding noise label -1)
            unique_clusters = set(cluster_labels)
            unique_clusters.discard(-1)
            num_clusters = len(unique_clusters)
            
            # Count noise points
            num_noise = np.sum(cluster_labels == -1)
            noise_ratio = num_noise / len(cluster_labels)
            
            # Calculate samples per cluster
            cluster_sizes = []
            for cluster_id in unique_clusters:
                cluster_size = np.sum(cluster_labels == cluster_id)
                cluster_sizes.append(cluster_size)
            
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
            min_cluster_size_actual = np.min(cluster_sizes) if cluster_sizes else 0
            max_cluster_size_actual = np.max(cluster_sizes) if cluster_sizes else 0
            
            results.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'num_clusters': num_clusters,
                'noise_ratio': noise_ratio,
                'avg_cluster_size': avg_cluster_size,
                'min_size': min_cluster_size_actual,
                'max_size': max_cluster_size_actual,
                'cluster_labels': cluster_labels
            })
            
            logger.info(f"  → {num_clusters} clusters, {noise_ratio*100:.1f}% noise, "
                       f"avg size: {avg_cluster_size:.1f}")
    
    return results


def visualize_embeddings(slots, labels, cluster_labels, save_dir, title_suffix=""):
    """Visualize slot embeddings with PCA and t-SNE"""
    logger.info("Creating visualizations...")
    
    # Sample if too many points
    max_points = 2000
    if len(slots) > max_points:
        indices = np.random.choice(len(slots), max_points, replace=False)
        slots_vis = slots[indices]
        labels_vis = labels[indices]
        cluster_labels_vis = cluster_labels[indices]
    else:
        slots_vis = slots
        labels_vis = labels
        cluster_labels_vis = cluster_labels
    
    # PCA projection
    logger.info("Computing PCA...")
    pca = PCA(n_components=2)
    slots_pca = pca.fit_transform(slots_vis)
    
    # t-SNE projection
    logger.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    slots_tsne = tsne.fit_transform(slots_vis)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # PCA colored by class labels
    ax = axes[0, 0]
    scatter = ax.scatter(slots_pca[:, 0], slots_pca[:, 1], 
                        c=labels_vis, cmap='tab10', s=10, alpha=0.6)
    ax.set_title(f'PCA - Colored by Class Label{title_suffix}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.colorbar(scatter, ax=ax, label='Class')
    
    # PCA colored by cluster assignments
    ax = axes[0, 1]
    scatter = ax.scatter(slots_pca[:, 0], slots_pca[:, 1], 
                        c=cluster_labels_vis, cmap='tab20', s=10, alpha=0.6)
    ax.set_title(f'PCA - Colored by HDBSCAN Cluster{title_suffix}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # t-SNE colored by class labels
    ax = axes[1, 0]
    scatter = ax.scatter(slots_tsne[:, 0], slots_tsne[:, 1], 
                        c=labels_vis, cmap='tab10', s=10, alpha=0.6)
    ax.set_title(f't-SNE - Colored by Class Label{title_suffix}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Class')
    
    # t-SNE colored by cluster assignments
    ax = axes[1, 1]
    scatter = ax.scatter(slots_tsne[:, 0], slots_tsne[:, 1], 
                        c=cluster_labels_vis, cmap='tab20', s=10, alpha=0.6)
    ax.set_title(f't-SNE - Colored by HDBSCAN Cluster{title_suffix}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.tight_layout()
    save_path = save_dir / f'embeddings_visualization{title_suffix.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()


def plot_hdbscan_results(results, save_dir):
    """Plot HDBSCAN parameter sweep results"""
    logger.info("Plotting HDBSCAN parameter sweep...")
    
    # Extract data
    min_cluster_sizes = [r['min_cluster_size'] for r in results]
    min_samples_list = [r['min_samples'] for r in results]
    num_clusters = [r['num_clusters'] for r in results]
    noise_ratios = [r['noise_ratio'] * 100 for r in results]
    avg_sizes = [r['avg_cluster_size'] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of clusters
    ax = axes[0, 0]
    for min_samples in sorted(set(min_samples_list)):
        mask = [ms == min_samples for ms in min_samples_list]
        x = [min_cluster_sizes[i] for i, m in enumerate(mask) if m]
        y = [num_clusters[i] for i, m in enumerate(mask) if m]
        ax.plot(x, y, 'o-', label=f'min_samples={min_samples}')
    ax.set_xlabel('min_cluster_size')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Number of Clusters vs Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Noise ratio
    ax = axes[0, 1]
    for min_samples in sorted(set(min_samples_list)):
        mask = [ms == min_samples for ms in min_samples_list]
        x = [min_cluster_sizes[i] for i, m in enumerate(mask) if m]
        y = [noise_ratios[i] for i, m in enumerate(mask) if m]
        ax.plot(x, y, 'o-', label=f'min_samples={min_samples}')
    ax.set_xlabel('min_cluster_size')
    ax.set_ylabel('Noise Ratio (%)')
    ax.set_title('Noise Ratio vs Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average cluster size
    ax = axes[1, 0]
    for min_samples in sorted(set(min_samples_list)):
        mask = [ms == min_samples for ms in min_samples_list]
        x = [min_cluster_sizes[i] for i, m in enumerate(mask) if m]
        y = [avg_sizes[i] for i, m in enumerate(mask) if m]
        ax.plot(x, y, 'o-', label=f'min_samples={min_samples}')
    ax.set_xlabel('min_cluster_size')
    ax.set_ylabel('Average Cluster Size')
    ax.set_title('Average Cluster Size vs Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary heatmap (number of clusters)
    ax = axes[1, 1]
    unique_min_cluster = sorted(set(min_cluster_sizes))
    unique_min_samples = sorted(set(min_samples_list))
    
    heatmap_data = np.zeros((len(unique_min_samples), len(unique_min_cluster)))
    for r in results:
        i = unique_min_samples.index(r['min_samples'])
        j = unique_min_cluster.index(r['min_cluster_size'])
        heatmap_data[i, j] = r['num_clusters']
    
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='viridis',
                xticklabels=unique_min_cluster, yticklabels=unique_min_samples,
                ax=ax, cbar_kws={'label': 'Number of Clusters'})
    ax.set_xlabel('min_cluster_size')
    ax.set_ylabel('min_samples')
    ax.set_title('Number of Clusters Heatmap')
    
    plt.tight_layout()
    save_path = save_dir / 'hdbscan_parameter_sweep.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved parameter sweep plot to {save_path}")
    plt.close()


def print_recommendations(results):
    """Print recommendations based on results"""
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)
    
    # Find configurations that produce 10-20 clusters (reasonable for 10 classes)
    good_configs = [r for r in results if 10 <= r['num_clusters'] <= 20]
    
    if good_configs:
        logger.info("\nConfigurations producing 10-20 clusters (good for 10 classes):")
        for config in good_configs:
            logger.info(f"  min_cluster_size={config['min_cluster_size']}, "
                       f"min_samples={config['min_samples']}")
            logger.info(f"    → {config['num_clusters']} clusters, "
                       f"{config['noise_ratio']*100:.1f}% noise, "
                       f"avg size: {config['avg_cluster_size']:.1f}")
        
        # Find best: most clusters in range, least noise
        best = max(good_configs, key=lambda x: (x['num_clusters'], -x['noise_ratio']))
        logger.info(f"\nRECOMMENDED: min_cluster_size={best['min_cluster_size']}, "
                   f"min_samples={best['min_samples']}")
    else:
        logger.info("\nNo configurations produced 10-20 clusters.")
        logger.info("Consider these alternatives:")
        
        # Show closest to 15 clusters
        closest = min(results, key=lambda x: abs(x['num_clusters'] - 15))
        logger.info(f"\nClosest to 15 clusters: min_cluster_size={closest['min_cluster_size']}, "
                   f"min_samples={closest['min_samples']}")
        logger.info(f"  → {closest['num_clusters']} clusters, "
                   f"{closest['noise_ratio']*100:.1f}% noise")
    
    logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning with Visualization')
    parser.add_argument('--adaslot_ckpt', type=str, required=True,
                       help='Path to AdaSlot checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for CIFAR-100')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loading')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Max samples per class')
    parser.add_argument('--max_batches', type=int, default=20,
                       help='Max batches to extract embeddings from')
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                       help='Output directory for results')
    
    # HDBSCAN parameter grid
    parser.add_argument('--min_cluster_sizes', type=int, nargs='+',
                       default=[5, 10, 15, 20, 25, 30],
                       help='min_cluster_size values to test')
    parser.add_argument('--min_samples_list', type=int, nargs='+',
                       default=[3, 5, 10],
                       help='min_samples values to test')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load AdaSlot
    adaslot = load_adaslot(args.adaslot_ckpt, device)
    
    # Load Task 1 data
    logger.info("Loading CIFAR-100 Task 1 data...")
    train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
        n_tasks=10,
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
        max_samples_per_task=args.max_samples  # Now per-class!
    )
    
    task1_loader = train_loaders[0]
    logger.info(f"Task 1 loader: {len(task1_loader.dataset)} samples")
    
    # Extract slot embeddings
    slots, labels = extract_slot_embeddings(
        adaslot, task1_loader, device, 
        max_batches=args.max_batches
    )
    
    # Test HDBSCAN parameters
    param_grid = {
        'min_cluster_size': args.min_cluster_sizes,
        'min_samples': args.min_samples_list
    }
    
    results = test_hdbscan_params(slots, labels, param_grid)
    
    # Plot parameter sweep
    plot_hdbscan_results(results, output_dir)
    
    # Visualize best few configurations
    logger.info("\nCreating visualizations for selected configurations...")
    
    # Sort by how close to ideal (10-20 clusters)
    results_sorted = sorted(results, 
                          key=lambda x: (abs(x['num_clusters'] - 15), x['noise_ratio']))
    
    # Visualize top 3
    for i, result in enumerate(results_sorted[:3]):
        title = (f" (min_cluster_size={result['min_cluster_size']}, "
                f"min_samples={result['min_samples']}, "
                f"{result['num_clusters']} clusters)")
        visualize_embeddings(slots, labels, result['cluster_labels'], 
                           output_dir, title_suffix=title)
    
    # Print recommendations
    print_recommendations(results)
    
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
