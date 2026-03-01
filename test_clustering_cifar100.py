"""
Test Clustering on CIFAR-100 Task 1 with AdaSlot Checkpoint

Load checkpoint, extract slot embeddings, cluster and visualize.
"""

from torch.utils.data import DataLoader
from avalanche.benchmarks.classic import SplitCIFAR100
import cont_src.models
from cont_src.clustering import CLUSTERING_REGISTRY, IdentityWrapper
from cont_src.core.registry import MODEL_REGISTRY
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# Avalanche for continual learning datasets


class SimpleCNNEncoder(nn.Module):
    """Simple CNN encoder matching training setup."""

    def __init__(self, output_dim=64):  # Match checkpoint feature_dim
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, output_dim, 3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        b, c, h, w = x.shape
        x = x.view(b, h*w, c)
        return x


def extract_slots(model, encoder, dataloader, device, max_batches=10):
    """
    Extract slot embeddings from data.

    Returns:
        slots: (N, slot_dim) tensor
        labels: (N,) array of class labels
    """
    model.eval()
    encoder.eval()

    all_slots = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting slots")):
            if i >= max_batches:
                break

            images = batch[0].to(device)
            labels = batch[1]

            # Extract features
            features = encoder(images)

            # Get slots
            outputs = model(features, global_step=0)
            slots = outputs['slots']  # (B, K, D)

            # Flatten: (B, K, D) -> (B*K, D)
            B, K, D = slots.shape
            slots_flat = slots.reshape(B * K, D)

            # Repeat labels for each slot
            labels_repeated = labels.unsqueeze(1).repeat(1, K).reshape(-1)

            all_slots.append(slots_flat.cpu())
            all_labels.append(labels_repeated)

    # Concatenate
    all_slots = torch.cat(all_slots, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_slots, all_labels


def test_clustering_algorithm(name, config, embeddings, true_labels, output_dir):
    """Test a single clustering algorithm and visualize."""
    print(f"\n{'='*80}")
    print(f"Testing: {name} - {config}")
    print(f"{'='*80}")

    # Build clustering
    try:
        clustering = CLUSTERING_REGISTRY.build(name, **config)
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # Fit and predict
    try:
        result = clustering.fit_predict(embeddings)
    except Exception as e:
        print(f"ERROR during clustering: {e}")
        return None

    # Print results
    print(f"\nResults:")
    print(f"  Samples: {len(embeddings)}")
    print(f"  True classes: {len(np.unique(true_labels))}")
    print(f"  Found clusters: {result.n_clusters}")
    print(f"  Noise points: {(result.labels == -1).sum()}")

    if result.scores:
        print(f"\nQuality Scores:")
        for metric, value in result.scores.items():
            print(f"  {metric}: {value:.4f}")

    # Visualize
    visualize_clusters(
        embeddings.numpy(),
        result.labels,
        true_labels,
        title=f"{name} - {result.n_clusters} clusters",
        save_path=output_dir /
        f"cluster_{name}_{config.get('eps', config.get('min_cluster_size', 'default'))}.png"
    )

    return result


def visualize_clusters(embeddings, pred_labels, true_labels, title, save_path):
    """
    Visualize clustering results with t-SNE.

    Left plot: Predicted clusters
    Right plot: True labels (for comparison)
    """
    from sklearn.manifold import TSNE

    print(f"  Computing t-SNE projection...")

    # Reduce to 2D with lower perplexity for faster computation
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(embeddings)//4), n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot predicted clusters
    ax = axes[0]
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=pred_labels,
        cmap='tab20',
        alpha=0.6,
        s=10
    )
    ax.set_title(f"{title}\n(Predicted Clusters)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    # Plot true labels
    ax = axes[1]
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=true_labels,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    ax.set_title("True Class Labels\n(Ground Truth)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax, label="Class")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    """Main test function."""
    print("="*80)
    print("CLUSTERING TEST: CIFAR-100 Task 1 + AdaSlot Checkpoint")
    print("="*80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path("clustering_results")
    output_dir.mkdir(exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Output dir: {output_dir}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Load CIFAR-100 using Avalanche
    benchmark = SplitCIFAR100(
        n_experiences=10,
        return_task_id=False,
        seed=42
    )

    # Get task 0 (first experience)
    task_0 = benchmark.train_stream[0]

    # Create dataloader
    dataloader = DataLoader(
        task_0.dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    print(f"Task 0 (classes 0-9):")
    print(f"  Samples: {len(task_0.dataset)}")
    print(f"  Classes: {task_0.classes_in_this_experience}")

    # Build model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)

    encoder = SimpleCNNEncoder(output_dim=64).to(
        device)  # Match checkpoint feature_dim!

    model = MODEL_REGISTRY.build(
        'adaslot_module',
        num_slots=7,
        slot_dim=64,
        feature_dim=64,  # Match checkpoint!
        kvq_dim=128,     # Match checkpoint!
        mlp_hidden_dim=128,  # Match checkpoint MLP
        num_iterations=3,
        use_gumbel=True,
        gumbel_low_bound=1,
        use_primitive=False,
        use_decoder=False,
        init_temperature=1.0,
        min_temperature=0.5,
        temperature_anneal_rate=0.00003,
    ).to(device)

    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"AdaSlot: {sum(p.numel() for p in model.parameters()):,} params")

    # Try to load checkpoint
    checkpoint_dir = Path("checkpoints/slot_attention/adaslot_real")
    checkpoint_path = checkpoint_dir / "CLEVR10.ckpt"

    if checkpoint_path.exists():
        print(f"\nAttempting to load checkpoint: {checkpoint_path}")
        try:
            # Load with weights_only=False
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False)
            checkpoint_state = checkpoint['state_dict']

            # Map checkpoint keys to model keys
            model_state = model.state_dict()
            mapped_state = {}

            for ckpt_key, value in checkpoint_state.items():
                model_key = ckpt_key

                # Map: models.conditioning.* -> slot_attention.*
                if model_key.startswith('models.conditioning.'):
                    model_key = model_key.replace(
                        'models.conditioning.', 'slot_attention.')

                # Map: models.perceptual_grouping.slot_attention.* -> slot_attention.*
                if model_key.startswith('models.perceptual_grouping.slot_attention.'):
                    model_key = model_key.replace(
                        'models.perceptual_grouping.slot_attention.', 'slot_attention.')

                # Rename: slots_logsigma -> slots_log_sigma
                model_key = model_key.replace(
                    'slots_logsigma', 'slots_log_sigma')

                # Map: ff_mlp.module.* -> mlp.* (remove module wrapper)
                model_key = model_key.replace('ff_mlp.module.', 'mlp.')

                # Map: single_gumbel_score_network.* -> gumbel_selector.score_net.*
                model_key = model_key.replace(
                    'single_gumbel_score_network.', 'gumbel_selector.score_net.')

                # Check if key exists and shapes match
                if model_key in model_state:
                    if model_state[model_key].shape == value.shape:
                        mapped_state[model_key] = value

            # Load mapped weights
            if mapped_state:
                model.load_state_dict(mapped_state, strict=False)
                print(
                    f"✓ Loaded {len(mapped_state)}/{len(model_state)} keys from checkpoint")
            else:
                print("✗ No keys matched, using random initialization")

        except Exception as e:
            print(f"✗ Could not load checkpoint: {e}")
            print("Using random initialization")
    else:
        print(f"\n✗ No checkpoint found at {checkpoint_path}")
        print("Using random initialization")

    # Extract slot embeddings
    print("\n" + "="*80)
    print("EXTRACTING SLOT EMBEDDINGS")
    print("="*80)

    slots, labels = extract_slots(
        model, encoder, dataloader, device, max_batches=10)

    print(f"\nExtracted:")
    print(f"  Slot embeddings: {slots.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Unique classes: {len(np.unique(labels))}")

    # Sample if too many for t-SNE
    if len(slots) > 5000:
        print(f"\nSampling 5000/{len(slots)} for faster visualization...")
        indices = np.random.choice(len(slots), 5000, replace=False)
        slots_vis = slots[indices]
        labels_vis = labels[indices]
    else:
        slots_vis = slots
        labels_vis = labels

    # Test clustering algorithms with various hyperparameters
    print("\n" + "="*80)
    print("TESTING CLUSTERING ALGORITHMS")
    print("="*80)

    algorithms = [
        # DBSCAN with different eps values
        ('dbscan', {'eps': 0.3, 'min_samples': 5}),
        ('dbscan', {'eps': 0.5, 'min_samples': 5}),
        ('dbscan', {'eps': 0.8, 'min_samples': 5}),
        ('dbscan', {'eps': 1.0, 'min_samples': 10}),
        ('dbscan', {'eps': 1.5, 'min_samples': 10}),

        # HDBSCAN with different min_cluster_size
        ('hdbscan', {'min_cluster_size': 20, 'min_samples': 5}),
        ('hdbscan', {'min_cluster_size': 50, 'min_samples': 10}),
        ('hdbscan', {'min_cluster_size': 100, 'min_samples': 15}),

        # Agglomerative with distance threshold
        ('agglomerative', {'distance_threshold': 5.0,
         'n_clusters': None, 'linkage': 'average'}),
        ('agglomerative', {'distance_threshold': 10.0,
         'n_clusters': None, 'linkage': 'average'}),
        ('agglomerative', {'distance_threshold': 15.0,
         'n_clusters': None, 'linkage': 'average'}),

        # Bayesian GMM (auto-pruning)
        ('bayesian_gmm', {'n_clusters': 20}),
        ('bayesian_gmm', {'n_clusters': 30}),

        # Baseline: K-means with true number of classes
        ('kmeans', {'n_clusters': 10}),
    ]

    results = {}

    try:
        import hdbscan
        hdbscan_available = True
    except ImportError:
        hdbscan_available = False
        print("\nWARNING: hdbscan not installed, skipping HDBSCAN tests")
        print("Install with: pip install hdbscan")

    for name, config in algorithms:
        # Skip HDBSCAN if not available
        if name == 'hdbscan' and not hdbscan_available:
            continue

        result = test_clustering_algorithm(
            name,
            config,
            slots_vis,
            labels_vis,
            output_dir
        )

        if result:
            key = f"{name}_{list(config.values())[0]}"
            results[key] = {
                'n_clusters': result.n_clusters,
                'scores': result.scores,
                'n_noise': int((result.labels == -1).sum())
            }

    # Summary
    print("\n" + "="*80)
    print("CLUSTERING SUMMARY")
    print("="*80)

    print(f"\nTrue number of classes: 10")
    print(f"\nResults sorted by cluster count:\n")

    for key in sorted(results.keys(), key=lambda k: results[k]['n_clusters']):
        res = results[key]
        print(
            f"{key:40s}: {res['n_clusters']:3d} clusters, {res['n_noise']:5d} noise points")
        if res['scores']:
            print(
                f"                                          Silhouette: {res['scores'].get('silhouette', 0):.3f}")

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\nVisualizations saved to: {output_dir}/")
    print("\nRecommendations:")
    print("  - Check visualizations to see cluster quality")
    print("  - Look for algorithms with ~10 clusters and high silhouette")
    print("  - Tune hyperparameters based on visual inspection")


if __name__ == '__main__':
    main()
