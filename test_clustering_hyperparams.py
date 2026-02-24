"""
Test HDBSCAN/DBSCAN hyperparameters and visualize slot embeddings
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from models.adaslot.model import AdaSlotModel
from data.continual_cifar100 import get_continual_cifar100_loaders

print("="*80)
print("Testing Clustering Hyperparameters for Slot Embeddings")
print("="*80)

# ============================================
# 1. Load AdaSlot checkpoint
# ============================================
print("\n[1] Loading AdaSlot...")

# Try different checkpoints
checkpoint_options = [
    ("Trained", "checkpoints/compositional_runs/run_20260224_152646/adaslot_final.pt"),
    ("Pretrained COCO", "checkpoints/slot_attention/adaslot_real/COCO.ckpt"),
    ("Pretrained CLEVR10", "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"),
]

adaslot = None
for name, path in checkpoint_options:
    try:
        print(f"  Trying {name}: {path}")
        adaslot = AdaSlotModel(num_slots=7, slot_dim=64)
        ckpt = torch.load(path, map_location='cuda', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in ckpt:
            adaslot.load_state_dict(ckpt['model'])
        elif 'state_dict' in ckpt:
            adaslot.load_state_dict(ckpt['state_dict'])
        else:
            adaslot.load_state_dict(ckpt)
        
        print(f"  ✓ Loaded {name}")
        break
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        continue

if adaslot is None:
    print("ERROR: Could not load any AdaSlot checkpoint!")
    sys.exit(1)

adaslot.cuda()
adaslot.eval()

# ============================================
# 2. Extract slots from Task 1 training data
# ============================================
print("\n[2] Extracting slots from Task 1 training data...")

train_loaders, _, class_order = get_continual_cifar100_loaders(
    n_tasks=10, batch_size=32, num_workers=0, max_samples_per_task=200
)

all_slots = []
all_labels = []

with torch.no_grad():
    for X, y in train_loaders[0]:
        X = X.cuda()
        X = F.interpolate(X, size=(128, 128), mode='bilinear', align_corners=False)
        slots = adaslot.encode(X).detach().cpu()  # (B, K, 64)
        
        B, K, D = slots.shape
        slots_flat = slots.view(B*K, D)
        all_slots.append(slots_flat)
        
        # Repeat labels for each slot
        labels_repeated = y.unsqueeze(1).repeat(1, K).view(-1)
        all_labels.append(labels_repeated)

all_slots = torch.cat(all_slots, dim=0).numpy()  # (N, 64)
all_labels = torch.cat(all_labels, dim=0).numpy()  # (N,)

print(f"  Total slots: {all_slots.shape[0]}")
print(f"  Slot dimension: {all_slots.shape[1]}")
print(f"  Unique labels: {len(np.unique(all_labels))}")

# ============================================
# 3. Test different clustering settings
# ============================================
print("\n[3] Testing clustering with different hyperparameters...")

try:
    from hdbscan import HDBSCAN
    has_hdbscan = True
except ImportError:
    from sklearn.cluster import DBSCAN
    has_hdbscan = False
    print("  Note: hdbscan not available, using sklearn DBSCAN")

# Test configurations
configs = [
    {"min_cluster_size": 50, "min_samples": 20, "name": "Very Conservative"},
    {"min_cluster_size": 30, "min_samples": 15, "name": "Conservative"},
    {"min_cluster_size": 20, "min_samples": 10, "name": "Moderate"},
    {"min_cluster_size": 15, "min_samples": 8, "name": "Balanced"},
    {"min_cluster_size": 10, "min_samples": 5, "name": "Permissive"},
    {"min_cluster_size": 5, "min_samples": 3, "name": "Very Permissive"},
]

results = []

for cfg in configs:
    try:
        if has_hdbscan:
            clusterer = HDBSCAN(
                min_cluster_size=cfg["min_cluster_size"],
                min_samples=cfg["min_samples"],
                metric='euclidean',
                cluster_selection_method='eom',
            )
        else:
            # DBSCAN fallback
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=cfg["min_samples"]).fit(all_slots)
            distances, _ = nbrs.kneighbors(all_slots)
            eps = np.percentile(distances[:, -1], 50)
            clusterer = DBSCAN(eps=eps, min_samples=cfg["min_samples"])
        
        labels = clusterer.fit_predict(all_slots)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        # Calculate samples per cluster
        cluster_ids = [c for c in set(labels) if c != -1]
        if cluster_ids:
            cluster_sizes = [np.sum(labels == c) for c in cluster_ids]
            avg_size = np.mean(cluster_sizes)
            min_size = np.min(cluster_sizes)
            max_size = np.max(cluster_sizes)
        else:
            avg_size = min_size = max_size = 0
        
        results.append({
            "name": cfg["name"],
            "min_cluster_size": cfg["min_cluster_size"],
            "min_samples": cfg["min_samples"],
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": 100 * n_noise / len(labels),
            "avg_size": avg_size,
            "min_size": min_size,
            "max_size": max_size,
            "labels": labels
        })
        
        print(f"\n  {cfg['name']:20s} (min_size={cfg['min_cluster_size']}, min_samples={cfg['min_samples']})")
        print(f"    → Agents: {n_clusters}, Noise: {n_noise}/{len(labels)} ({100*n_noise/len(labels):.1f}%)")
        if n_clusters > 0:
            print(f"    → Samples/agent: avg={avg_size:.1f}, min={min_size}, max={max_size}")
            if min_size < 10:
                print(f"    → ⚠️  Some clusters too small to train VAE!")
        
    except Exception as e:
        print(f"  {cfg['name']}: Failed - {e}")

# ============================================
# 4. Visualization with PCA and t-SNE
# ============================================
print("\n[4] Generating visualizations...")

# Subsample for t-SNE (too slow for 1400 points)
n_vis = min(1000, all_slots.shape[0])
indices = np.random.choice(all_slots.shape[0], n_vis, replace=False)
slots_vis = all_slots[indices]
labels_vis = all_labels[indices]

# PCA
print("  Computing PCA...")
pca = PCA(n_components=2)
slots_pca = pca.fit_transform(slots_vis)

# t-SNE
print("  Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
slots_tsne = tsne.fit_transform(slots_vis)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Slot Embedding Visualization (Task 1)', fontsize=16)

# Row 1: PCA and t-SNE colored by true labels
ax = axes[0, 0]
scatter = ax.scatter(slots_pca[:, 0], slots_pca[:, 1], c=labels_vis, cmap='tab10', s=10, alpha=0.6)
ax.set_title('PCA - Colored by True Labels')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
plt.colorbar(scatter, ax=ax)

ax = axes[0, 1]
scatter = ax.scatter(slots_tsne[:, 0], slots_tsne[:, 1], c=labels_vis, cmap='tab10', s=10, alpha=0.6)
ax.set_title('t-SNE - Colored by True Labels')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax)

# Row 1, Col 3: Summary table
ax = axes[0, 2]
ax.axis('off')
summary_text = "Clustering Results Summary\n" + "="*40 + "\n\n"
for r in results[:3]:  # Top 3
    summary_text += f"{r['name']}:\n"
    summary_text += f"  Agents: {r['n_clusters']}\n"
    summary_text += f"  Noise: {r['noise_pct']:.1f}%\n"
    if r['n_clusters'] > 0:
        summary_text += f"  Avg size: {r['avg_size']:.1f}\n"
        summary_text += f"  Min size: {r['min_size']}\n\n"
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace')

# Row 2: Best 3 clustering results
best_results = sorted(results, key=lambda x: x['n_clusters'] if x['n_clusters'] > 0 and x['min_size'] >= 10 else 999)[:3]

for idx, result in enumerate(best_results):
    if idx >= 3:
        break
    
    # Get cluster labels for visualization indices
    cluster_labels_vis = result['labels'][indices]
    
    # Plot on t-SNE
    ax = axes[1, idx]
    
    # Color each cluster differently
    unique_clusters = [c for c in np.unique(cluster_labels_vis) if c != -1]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot noise points first (gray)
    noise_mask = cluster_labels_vis == -1
    if noise_mask.any():
        ax.scatter(slots_tsne[noise_mask, 0], slots_tsne[noise_mask, 1], 
                  c='lightgray', s=10, alpha=0.3, label='Noise')
    
    # Plot clusters
    for cluster_id, color in zip(unique_clusters, colors):
        mask = cluster_labels_vis == cluster_id
        ax.scatter(slots_tsne[mask, 0], slots_tsne[mask, 1],
                  c=[color], s=15, alpha=0.7, label=f'C{cluster_id}')
    
    ax.set_title(f'{result["name"]}\n{result["n_clusters"]} agents, {result["noise_pct"]:.0f}% noise')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    if idx == 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

plt.tight_layout()
output_path = 'slot_clustering_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# ============================================
# 5. Recommendations
# ============================================
print("\n[5] Recommendations:")
print("="*80)

viable = [r for r in results if r['n_clusters'] > 0 and r['min_size'] >= 10]
if viable:
    best = min(viable, key=lambda x: abs(x['n_clusters'] - 8))  # Target ~8 agents
    print(f"\n✓ RECOMMENDED: {best['name']}")
    print(f"  min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']}")
    print(f"  Will spawn {best['n_clusters']} agents")
    print(f"  Each agent has {best['min_size']}-{best['max_size']} samples (avg: {best['avg_size']:.1f})")
    print(f"  Noise: {best['n_noise']} slots ({best['noise_pct']:.1f}%)")
    print(f"\n  Command:")
    print(f"  python train_compositional.py --phase task1 \\")
    print(f"    --min_cluster_size {best['min_cluster_size']} \\")
    print(f"    --min_samples_cluster {best['min_samples']} \\")
    print(f"    --test_mode --max_samples 200")
else:
    print("\n⚠️  No configuration produced viable clusters!")
    print("  Try increasing --max_samples (more data needed)")

print("\n" + "="*80)
