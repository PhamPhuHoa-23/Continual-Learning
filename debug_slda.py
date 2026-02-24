"""Debug SLDA and routing after training"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, 'src')
from compositional.pipeline import CompositionalRoutingPipeline
from models.adaslot.model import AdaSlotModel
from data.continual_cifar100 import get_continual_cifar100_loaders

# Load checkpoint
print("="*60)
print("Debugging SLDA and Routing")
print("="*60)

checkpoint_path = "checkpoints/compositional_runs/run_20260224_154601/pipeline_task1.pt"
adaslot_path = "checkpoints/compositional_runs/run_20260224_152646/adaslot_final.pt"

# Load data
train_loaders, test_loaders, _ = get_continual_cifar100_loaders(
    n_tasks=10, batch_size=8, num_workers=0, max_samples_per_task=50
)

# Load AdaSlot
adaslot = AdaSlotModel(num_slots=7, slot_dim=64)
ckpt = torch.load(adaslot_path, map_location='cuda', weights_only=False)
# Check if it's a state_dict directly or wrapped
if 'model' in ckpt:
    adaslot.load_state_dict(ckpt['model'])
else:
    adaslot.load_state_dict(ckpt)
adaslot.cuda()
adaslot.eval()

# Create and load pipeline
from train_compositional import load_pipeline

pipeline = CompositionalRoutingPipeline(
    slot_dim=64, d_h=128, latent_dim=32,
    theta_match=-50.0, b_min=20, n_min=5, rho_min=0.7,
    device='cuda'
)
load_pipeline(pipeline, checkpoint_path)

print(f"\n[Pipeline Status]")
print(f"  Agents: {len(pipeline.agents)}")
print(f"  VAEs: {len(pipeline.router.vaes)}")
print(f"  SLDA classes: {len(pipeline.slda.means)}")
print(f"  SLDA total samples: {pipeline.slda.total_count}")

# Check SLDA means
print(f"\n[SLDA Class Means (first 3 dims)]")
for cls_id, mean in list(pipeline.slda.means.items())[:5]:
    print(f"  Class {cls_id}: {mean[:3]} ... (norm: {(mean**2).sum()**0.5:.4f})")

# Test routing on Task 1 test data
print(f"\n[Testing on Task 1 test data]")
n_assigned = 0
n_unassigned = 0
agent_counts = {}

with torch.no_grad():
    for X, y in test_loaders[0]:
        X = X.cuda()
        X = F.interpolate(X, size=(128, 128), mode='bilinear', align_corners=False)
        slots = adaslot.encode(X).detach()  # (B, K, 64)
        
        B, K, _ = slots.shape
        slots_flat = slots.view(B*K, 64)
        sigma_flat = pipeline._route_slots(slots_flat)
        
        for aid in sigma_flat:
            if aid == "unassigned":
                n_unassigned += 1
            else:
                n_assigned += 1
                agent_counts[aid] = agent_counts.get(aid, 0) + 1
        
        break  # Just one batch

total = n_assigned + n_unassigned
print(f"  Assigned: {n_assigned}/{total} ({100*n_assigned/total:.1f}%)")
print(f"  Unassigned: {n_unassigned}/{total} ({100*n_unassigned/total:.1f}%)")
print(f"  Agents used: {len(agent_counts)} / {len(pipeline.agents)}")

# Show prediction distribution
print(f"\n[Predictions on Task 1 (1 batch)]")
with torch.no_grad():
    for X, y in test_loaders[0]:
        X = X.cuda()
        X = F.interpolate(X, size=(128, 128), mode='bilinear', align_corners=False)
        slots = adaslot.encode(X).detach()
        
        preds = pipeline.predict(slots)
        
        print(f"  True labels: {y.tolist()}")
        print(f"  Predictions: {preds.tolist()}")
        print(f"  Unique predictions: {set(preds.tolist())}")
        
        break

# Check hidden representations diversity
print(f"\n[Hidden Representation Diversity]")
with torch.no_grad():
    for X, y in test_loaders[0]:
        X = X.cuda()
        X = F.interpolate(X, size=(128, 128), mode='bilinear', align_corners=False)
        slots = adaslot.encode(X).detach()
        
        B, K, _ = slots.shape
        slots_flat = slots.view(B*K, 64)
        sigma_flat = pipeline._route_slots(slots_flat)
        sigma_2d = [sigma_flat[b*K:(b+1)*K] for b in range(B)]
        
        h = torch.zeros(B, K, 128, device='cuda')
        for b in range(B):
            for k in range(K):
                aid = sigma_2d[b][k]
                if aid != "unassigned" and aid in pipeline.agents:
                    h[b, k] = pipeline.agents[aid](slots[b, k].unsqueeze(0)).squeeze(0)
        
        H = pipeline.aggregator(h, sigma_2d)  # (B, 128)
        
        # Check if H is diverse
        H_norm = F.normalize(H, dim=1)
        sim_matrix = H_norm @ H_norm.T
        
        print(f"  H shape: {H.shape}")
        print(f"  H mean: {H.mean().item():.4f}, std: {H.std().item():.4f}")
        print(f"  H norm: min={H.norm(dim=1).min():.4f}, max={H.norm(dim=1).max():.4f}")
        print(f"  Pairwise similarity: min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}, mean={sim_matrix.mean():.4f}")
        
        if H.std() < 0.01:
            print(f"\n[WARNING] H has very low variance! All outputs similar!")
        
        break

print("\n" + "="*60)
