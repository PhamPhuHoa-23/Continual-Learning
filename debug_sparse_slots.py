"""
Diagnostic: Tại sao sparse loss không drop slots?
Runs locally với CLEVR10.ckpt + random CIFAR-like tensors.
"""
from src.models.adaslot.model import AdaSlotModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

CKPT = "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
B = 8
NUM_SLOTS = 11

print(f"Device: {DEVICE}")

# -----------------------------------------------------------------------
# 1. Load model
# -----------------------------------------------------------------------

backbone = AdaSlotModel(
    resolution=(IMG_SIZE, IMG_SIZE),
    num_slots=NUM_SLOTS,
    slot_dim=64,
    num_iterations=3,
    feature_dim=64,
    kvq_dim=128,
    low_bound=1,
).to(DEVICE)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
backbone.load_state_dict(ckpt["state_dict"], strict=True)
print(f"Checkpoint loaded OK  (num_slots={NUM_SLOTS})")

# -----------------------------------------------------------------------
# 2. Forward pass — check raw hard_keep_decision values
# -----------------------------------------------------------------------
backbone.eval()
imgs = torch.randn(B, 3, IMG_SIZE, IMG_SIZE,
                   device=DEVICE)  # random CIFAR-like

with torch.no_grad():
    out = backbone(imgs)

hkd = out["hard_keep_decision"]   # (B, K) — binary {0, 1}
skp = out["slots_keep_prob"]      # (B, K) — soft [0, 1]
recon = out["reconstruction"]

print(f"\n--- Forward pass stats (no grad, random input) ---")
print(f"hard_keep_decision  : shape={hkd.shape}  mean={hkd.mean():.3f}  "
      f"sum/sample={hkd.sum(dim=1).float().mean():.2f}/{NUM_SLOTS}")
print(
    f"slots_keep_prob     : min={skp.min():.4f}  max={skp.max():.4f}  mean={skp.mean():.4f}")

recon_loss = F.mse_loss(recon, imgs, reduction="sum") / B
print(f"\nrecon_loss (mse_sum): {recon_loss.item():.2f}")
print(f"  → same with mean  : {F.mse_loss(recon, imgs).item():.6f}")
print(
    f"  → scale ratio     : {recon_loss.item() / F.mse_loss(recon, imgs).item():.0f}×")

# -----------------------------------------------------------------------
# 3. Gradient experiment — compare grad magnitudes at gumbel gate
# -----------------------------------------------------------------------
backbone.train()

# Find the gumbel score network
gumbel_net = None
for name, mod in backbone.named_modules():
    if "gumbel_score" in name.lower() or "single_gumbel" in name.lower():
        gumbel_net = mod
        gumbel_name = name
        break

if gumbel_net is None:
    print("\nCould not find gumbel score network — trying perceptual_grouping")
    for name, mod in backbone.named_modules():
        print(f"  {name}")
else:
    print(f"\nGumbel gate module: {gumbel_name}")

print("\n--- Gradient experiment (w_sparse sweep) ---")
print(f"{'w_sparse':>10} | {'grad_gate':>12} | {'grad_ratio':>12} | {'active_slots':>12}")
print("-" * 55)

for w_sparse in [0.0, 1.0, 10.0, 100.0, 1000.0]:
    backbone.zero_grad()
    imgs2 = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    out2 = backbone(imgs2)
    hkd2 = out2["hard_keep_decision"]
    recon2 = out2["reconstruction"]

    l_recon = 1.0 * F.mse_loss(recon2, imgs2, reduction="sum") / B
    l_sparse = w_sparse * hkd2.mean()
    loss = l_recon + l_sparse
    loss.backward()

    # Grab gradient at gumbel gate (first param)
    gate_grad = None
    if gumbel_net is not None:
        for p in gumbel_net.parameters():
            if p.grad is not None:
                gate_grad = p.grad.abs().mean().item()
                break

    # Compare: grad from recon alone vs sparse
    backbone.zero_grad()
    l_recon_only = 1.0 * F.mse_loss(
        backbone(imgs2)["reconstruction"], imgs2, reduction="sum") / B
    l_recon_only.backward()
    recon_gate_grad = None
    if gumbel_net is not None:
        for p in gumbel_net.parameters():
            if p.grad is not None:
                recon_gate_grad = p.grad.abs().mean().item()
                break

    n_active = (hkd2 > 0.5).float().sum(dim=1).mean().item()
    ratio = (
        gate_grad / recon_gate_grad) if (gate_grad and recon_gate_grad) else float("nan")
    gstr = f"{gate_grad:.3e}" if gate_grad else "N/A"
    print(f"{w_sparse:>10.1f} | {gstr:>12} | {ratio:>12.2f} | {n_active:>12.1f}/{NUM_SLOTS}")

# -----------------------------------------------------------------------
# 4. Key verdict
# -----------------------------------------------------------------------
print("\n--- VERDICT ---")
backbone.zero_grad()
imgs3 = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
out3 = backbone(imgs3)
l_recon_v = F.mse_loss(out3["reconstruction"], imgs3, reduction="sum") / B
l_sparse_v = out3["hard_keep_decision"].mean()
print(f"w_recon=1.0   → raw L_recon   = {l_recon_v.item():.2f}")
print(
    f"w_sparse=1    → raw L_sparse  = {l_sparse_v.item():.4f}  (= mean active frac)")
print(f"\nTo balance: need w_sparse ≈ {l_recon_v.item():.0f}× to equal recon")
print(f"Per-slot recon contribution ≈ {l_recon_v.item()/NUM_SLOTS:.2f}")
print(f"Per-slot sparse contribution (w=100) ≈ {100.0/NUM_SLOTS:.2f}")
print(
    f"\n→ Recon dominates by {l_recon_v.item()/NUM_SLOTS / (100.0/NUM_SLOTS):.1f}×  at w_sparse=100")
print(f"→ Need w_sparse ≈ {l_recon_v.item():.0f} to break even")

# -----------------------------------------------------------------------
# 5. Test: what happens if we use mse MEAN instead of mse_sum?
# -----------------------------------------------------------------------
print("\n--- If w_recon = 1/(C*H*W) (normalized) ---")
w_norm = 1.0 / (3 * IMG_SIZE * IMG_SIZE)
l_recon_norm = w_norm * l_recon_v
print(f"w_recon = {w_norm:.2e}")
print(f"normalized L_recon = {l_recon_norm.item():.6f}  (≈ per-pixel MSE)")
print(f"at w_sparse=10 → L_sparse = {10 * l_sparse_v.item():.4f}")
print(
    f"→ sparse/recon ratio = {10 * l_sparse_v.item() / l_recon_norm.item():.1f}×")
print(
    f"\nWith w_recon=1/(C·H·W), w_sparse=10 → sparse already {10 * l_sparse_v.item() / l_recon_norm.item():.0f}× bigger than recon!")
print("→ FIX: Set P0_W_RECON = 1.0 and w_sparse should be relative to mse_sum scale.")
print(
    f"   Recommended w_sparse for CLEVR10 result (original linear_weight=10 with mse_sum): ~{10:.0f}")
print(f"   Your current P0_W_SPARSE = 0.0 (correct for phase 0!)")
print(
    f"   For post-phase-0 sparse: w_sparse ~ {l_recon_v.item() * 0.1:.0f} to drop ~10% slots")
