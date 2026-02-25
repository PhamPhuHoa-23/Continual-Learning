"""
Focused test for the vae.score() (B,K,D) reshape fix in Phase A, B, and SLDA routing.
Does NOT train — just verifies shapes are correct through the routing paths.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────────
B, K, D = 8, 7, 64   # batch=8, num_slots=7, slot_dim=64
N_AGENTS = 3

print(f"Testing with B={B} K={K} D={D} N_AGENTS={N_AGENTS}")

# ── Build fake VAEs and agents ────────────────────────────────────────────────
from cont_src.models.routers.slot_vae import SlotVAE
from cont_src.models.agents.residual_mlp_agent import ResidualMLPAgent

vaes = []
for i in range(N_AGENTS):
    vae = SlotVAE(slot_dim=D, latent_dim=16, device="cpu")
    # Train on small random data so stats are populated
    fake_slots = torch.randn(50, D)
    vae.train_vae(fake_slots, epochs=5, verbose=False)
    vae.update_stats(fake_slots)
    vaes.append(vae)

agents = [ResidualMLPAgent(input_dim=D, output_dim=D, num_blocks=2).to("cpu")
          for _ in range(N_AGENTS)]

slots = torch.randn(B, K, D)  # the exact shape that was crashing

# ── Test 1: Phase A routing ───────────────────────────────────────────────────
print("\n[1] Phase A _route()")
from cont_src.training.configs import PhaseAConfig
from cont_src.training.agent_phase_a import AgentPhaseATrainer

# Build a dummy slot model
class DummySlotModel(nn.Module):
    def forward(self, imgs, **kw):
        return {"slots": torch.randn(imgs.shape[0], K, D),
                "hard_keep_decision": torch.ones(imgs.shape[0], K)}

cfg_a = PhaseAConfig(max_steps=1, lr=1e-3)
trainer_a = AgentPhaseATrainer(cfg_a, DummySlotModel(), vaes, agents)
assignments = trainer_a._route(slots)
assert assignments.shape == (B, K), f"Expected ({B},{K}), got {assignments.shape}"
assert assignments.dtype == torch.long
assert assignments.min() >= 0 and assignments.max() < N_AGENTS
print(f"  assignments.shape={tuple(assignments.shape)}  range=[{assignments.min()},{assignments.max()}]  PASS")

# ── Test 2: Phase B soft weights ─────────────────────────────────────────────
print("\n[2] Phase B _soft_weights()")
from cont_src.training.configs import PhaseBConfig
from cont_src.training.agent_phase_b import AgentPhaseBTrainer

class DummySlotModel2(nn.Module):
    def forward(self, imgs, **kw):
        return {"slots": torch.randn(imgs.shape[0], K, D),
                "hard_keep_decision": torch.ones(imgs.shape[0], K),
                "recon": torch.randn(imgs.shape[0], 3, 128, 128),
                "mask": torch.ones(imgs.shape[0], K)}

cfg_b = PhaseBConfig(max_steps=1, lr=1e-3)
trainer_b = AgentPhaseBTrainer(cfg_b, DummySlotModel2(), vaes, agents)
weights = trainer_b._soft_route(slots, temperature=1.0)
assert weights.shape == (B, K, N_AGENTS), f"Expected ({B},{K},{N_AGENTS}), got {weights.shape}"
assert abs(weights.sum(dim=-1).mean().item() - 1.0) < 1e-4, "weights should sum to 1 over agents"
print(f"  weights.shape={tuple(weights.shape)}  sum_over_agents≈{weights.sum(dim=-1).mean():.4f}  PASS")

# ── Test 3: SLDA routing (raw code path, no full trainer needed) ─────────────
print("\n[3] SLDA routing (direct code path)")

with torch.no_grad():
    B2, K2, D2 = slots.shape
    slots_flat = slots.reshape(B2 * K2, D2)
    scores = torch.stack([vae.score(slots_flat) for vae in vaes], dim=-1)
    scores = scores.reshape(B2, K2, len(vaes))
    assignments_slda = scores.argmax(dim=-1)

assert assignments_slda.shape == (B, K), f"Expected ({B},{K}), got {assignments_slda.shape}"
print(f"  assignments.shape={tuple(assignments_slda.shape)}  PASS")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("ALL ROUTING TESTS PASSED")
print("="*50)
