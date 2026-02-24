import torch
from src.compositional.pipeline import CompositionalRoutingPipeline

def test_pipeline_grads():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")
    
    # Init pipeline
    loss_weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}
    pipeline = CompositionalRoutingPipeline(
        slot_dim=64, d_h=128, latent_dim=32, 
        theta_match=100.0, theta_novel=200.0, 
        b_min=2, rho_min=0.5, 
        loss_weights=loss_weights, device=device
    )
    
    # Spawn 2 agents
    pipeline._spawn_agent("agent_0")
    pipeline._spawn_agent("agent_1")
    
    # Verify key device
    print("agent_0 key device:", pipeline.aggregator.keys["agent_0"].device)
    
    # Fake inputs
    B, K, slot_dim = 2, 4, 64
    slots = torch.randn(B, K, slot_dim, device=device, requires_grad=True)
    y = torch.tensor([0, 1], device=device)
    
    # Fake routing: everything to agent_0
    pipeline._route_slots = lambda s: ["agent_0"] * (B * K)
    
    # Forward + losses
    losses = pipeline.compute_losses(slots, y, task_id=1)
    print("Losses before backward:", {k: v.item() for k, v in losses.items()})
    
    # Backward
    losses["total"].backward()
    
    # Check grads
    print("key grad:", pipeline.aggregator.keys["agent_0"].grad is not None)
    
test_pipeline_grads()
