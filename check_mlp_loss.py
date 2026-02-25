import torch
from src.slot_multi_agent.estimators import MLPEstimator

def check():
    mlp = MLPEstimator(num_agents=50, slot_dim=64)
    # create dummy slot and agent id
    slot = torch.randn(10, 64)
    agent_id = 0
    score = mlp(slot, agent_id)
    print("Initial score stats: ", score.mean().item(), score.std().item())

    # dummy target
    target = torch.ones(10) * 0.9
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    for i in range(100):
        optimizer.zero_grad()
        score = mlp(slot, agent_id)
        loss = torch.nn.functional.mse_loss(score, target)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f"Step {i}: Loss: {loss.item():.4f}, Score: {score.mean().item():.4f}")

if __name__ == "__main__":
    check()
