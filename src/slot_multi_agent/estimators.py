"""
Performance Estimators: Sub-networks that estimate agent performance on slots.

Two types:
1. VAE-based: Uses reconstruction error as difficulty estimate
2. MLP-based: Direct learned mapping slot + agent_id → performance score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VAEEstimator(nn.Module):
    """
    VAE-based performance estimator.
    
    Idea: Reconstruction error indicates slot difficulty/unusualness.
    High error → difficult slot → might need specialized agent.
    
    This is a LIGHTWEIGHT VAE (encoder + decoder), not the full VAE from models/vae.
    """
    
    def __init__(
        self,
        agent_id: int,
        slot_dim: int = 64,
        latent_dim: int = 16,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.agent_id = agent_id
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim
        
        # Encoder: slot → latent
        self.encoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: latent → slot
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
        # Learnable threshold for this agent
        self.threshold = nn.Parameter(torch.tensor(1.0))
    
    def encode(self, slot: torch.Tensor):
        """Encode slot to latent distribution."""
        h = self.encoder(slot)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor):
        """Decode latent to reconstruction."""
        return self.decoder(z)
    
    def forward(self, slot: torch.Tensor):
        """
        Forward pass through VAE.
        
        Args:
            slot: (batch_size, slot_dim) or (slot_dim,)
            
        Returns:
            recon: Reconstruction
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(slot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def estimate_performance(self, slot: torch.Tensor) -> torch.Tensor:
        """
        Estimate performance score based on reconstruction error.
        
        Logic:
        - Low reconstruction error → slot is "familiar" → high confidence
        - High reconstruction error → slot is unusual → low confidence
        
        Args:
            slot: (batch_size, slot_dim) or (slot_dim,)
            
        Returns:
            score: (batch_size,) or scalar - performance estimate in [0, 1]
                Higher score = better expected performance
        """
        with torch.no_grad():
            recon, _, _ = self.forward(slot)
            
            # Reconstruction error
            if slot.dim() == 1:
                error = F.mse_loss(recon, slot, reduction='none').mean()
            else:
                error = F.mse_loss(recon, slot, reduction='none').mean(dim=-1)
            
            # Convert error to score
            # Use learned threshold
            score = torch.sigmoid(self.threshold - error)
            
        return score
    
    def compute_loss(self, slot: torch.Tensor, beta: float = 1.0):
        """
        Compute VAE loss for training.
        
        Args:
            slot: (batch_size, slot_dim)
            beta: Weight for KL divergence
            
        Returns:
            loss: Scalar loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
        """
        recon, mu, logvar = self.forward(slot)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, slot, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return loss, recon_loss, kl_loss


class MLPEstimator(nn.Module):
    """
    MLP-based performance estimator.
    
    Direct learned mapping: (slot, agent_id) → performance score
    
    This is trained supervised with historical data:
    - Input: slot representation + agent embedding
    - Output: predicted performance score
    - Ground truth: measured performance from running agent
    """
    
    def __init__(
        self,
        num_agents: int = 50,
        slot_dim: int = 64,
        hidden_dim: int = 128,
        agent_embed_dim: int = 32
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.slot_dim = slot_dim
        self.agent_embed_dim = agent_embed_dim
        
        # Agent embeddings (learnable)
        self.agent_embeddings = nn.Embedding(num_agents, agent_embed_dim)
        
        # MLP: [slot, agent_embedding] → score
        input_dim = slot_dim + agent_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, slot: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Predict performance score.
        
        Args:
            slot: (batch_size, slot_dim) or (slot_dim,)
            agent_id: Integer agent ID
            
        Returns:
            score: (batch_size,) or scalar - performance estimate in [0, 1]
        """
        # Get agent embedding
        agent_id_tensor = torch.tensor([agent_id], device=slot.device)
        agent_embed = self.agent_embeddings(agent_id_tensor)  # (1, agent_embed_dim)
        
        # Handle batch or single input
        if slot.dim() == 1:
            # Single slot
            slot = slot.unsqueeze(0)  # (1, slot_dim)
            single_input = True
        else:
            single_input = False
        
        # Expand agent embedding to match batch size
        batch_size = slot.size(0)
        agent_embed = agent_embed.expand(batch_size, -1)  # (batch_size, agent_embed_dim)
        
        # Concatenate
        combined = torch.cat([slot, agent_embed], dim=-1)  # (batch_size, slot_dim + agent_embed_dim)
        
        # Predict
        score = self.mlp(combined).squeeze(-1)  # (batch_size,)
        
        if single_input:
            score = score.squeeze(0)  # Scalar
        
        return score
    
    def estimate_performance(self, slot: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Estimate performance score (same as forward for consistency).
        
        Args:
            slot: (batch_size, slot_dim) or (slot_dim,)
            agent_id: Integer agent ID
            
        Returns:
            score: (batch_size,) or scalar
        """
        with torch.no_grad():
            return self.forward(slot, agent_id)
    
    def compute_loss(
        self,
        slot: torch.Tensor,
        agent_id: int,
        true_performance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised loss.
        
        Args:
            slot: (batch_size, slot_dim)
            agent_id: Integer agent ID
            true_performance: (batch_size,) - ground truth performance scores
            
        Returns:
            loss: MSE between predicted and true performance
        """
        pred_performance = self.forward(slot, agent_id)
        loss = F.mse_loss(pred_performance, true_performance)
        return loss


class HybridEstimator(nn.Module):
    """
    Hybrid estimator: Combines VAE and MLP predictions.
    
    Idea: VAE captures difficulty, MLP captures agent-slot compatibility.
    """
    
    def __init__(
        self,
        agent_id: int,
        num_agents: int = 50,
        slot_dim: int = 64,
        vae_weight: float = 0.5
    ):
        super().__init__()
        
        self.agent_id = agent_id
        self.vae_weight = vae_weight
        
        # VAE estimator (per-agent)
        self.vae_estimator = VAEEstimator(agent_id=agent_id, slot_dim=slot_dim)
        
        # MLP estimator (shared across all agents, conditioned on agent_id)
        # Note: This should be shared, so instantiate separately and pass in
        self.mlp_estimator = None  # Set externally
    
    def set_mlp_estimator(self, mlp_estimator: MLPEstimator):
        """Set shared MLP estimator."""
        self.mlp_estimator = mlp_estimator
    
    def estimate_performance(self, slot: torch.Tensor) -> torch.Tensor:
        """
        Combine VAE and MLP estimates.
        
        Args:
            slot: (batch_size, slot_dim) or (slot_dim,)
            
        Returns:
            score: Combined performance estimate
        """
        # VAE score
        vae_score = self.vae_estimator.estimate_performance(slot)
        
        # MLP score
        if self.mlp_estimator is not None:
            mlp_score = self.mlp_estimator.estimate_performance(slot, self.agent_id)
        else:
            mlp_score = 0.5  # Default if MLP not set
        
        # Weighted combination
        combined_score = (
            self.vae_weight * vae_score +
            (1 - self.vae_weight) * mlp_score
        )
        
        return combined_score

