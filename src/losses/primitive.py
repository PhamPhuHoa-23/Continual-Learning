"""
Primitive Loss for Slot Attention Training (CompSLOT Paper)

This implements the primitive selection and contrastive primitive loss
from the CompSLOT paper (Section 4.1, Equation 2-3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveSelector(nn.Module):
    """
    Learnable attention-based primitive selection mechanism.
    
    Aggregates K slots into a single primitive representation using
    a learnable primitive key K_p that measures slot significance.
    
    This is Equation 2 from the paper:
        S̄ = tanh(Linear(LN(S)))
        w_p = σ(τ_t * S̄ @ K_p)
        s_p = w_p^T @ S̄
    
    Args:
        slot_dim: Dimension of slot embeddings
        temperature: Temperature for slot selection sparsity (default: 100/√D_s)
    """
    
    def __init__(self, slot_dim: int, temperature: float = None):
        super().__init__()
        self.slot_dim = slot_dim
        
        # Default temperature from paper: 100/√D_s
        if temperature is None:
            temperature = 100.0 / (slot_dim ** 0.5)
        self.temperature = temperature
        
        # Layer norm
        self.ln = nn.LayerNorm(slot_dim)
        
        # Linear projection to unified similarity space
        self.linear = nn.Linear(slot_dim, slot_dim)
        
        # Learnable primitive key K_p
        self.primitive_key = nn.Parameter(torch.randn(slot_dim))
        nn.init.normal_(self.primitive_key, mean=0.0, std=0.02)
    
    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select and aggregate slots into primitive representation.
        
        Args:
            slots: (B, K, slot_dim) - slot embeddings
        
        Returns:
            primitives: (B, slot_dim) - aggregated primitive per image
            weights: (B, K) - selection weights for visualization
        """
        # Map slots to similarity space
        # S̄ = tanh(Linear(LN(S)))
        slots_norm = self.ln(slots)  # (B, K, D)
        slots_mapped = torch.tanh(self.linear(slots_norm))  # (B, K, D)
        
        # Compute similarity to primitive key
        # sim = S̄ @ K_p
        similarity = torch.matmul(
            slots_mapped, self.primitive_key
        )  # (B, K)
        
        # Apply temperature-scaled softmax
        # w_p = σ(τ_t * sim)
        weights = F.softmax(self.temperature * similarity, dim=1)  # (B, K)
        
        # Weighted aggregation
        # s_p = w_p^T @ S̄
        primitives = torch.einsum('bk,bkd->bd', weights, slots_mapped)  # (B, D)
        
        return primitives, weights


class PrimitiveLoss(nn.Module):
    """
    Contrastive Primitive Loss (CompSLOT Paper, Equation 3).
    
    Enforces intra-class consistency: primitives from images of the same
    class should be similar, and different classes should be dissimilar.
    
    Uses KL divergence between:
    - Label-based similarity d^y (ground truth)
    - Primitive similarity d^s (learned)
    
    This is different from standard contrastive loss—it uses KL divergence
    to match similarity distributions rather than pulling/pushing embeddings.
    
    Args:
        temperature: Temperature for primitive similarity (default: 0.1)
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        primitives: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between label and primitive similarities.
        
        Args:
            primitives: (B, D) - primitive representations
            labels: (B,) - class labels
        
        Returns:
            loss: Scalar KL divergence loss
        """
        B = primitives.size(0)
        device = primitives.device
        
        # ─── Step 1: Label similarity (ground truth) ───
        # d^y_i,j = sim(I_i, I_j) / Σ_k sim(I_i, I_k)
        # where sim(I_i, I_j) = 1 if same class, else 0
        
        labels = labels.contiguous().view(-1, 1)  # (B, 1)
        label_mask = (labels == labels.T).float()  # (B, B)
        
        # Normalize each row to sum to 1
        label_sim = label_mask / (label_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, B)
        
        # ─── Step 2: Primitive similarity (learned) ───
        # d^s_i,j = exp(τ_p * sim(s^p_i, s^p_j)) / Σ_k exp(τ_p * sim(s^p_i, s^p_k))
        
        # Normalize primitives
        primitives_norm = F.normalize(primitives, dim=1)  # (B, D)
        
        # Cosine similarity matrix
        cos_sim = torch.matmul(primitives_norm, primitives_norm.T)  # (B, B)
        
        # Temperature-scaled softmax
        primitive_sim = F.softmax(self.temperature * cos_sim, dim=1)  # (B, B)
        
        # ─── Step 3: KL Divergence ───
        # L_p = Σ_i Σ_j d^y_i,j * log(d^y_i,j / d^s_i,j)
        
        # Avoid log(0) and division by zero
        label_sim = torch.clamp(label_sim, min=1e-8)
        primitive_sim = torch.clamp(primitive_sim, min=1e-8)
        
        # KL divergence (mean over batch for balanced gradients)
        # Sum over all B×B pairs, then divide by B to normalize
        kl_div = (label_sim * torch.log(label_sim / primitive_sim)).sum() / B
        
        return kl_div


class SlotReconstructionLoss(nn.Module):
    """
    MSE reconstruction loss for slot attention (paper Section 4.1, Equation 1).
    
    Reconstructs patch features from position-augmented slots.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        reconstructed: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            reconstructed: (B, N, D) or (B, D, H, W)
            target: (B, N, D) or (B, D, H, W)
        
        Returns:
            loss: MSE loss (mean per pixel for balanced gradients)
        """
        return F.mse_loss(reconstructed, target, reduction='mean')


# ─── Combined Loss for Concept Learning ────────────────────────────────

class ConceptLearningLoss(nn.Module):
    """
    Combined loss for concept learning stage:
        L_slot = L_re + α * L_p
    
    Where:
        - L_re: Reconstruction loss
        - L_p: Primitive loss (intra-class consistency)
        - α: Weight coefficient
    
    Args:
        alpha: Weight for primitive loss (default: 10, from paper Table 4)
        temperature_p: Temperature for primitive similarity (default: 10)
    """
    
    def __init__(self, alpha: float = 10.0, temperature_p: float = 10.0):
        super().__init__()
        self.alpha = alpha
        
        self.recon_loss = SlotReconstructionLoss()
        self.primitive_loss = PrimitiveLoss(temperature=temperature_p)
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        primitives: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Compute combined concept learning loss.
        
        Args:
            reconstructed: Reconstructed features
            target: Ground truth features
            primitives: Primitive representations
            labels: Class labels
        
        Returns:
            dict with keys:
                - 'total': Total loss
                - 'recon': Reconstruction loss
                - 'primitive': Primitive loss
        """
        loss_recon = self.recon_loss(reconstructed, target)
        loss_prim = self.primitive_loss(primitives, labels)
        
        total = loss_recon + self.alpha * loss_prim
        
        return {
            'total': total,
            'recon': loss_recon,
            'primitive': loss_prim
        }
