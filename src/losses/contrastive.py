"""
Contrastive and Clustering Losses for Slot Attention Training

These losses help slot embeddings become more discriminative and cluster-friendly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    Pulls together embeddings from the same class, pushes apart different classes.
    
    Reference:
        Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
        https://arxiv.org/abs/2004.11362
    
    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Normalized embeddings (B, D)
            labels: Class labels (B,)
        
        Returns:
            loss: Scalar contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix: (B, B)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


class PrototypeLoss(nn.Module):
    """
    Prototype-based loss for clustering.
    
    Learns K class prototypes and pulls embeddings toward their class prototype.
    Simpler than SupCon, works well when batch sizes are small.
    
    Args:
        num_classes: Number of classes
        embedding_dim: Dimension of embeddings
        temperature: Temperature for softmax (default: 0.1)
        momentum: EMA momentum for prototype updates (default: 0.9)
    """
    
    def __init__(
        self, 
        num_classes: int,
        embedding_dim: int,
        temperature: float = 0.1,
        momentum: float = 0.9
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.momentum = momentum
        
        # Learnable prototypes (K, D)
        self.register_buffer(
            'prototypes', 
            F.normalize(torch.randn(num_classes, embedding_dim), dim=1)
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Pull embeddings toward their class prototype.
        
        Args:
            features: Embeddings (B, D)
            labels: Class labels (B,)
        
        Returns:
            loss: Scalar prototype loss
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity to all prototypes: (B, K)
        similarity = torch.matmul(features, self.prototypes.T) / self.temperature
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        # Update prototypes via EMA
        with torch.no_grad():
            for c in range(self.num_classes):
                if (labels == c).any():
                    # Get features for this class
                    class_features = features[labels == c]
                    # Compute mean
                    class_mean = class_features.mean(dim=0)
                    # EMA update
                    self.prototypes[c] = (
                        self.momentum * self.prototypes[c] + 
                        (1 - self.momentum) * class_mean
                    )
                    # Renormalize
                    self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)
        
        return loss


class SlotClusteringLoss(nn.Module):
    """
    Slot-aware clustering loss for AdaSlot training.
    
    Aggregates multiple slots per image, then applies contrastive/prototype loss
    to the aggregated representation to make slots cluster-friendly.
    
    Args:
        loss_type: 'contrastive' or 'prototype'
        temperature: Temperature parameter
        num_classes: Number of classes (for prototype loss)
        embedding_dim: Slot dimension (for prototype loss)
        aggregation: How to aggregate slots ('mean', 'max', 'attention')
    """
    
    def __init__(
        self,
        loss_type: str = 'contrastive',
        temperature: float = 0.07,
        num_classes: int = 10,
        embedding_dim: int = 64,
        aggregation: str = 'mean'
    ):
        super().__init__()
        self.aggregation = aggregation
        
        if loss_type == 'contrastive':
            self.loss_fn = SupervisedContrastiveLoss(temperature=temperature)
        elif loss_type == 'prototype':
            self.loss_fn = PrototypeLoss(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def aggregate_slots(
        self, 
        slots: torch.Tensor, 
        masks: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Aggregate multiple slots per image into single representation.
        
        Args:
            slots: (B, num_slots, slot_dim)
            masks: Optional attention masks (B, num_slots, H, W)
        
        Returns:
            aggregated: (B, slot_dim)
        """
        if self.aggregation == 'mean':
            # Simple average
            return slots.mean(dim=1)
        
        elif self.aggregation == 'max':
            # Max pooling
            return slots.max(dim=1)[0]
        
        elif self.aggregation == 'attention':
            # Weighted average using mask importance
            if masks is None:
                return slots.mean(dim=1)
            # Use mask coverage as weight
            weights = masks.flatten(2).sum(dim=2)  # (B, num_slots)
            weights = F.softmax(weights, dim=1).unsqueeze(2)  # (B, num_slots, 1)
            return (slots * weights).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def forward(
        self, 
        slots: torch.Tensor, 
        labels: torch.Tensor,
        masks: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute clustering loss on aggregated slot representations.
        
        Args:
            slots: (B, num_slots, slot_dim)
            labels: Class labels (B,)
            masks: Optional attention masks (B, num_slots, H, W)
        
        Returns:
            loss: Scalar clustering loss
        """
        # Aggregate slots per image
        aggregated = self.aggregate_slots(slots, masks)  # (B, slot_dim)
        
        # Apply contrastive/prototype loss
        loss = self.loss_fn(aggregated, labels)
        
        return loss
