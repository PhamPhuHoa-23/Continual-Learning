"""
Complete Slot-based Multi-Agent System.

End-to-end pipeline:
1. Image → Slot Attention → Slots (object tokens)
2. For each slot: Estimators → Top-k agents → Hidden labels
3. Aggregate all slots → CRP Expert Assignment → Prediction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..models.slot_attention import SlotAttention
from ..models.slot_attention.encoder import CNNEncoder
from .estimators import VAEEstimator, MLPEstimator
from .selector import TopKAgentSelector
from .atomic_agent import ResidualMLPAgent, create_agent_pool
from .aggregator import create_aggregator


class SlotMultiAgentSystem(nn.Module):
    """
    Complete slot-based multi-agent continual learning system.
    
    Architecture:
        Image (B, 3, H, W)
          ↓
        CNN Encoder → Features
          ↓
        Slot Attention → Slots (B, num_slots, slot_dim)
          ↓
        For EACH slot:
          ├─ Performance Estimators (50 agents) → Scores
          ├─ Top-K Selector → Select 3 best agents
          ├─ 3 Agents → 3 hidden labels (128-dim each)
          └─ Concat → 384-dim
          ↓
        Aggregate all slots:
          7 slots × 384-dim → 2688-dim (or mean → 384-dim)
          ↓
        CRP Expert Aggregator → Final prediction
    """
    
    def __init__(
        self,
        num_agents: int = 50,
        num_slots: int = 7,
        slot_dim: int = 64,
        hidden_dim: int = 128,  # Hidden label dimension (output of each agent)
        k: int = 3,
        num_classes: int = 100,
        input_channels: int = 3,
        input_size: int = 32,
        estimator_type: str = 'vae',  # 'vae' or 'mlp'
        aggregator_type: str = 'crp',  # 'crp', 'hoeffding', 'ensemble'
        aggregate_mode: str = 'concat',  # 'concat' or 'mean'
        device: str = 'cpu'
    ):
        """
        Args:
            num_agents: Number of specialized agents (default: 50)
            num_slots: Number of slots from Slot Attention (default: 7)
            slot_dim: Dimension of each slot (default: 64)
            hidden_dim: Hidden label dimension output by each agent (default: 128)
                This is the embedding size that goes to the decision tree
            k: Top-k agents to select per slot (default: 3)
            num_classes: Total number of classes (default: 100 for CIFAR-100)
            input_channels: Image channels (default: 3)
            input_size: Image size (default: 32 for CIFAR)
            estimator_type: Type of performance estimator
            aggregator_type: Type of decision tree aggregator
            aggregate_mode: How to aggregate slot outputs
            device: Device to run on
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.num_classes = num_classes
        self.aggregate_mode = aggregate_mode
        self.device = device
        
        # 1. Encoder (CNN)
        self.encoder = CNNEncoder(
            input_channels=input_channels,
            hidden_dims=[64, 64, 64, 64],
            output_dim=256
        ).to(device)
        
        # 2. Slot Attention
        self.slot_attention = SlotAttention(
            num_iterations=3,
            num_slots=num_slots,
            slot_size=slot_dim,
            mlp_hidden_size=128,
            epsilon=1e-8
        ).to(device)
        
        # Initialize slots
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        
        # 3. Performance Estimators
        if estimator_type == 'vae':
            self.estimators = nn.ModuleList([
                VAEEstimator(agent_id=i, slot_dim=slot_dim)
                for i in range(num_agents)
            ]).to(device)
        elif estimator_type == 'mlp':
            # Shared MLP estimator (conditioned on agent_id)
            self.estimators = [
                MLPEstimator(num_agents=num_agents, slot_dim=slot_dim).to(device)
            ] * num_agents
        else:
            raise ValueError(f"Unknown estimator_type: {estimator_type}")
        
        # 4. Top-K Selector
        self.selector = TopKAgentSelector(estimators=self.estimators, k=k)
        
        # 5. Atomic Agents (50 agents with same architecture, different weights)
        self.agents = create_agent_pool(
            num_agents=num_agents,
            agent_type='residual',
            slot_dim=slot_dim,
            output_dim=hidden_dim,  # Configurable hidden label dimension
            hidden_dim=256,  # Internal MLP hidden dimension
            num_blocks=3
        ).to(device)
        
        # 6. Aggregator (CRP Expert Assignment)
        # Input dimension depends on aggregate mode
        if aggregate_mode == 'concat':
            # num_slots × k × hidden_dim
            agg_input_dim = num_slots * k * hidden_dim
        elif aggregate_mode == 'mean':
            # Mean over slots: k × hidden_dim
            agg_input_dim = k * hidden_dim
        else:
            raise ValueError(f"Unknown aggregate_mode: {aggregate_mode}")
        
        self.aggregator = create_aggregator(
            aggregator_type=aggregator_type,
            feature_dim=agg_input_dim,
            num_classes=num_classes,
            device=device,
        )
        
        self.to(device)
    
    def forward(
        self,
        images: torch.Tensor,
        return_metadata: bool = False
    ) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Forward pass through complete system.
        
        Args:
            images: (B, C, H, W) - input images
            return_metadata: If True, return intermediate results
        
        Returns:
            predictions: (B,) - class predictions (None if tree not fitted)
            metadata: Dict with intermediate results
        """
        B = images.size(0)
        
        # Step 1: Extract features
        features = self.encoder(images)  # (B, 256, H', W')
        B, D, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        # Step 2: Slot Attention
        slots = self.slots_mu.expand(B, -1, -1)
        slots = slots + torch.randn_like(slots) * self.slots_logsigma.exp()
        slots, attn = self.slot_attention(features_flat, slots)
        # slots: (B, num_slots, slot_dim)
        
        # Step 3: For each slot, select top-k agents and get hidden labels
        all_hidden_labels = []
        all_selections = []
        
        for b in range(B):
            sample_hidden = []
            sample_selections = []
            
            for slot_idx in range(self.num_slots):
                slot = slots[b, slot_idx]  # (slot_dim,)
                
                # Select top-k agents
                selected_ids, scores = self.selector.select_top_k(
                    slot, return_scores=True
                )
                sample_selections.append({
                    'slot_idx': slot_idx,
                    'agent_ids': selected_ids,
                    'scores': scores
                })
                
                # Get hidden labels from selected agents
                slot_hidden_labels = []
                for agent_id in selected_ids:
                    agent = self.agents[agent_id]
                    with torch.no_grad() if not self.training else torch.enable_grad():
                        hidden_label = agent(slot)  # (128,)
                    slot_hidden_labels.append(hidden_label)
                
                # Concatenate agent outputs for this slot
                slot_concat = torch.cat(slot_hidden_labels)  # (k * 128,)
                sample_hidden.append(slot_concat)
            
            # Aggregate across all slots
            if self.aggregate_mode == 'concat':
                # Concat all slots: (num_slots * k * 128,)
                sample_aggregated = torch.cat(sample_hidden)
            elif self.aggregate_mode == 'mean':
                # Mean over slots: (k * 128,)
                sample_aggregated = torch.stack(sample_hidden).mean(dim=0)
            
            all_hidden_labels.append(sample_aggregated)
            all_selections.append(sample_selections)
        
        # Stack batch
        hidden_labels = torch.stack(all_hidden_labels)  # (B, tree_input_dim)
        
        # Step 4: CRP Expert prediction
        with torch.no_grad():
            hidden_labels_np = hidden_labels.cpu().numpy()
            try:
                preds = [
                    self.aggregator.predict_one(hl)
                    for hl in hidden_labels_np
                ]
                if all(p is not None for p in preds):
                    predictions = torch.tensor(preds, device=images.device)
                else:
                    predictions = None
            except Exception:
                # Aggregator not ready yet
                predictions = None
        
        # Metadata
        metadata = {
            'slots': slots,
            'attention': attn,
            'hidden_labels': hidden_labels,
            'selections': all_selections
        }
        
        if return_metadata:
            return predictions, metadata
        else:
            return predictions
    
    def train_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict:
        """
        Single training step (incremental learning).
        
        Args:
            images: (B, C, H, W)
            targets: (B,) - class labels
        
        Returns:
            info: Dict with training info
        """
        # Forward pass
        predictions, metadata = self.forward(images, return_metadata=True)
        
        # Incremental CRP expert learning (one-by-one)
        hidden_labels_np = metadata['hidden_labels'].cpu().detach().numpy()
        targets_np = targets.cpu().numpy()
        
        for hl, t in zip(hidden_labels_np, targets_np):
            self.aggregator.learn_one(hl, int(t))
        
        # Compute accuracy (if tree is fitted)
        if predictions is not None:
            accuracy = (predictions == targets).float().mean().item()
        else:
            accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'num_samples': len(targets)
        }
    
    def evaluate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict:
        """
        Evaluate on batch.
        
        Args:
            images: (B, C, H, W)
            targets: (B,)
        
        Returns:
            metrics: Dict with evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            predictions, metadata = self.forward(images, return_metadata=True)
        
        if predictions is None:
            return {'accuracy': 0.0, 'error': 'Tree not fitted'}
        
        accuracy = (predictions == targets).float().mean().item()
        
        # Per-class accuracy
        per_class_acc = {}
        for cls in torch.unique(targets):
            mask = targets == cls
            if mask.sum() > 0:
                cls_acc = (predictions[mask] == cls).float().mean().item()
                per_class_acc[cls.item()] = cls_acc
        
        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'num_samples': len(targets)
        }
    
    def get_system_info(self) -> Dict:
        """Get system statistics."""
        info = {
            'num_agents': self.num_agents,
            'num_slots': self.num_slots,
            'slot_dim': self.slot_dim,
            'k': self.k,
            'num_classes': self.num_classes,
            'aggregate_mode': self.aggregate_mode
        }
        
        # Tree info
        if hasattr(self.aggregator, 'get_info'):
            info['tree'] = self.aggregator.get_info()
        
        # Agent FLOPs
        total_flops = sum(agent.compute_flops() for agent in self.agents[:self.k])
        info['avg_flops_per_sample'] = total_flops * self.num_slots
        
        return info
    
    def save_checkpoint(self, path: str):
        """Save system checkpoint."""
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'slot_attention': self.slot_attention.state_dict(),
            'slots_mu': self.slots_mu,
            'slots_logsigma': self.slots_logsigma,
            'estimators': [e.state_dict() for e in self.estimators],
            'agents': [a.state_dict() for a in self.agents],
            'config': {
                'num_agents': self.num_agents,
                'num_slots': self.num_slots,
                'slot_dim': self.slot_dim,
                'k': self.k,
                'num_classes': self.num_classes
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load system checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.slot_attention.load_state_dict(checkpoint['slot_attention'])
        self.slots_mu = checkpoint['slots_mu']
        self.slots_logsigma = checkpoint['slots_logsigma']
        
        for i, state in enumerate(checkpoint['estimators']):
            self.estimators[i].load_state_dict(state)
        
        for i, state in enumerate(checkpoint['agents']):
            self.agents[i].load_state_dict(state)
        
        print(f"Checkpoint loaded from {path}")

