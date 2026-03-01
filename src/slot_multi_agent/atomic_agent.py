"""
Atomic Agents: Specialized modules that process slots and output hidden labels.

Each agent learns via DINO-style self-supervised learning to output discrete
probability distributions (hidden labels) over learned prototypes.

Architecture:
    - ResidualMLP backbone: processes slot tokens
    - DINO head: outputs logits over K prototypes
    - Output: softmax probabilities (continuous hidden label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualMLPAgent(nn.Module):
    """
    Atomic agent that processes slots and outputs hidden labels.
    
    Trained with DINO-style self-supervised learning. Each agent learns
    K prototypes and outputs a probability distribution (hidden label).
    
    Architecture:
        Input slot → Residual MLP blocks → DINO head → softmax → hidden label
    
    Args:
        slot_dim: Dimension of input slot token (default: 64)
        hidden_dim: Hidden dimension for MLP blocks (default: 256)
        num_prototypes: Number of prototypes (discrete concepts) (default: 256)
        num_blocks: Number of residual blocks (default: 3)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> agent = ResidualMLPAgent(slot_dim=64, num_prototypes=256)
        >>> slot = torch.randn(8, 64)  # batch_size=8
        >>> hidden_label = agent(slot)  # (8, 256) softmax probabilities
    """
    
    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dim: int = 256,
        num_prototypes: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.num_blocks = num_blocks
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # DINO-style projection head
        # Similar to DINOv2: 3-layer MLP with bottleneck
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),  # Bottleneck
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_prototypes)
        )
    
    def forward(
        self,
        slot: torch.Tensor,
        return_logits: bool = False,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass: slot → hidden label (probability distribution).
        
        Args:
            slot: Input slot token (batch_size, slot_dim)
            return_logits: If True, return logits. If False, return softmax probs.
            temperature: Temperature for softmax (default: 1.0)
        
        Returns:
            hidden_label: (batch_size, num_prototypes)
                If return_logits=True: raw logits
                If return_logits=False: softmax probabilities (sums to 1)
        """
        # Input projection
        x = self.input_proj(slot)  # (batch_size, hidden_dim)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # DINO head
        logits = self.head(x)  # (batch_size, num_prototypes)
        
        if return_logits:
            return logits
        else:
            # Softmax with temperature
            probs = F.softmax(logits / temperature, dim=-1)
            return probs
    
    def get_embedding(self, slot: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate embedding (before head) for analysis.
        
        Args:
            slot: Input slot (batch_size, slot_dim)
        
        Returns:
            embedding: (batch_size, hidden_dim)
        """
        x = self.input_proj(slot)
        for block in self.blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual MLP block with LayerNorm and GELU activation.
    
    Architecture:
        x → Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm → (+x) → output
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # Residual connection


class DINOLoss(nn.Module):
    """
    DINO loss with centering and temperature sharpening.
    
    Implements the self-supervised learning objective from DINOv2:
    - Teacher outputs are centered (to prevent collapse)
    - Teacher outputs are sharpened with low temperature
    - Student learns to match teacher via cross-entropy
    
    Args:
        num_prototypes: Number of prototypes (output dimension)
        student_temp: Student temperature for softmax (default: 0.1)
        teacher_temp: Teacher temperature for softmax (default: 0.07, sharper)
        center_momentum: EMA momentum for center update (default: 0.9)
    
    Reference:
        DINOv2: Learning Robust Visual Features without Supervision
        https://github.com/facebookresearch/dinov2
    """
    
    def __init__(
        self,
        num_prototypes: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.07,
        center_momentum: float = 0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        # Running center (prevents mode collapse)
        self.register_buffer("center", torch.zeros(1, num_prototypes))
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DINO loss between student and teacher outputs.
        
        Args:
            student_logits: (batch_size, num_prototypes) - raw logits
            teacher_logits: (batch_size, num_prototypes) - raw logits
        
        Returns:
            loss: Scalar cross-entropy loss
        """
        # Ensure 2D (batch_size, num_prototypes)
        student_logits = student_logits.reshape(-1, student_logits.shape[-1])
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
        
        # Teacher: centering + sharpening
        teacher_centered = teacher_logits - self.center
        teacher_probs = F.softmax(teacher_centered / self.teacher_temp, dim=-1)
        
        # Student: sharpening only
        student_log_probs = F.log_softmax(student_logits / self.student_temp, dim=-1)
        
        # Cross-entropy loss
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        
        # Update center (EMA, no gradient)
        self._update_center(teacher_logits)
        
        return loss
    
    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor):
        """Update running center with EMA."""
        # Ensure 2D for mean calculation
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)


def create_agent_pool(
    num_agents: int,
    slot_dim: int,
    num_prototypes: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout: float = 0.1,
    device: str = 'cpu'
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """
    Create a pool of student and teacher agents.
    
    Teachers are initialized as copies of students and updated via EMA.
    Teachers do not receive gradients (requires_grad=False).
    
    Args:
        num_agents: Number of agents to create (default: 50)
        slot_dim: Slot dimension
        num_prototypes: Number of prototypes per agent
        hidden_dim: Hidden dimension for MLP
        num_blocks: Number of residual blocks
        dropout: Dropout rate
        device: Device to place agents on
    
    Returns:
        student_agents: ModuleList of trainable agents
        teacher_agents: ModuleList of EMA teacher agents (no grad)
    """
    student_agents = nn.ModuleList([
        ResidualMLPAgent(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            num_blocks=num_blocks,
            dropout=dropout
        )
        for _ in range(num_agents)
    ]).to(device)
    
    teacher_agents = nn.ModuleList([
        ResidualMLPAgent(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            num_blocks=num_blocks,
            dropout=dropout
        )
        for _ in range(num_agents)
    ]).to(device)
    
    # Initialize teachers with student weights
    for student, teacher in zip(student_agents, teacher_agents):
        teacher.load_state_dict(student.state_dict())
        teacher.requires_grad_(False)  # No gradients for teachers
    
    return student_agents, teacher_agents


@torch.no_grad()
def update_teacher(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float = 0.996
):
    """
    Update teacher parameters via EMA of student parameters.
    
    Formula:
        θ_teacher = momentum * θ_teacher + (1 - momentum) * θ_student
    
    Args:
        student: Student agent
        teacher: Teacher agent
        momentum: EMA momentum (default: 0.996, typical for DINO)
    """
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)


def update_all_teachers(
    student_agents: nn.ModuleList,
    teacher_agents: nn.ModuleList,
    momentum: float = 0.996
):
    """Update all teachers in a pool via EMA."""
    for student, teacher in zip(student_agents, teacher_agents):
        update_teacher(student, teacher, momentum)
