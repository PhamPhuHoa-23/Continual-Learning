import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimitiveLoss(nn.Module):
    """
    Matrix-level primitive loss (L_p).
    KL divergence between label and hidden-label similarity matrices.
    """
    def __init__(self, tau: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.tau = tau
        self.eps = eps

    def forward(self, H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Aggregated representations (batch_size, d_h)
            y: Labels (batch_size,)
        Returns:
            Scalar loss.
        """
        batch_size = H.size(0)
        
        # d^y_{ij} calculation
        y_mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        y_sum = y_mask.sum(dim=1, keepdim=True)
        d_y = y_mask / (y_sum + self.eps)

        # d^H_{ij} calculation
        H_norm = F.normalize(H, p=2, dim=1)
        sim_matrix = torch.matmul(H_norm, H_norm.t())
        exp_sim = torch.exp(self.tau * sim_matrix)
        exp_sum = exp_sim.sum(dim=1, keepdim=True)
        d_H = exp_sim / (exp_sum + self.eps)

        # KL divergence: sum( d_y * log(d_y / d_H) )
        loss = torch.sum(d_y * (torch.log(d_y + self.eps) - torch.log(d_H + self.eps)))
        return loss / batch_size

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (L_SupCon).
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Aggregated representations (batch_size, d_h)
            y: Labels (batch_size,)
        Returns:
            Scalar loss.
        """
        batch_size = H.size(0)
        H_norm = F.normalize(H, p=2, dim=1)
        sim_matrix = torch.matmul(H_norm, H_norm.t()) / self.tau

        # Mask for a != i
        mask_self = torch.eye(batch_size, device=H.device).bool()
        
        # Mask for same class
        mask_pos = (y.unsqueeze(0) == y.unsqueeze(1))
        mask_pos.fill_diagonal_(False) # Remove self from positive set

        # Denominator: sum over all a != i
        exp_sim = torch.exp(sim_matrix)
        exp_sim_non_self = exp_sim.clone()
        exp_sim_non_self[mask_self] = 0.0
        sum_exp_sim_non_self = exp_sim_non_self.sum(dim=1, keepdim=True)

        # Avoid log(0)
        log_prob = sim_matrix - torch.log(sum_exp_sim_non_self + 1e-8)
        
        # sum_{p in P(i)} log_prob
        mean_log_prob_pos = (mask_pos.float() * log_prob).sum(dim=1) / (mask_pos.float().sum(dim=1) + 1e-8)
        
        # -1/|P(i)| sum ...
        loss = -mean_log_prob_pos.mean()
        return loss

class AgentReconstructionLoss(nn.Module):
    """
    Agent reconstruction loss (anti-collapse) L_agent.
    """
    def __init__(self):
        super().__init__()

    def forward(self, dec_h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dec_h: Decoder outputs of agents for slots (batch_size * num_slots, slot_dim)
            s: Original slots (batch_size * num_slots, slot_dim)
        Returns:
            MSE loss.
        """
        return F.mse_loss(dec_h, s)

class LocalGeometryLoss(nn.Module):
    """
    Neighbor-robustness loss L_local.
    Requires affinity matrix e^{t-1}_{ij} from previous task instances (exemplars).
    """
    def __init__(self):
        super().__init__()

    def forward(self, H: torch.Tensor, e_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Aggregated representations for exemplars (num_exemplars, d_h)
            e_matrix: Affinity matrix e^{t-1}_{ij} (num_exemplars, num_exemplars)
        Returns:
            Scalar loss.
        """
        # H_i - H_j
        H_i = H.unsqueeze(1) # (N, 1, d_h)
        H_j = H.unsqueeze(0) # (1, N, d_h)
        diff_sq = torch.sum((H_i - H_j) ** 2, dim=2) # (N, N)
        
        loss = torch.sum(e_matrix * diff_sq) / (H.size(0) * H.size(0))
        return loss

class CompositionalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        tau_p: float = 1.0,
        tau_supcon: float = 0.1
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        self.loss_p = PrimitiveLoss(tau=tau_p)
        self.loss_supcon = SupConLoss(tau=tau_supcon)
        self.loss_agent = AgentReconstructionLoss()
        self.loss_local = LocalGeometryLoss()

    def forward(
        self, 
        H: torch.Tensor, 
        y: torch.Tensor, 
        dec_h: torch.Tensor, 
        s: torch.Tensor,
        e_matrix: torch.Tensor = None
    ) -> dict:
        
        p_val = self.loss_p(H, y)
        supcon_val = self.loss_supcon(H, y)
        agent_val = self.loss_agent(dec_h, s)
        
        total_loss = self.alpha * p_val + self.beta * supcon_val + self.gamma * agent_val
        
        local_val = torch.tensor(0.0, device=H.device)
        if e_matrix is not None and self.delta > 0:
            # Assumes H and e_matrix are aligned (e.g., computed on exemplar subset)
            local_val = self.loss_local(H, e_matrix)
            total_loss += self.delta * local_val
            
        return {
            'loss': total_loss,
            'l_p': p_val,
            'l_supcon': supcon_val,
            'l_agent': agent_val,
            'l_local': local_val
        }
