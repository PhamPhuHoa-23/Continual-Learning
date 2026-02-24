"""
VAE-based Uncertainty Estimation.

Uses reconstruction error as a proxy for uncertainty/difficulty:
- High reconstruction error -> Sample is unusual/out-of-distribution/difficult
- Low reconstruction error -> Sample is familiar/in-distribution/easy

This is one of the simplest and most intuitive approaches for RCCL.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from ...base.types import UncertaintyOutput
from .vae import ConvVAE, VAE


class VAEUncertaintyEstimator:
    """Estimate uncertainty using VAE reconstruction error.
    
    Core Idea:
    1. Train VAE to reconstruct familiar/in-distribution samples
    2. At test time, high reconstruction error indicates:
       - Out-of-distribution sample
       - Unusual/difficult sample
       - New class (in continual learning)
    3. Use reconstruction error as uncertainty metric
    
    This approach is:
    - Simple and intuitive
    - Computationally efficient (single forward pass)
    - Effective for detecting distribution shift
    - Useful for continual learning task selection
    
    Attributes:
        vae: VAE model (ConvVAE or VAE)
        running_mean: Running mean of reconstruction errors
        running_std: Running std of reconstruction errors
        num_samples_seen: Number of samples processed
    """
    
    def __init__(
        self,
        vae: nn.Module,
        normalize: bool = True,
        device: str = 'cpu'
    ):
        """Initialize VAE uncertainty estimator.
        
        Args:
            vae: Trained VAE model
            normalize: Whether to normalize errors using running statistics
            device: Device to run on
        """
        
        
        self.vae = vae.to(device)
        self.normalize = normalize
        self.device = device
        
        # Running statistics for normalization
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_std', torch.tensor(1.0))
        self.register_buffer('num_samples_seen', torch.tensor(0))
        
        # Threshold for OOD detection (adaptive)
        self.register_buffer('ood_threshold', torch.tensor(3.0))  # 3 std devs
    
    def update_statistics(self, errors: torch.Tensor):
        """Update running statistics for normalization.
        
        Args:
            errors: Reconstruction errors (B,)
        """
        if not self.normalize:
            return
        
        batch_size = errors.size(0)
        
        # Update running mean and std
        if self.num_samples_seen == 0:
            self.running_mean = errors.mean()
            self.running_std = errors.std() + 1e-8
        else:
            # Exponential moving average
            alpha = min(0.1, 100.0 / (self.num_samples_seen + 1))
            self.running_mean = (1 - alpha) * self.running_mean + alpha * errors.mean()
            self.running_std = (1 - alpha) * self.running_std + alpha * (errors.std() + 1e-8)
        
        self.num_samples_seen += batch_size
    
    def normalize_errors(self, errors: torch.Tensor) -> torch.Tensor:
        """Normalize reconstruction errors.
        
        Args:
            errors: Raw reconstruction errors (B,)
            
        Returns:
            Normalized errors (z-scores) (B,)
        """
        if not self.normalize or self.num_samples_seen < 10:
            return errors
        
        return (errors - self.running_mean) / (self.running_std + 1e-8)
    
    def estimate(self, x: torch.Tensor) -> UncertaintyOutput:
        """Estimate uncertainty using VAE reconstruction error.
        
        Args:
            x: Input data (B, C, H, W) for ConvVAE or (B, D) for VAE
            
        Returns:
            UncertaintyOutput with:
                - epistemic: Normalized reconstruction error (proxy for OOD)
                - aleatoric: KL divergence (spread of latent distribution)
                - total: Combined uncertainty
                - confidence: 1 - total uncertainty
        """
        self.vae.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            
            # Forward pass
            output = self.vae(x)
            recon = output['recon']
            mu = output['mu']
            logvar = output['logvar']
            
            # Reconstruction error (per sample)
            recon_error = torch.nn.functional.mse_loss(
                recon, x, reduction='none'
            )
            recon_error = recon_error.view(x.size(0), -1).mean(dim=1)
            
            # Update and normalize
            if self.training:
                self.update_statistics(recon_error)
            
            normalized_error = self.normalize_errors(recon_error)
            
            # KL divergence per sample (spread of latent distribution)
            # Higher KL -> more uncertain about latent representation
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_div = kl_div / mu.size(1)  # Normalize by latent dim
            
            # Epistemic uncertainty: normalized reconstruction error
            # (measures how unusual the sample is)
            epistemic = torch.sigmoid(normalized_error)  # Bound to [0, 1]
            
            # Aleatoric uncertainty: KL divergence
            # (measures variability in latent space)
            aleatoric = torch.sigmoid(kl_div * 0.1)  # Scale and bound
            
            # Total uncertainty
            total = epistemic + aleatoric
            total = total / 2.0  # Normalize to [0, 1]
            
            # Confidence
            confidence = 1.0 - total
            
        return UncertaintyOutput(
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence=confidence
        )
    
    def is_ood(self, x: torch.Tensor) -> torch.Tensor:
        """Detect out-of-distribution samples.
        
        Args:
            x: Input data
            
        Returns:
            Boolean mask (B,) indicating OOD samples
        """
        uncertainty = self.estimate(x)
        
        # Consider sample OOD if epistemic uncertainty is high
        return uncertainty.epistemic > (self.ood_threshold / 10.0)
    
    def compute_cost_estimate(self, x: torch.Tensor) -> float:
        """Estimate computational cost based on reconstruction error.
        
        Idea: Difficult samples (high reconstruction error) may need more
        computational resources (deeper networks, more iterations, etc.)
        
        Args:
            x: Input data
            
        Returns:
            Estimated cost multiplier
        """
        uncertainty = self.estimate(x)
        
        # Map uncertainty to cost multiplier [1.0, 2.0]
        # High uncertainty -> higher cost for proper processing
        cost_multiplier = 1.0 + uncertainty.total.mean().item()
        
        return cost_multiplier
    
    def train_vae(
        self,
        train_loader,
        num_epochs: int = 50,
        lr: float = 1e-3,
        beta: float = 1.0,
        verbose: bool = True
    ):
        """Train the VAE model.
        
        Args:
            train_loader: DataLoader for training
            num_epochs: Number of training epochs
            lr: Learning rate
            beta: Beta parameter for beta-VAE (KL weight)
            verbose: Print progress
        """
        self.vae.train()
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0
            
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.vae(data)
                
                # Compute loss
                loss_dict = self.vae.compute_loss(
                    data,
                    output['recon'],
                    output['mu'],
                    output['logvar'],
                    beta=beta
                )
                
                loss = loss_dict['loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += loss_dict['recon_loss'].item()
                total_kl += loss_dict['kl_loss'].item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} - '
                      f'Loss: {total_loss/len(train_loader):.4f}, '
                      f'Recon: {total_recon/len(train_loader):.4f}, '
                      f'KL: {total_kl/len(train_loader):.4f}')
        
        self.vae.eval()
        
        # Calibrate running statistics on training data
        if self.normalize:
            print("Calibrating normalization statistics...")
            all_errors = []
            with torch.no_grad():
                for data, _ in train_loader:
                    data = data.to(self.device)
                    output = self.vae(data)
                    recon_error = torch.nn.functional.mse_loss(
                        output['recon'], data, reduction='none'
                    )
                    recon_error = recon_error.view(data.size(0), -1).mean(dim=1)
                    all_errors.append(recon_error)
            
            all_errors = torch.cat(all_errors)
            self.running_mean = all_errors.mean()
            self.running_std = all_errors.std() + 1e-8
            self.num_samples_seen = len(all_errors)
            
            if verbose:
                print(f'Calibrated: mean={self.running_mean:.4f}, '
                      f'std={self.running_std:.4f}')

