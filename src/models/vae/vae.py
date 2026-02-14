"""
Variational Autoencoder (VAE) implementations.

VAE learns to encode inputs into a latent distribution and reconstruct them.
Reconstruction error serves as a proxy for uncertainty/difficulty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VAE(nn.Module):
    """Simple MLP-based VAE.
    
    For flattened inputs (e.g., MNIST flattened to 784).
    
    Attributes:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        hidden_dims: List of hidden dimensions for encoder/decoder
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list = [512, 256]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space: mu and log_var
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.
        
        Args:
            x: Input (B, input_dim)
            
        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon.
        
        Args:
            mu: Mean (B, latent_dim)
            logvar: Log variance (B, latent_dim)
            
        Returns:
            z: Sampled latent (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction.
        
        Args:
            z: Latent (B, latent_dim)
            
        Returns:
            Reconstruction (B, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input (B, input_dim)
            
        Returns:
            Dictionary with:
                - recon: Reconstruction (B, input_dim)
                - mu: Latent mean (B, latent_dim)
                - logvar: Latent log variance (B, latent_dim)
                - z: Sampled latent (B, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        recon: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss (ELBO).
        
        Loss = Reconstruction Loss + beta * KL Divergence
        
        Args:
            x: Original input
            recon: Reconstruction
            mu: Latent mean
            logvar: Latent log variance
            beta: Weight for KL term (beta-VAE)
            
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE or BCE depending on data)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class ConvVAE(nn.Module):
    """Convolutional VAE for images.
    
    Uses CNN encoder/decoder for better image modeling.
    
    Attributes:
        input_channels: Number of input channels (e.g., 3 for RGB)
        input_size: Input image size (assumes square)
        latent_dim: Latent space dimension
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 32,
        latent_dim: int = 128,
        hidden_dims: list = [32, 64, 128, 256]
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size after conv layers
        self.flatten_size = hidden_dims[-1] * (input_size // (2 ** len(hidden_dims))) ** 2
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims))
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    hidden_dims_reversed[i],
                    hidden_dims_reversed[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                nn.LeakyReLU()
            ])
        
        # Final layer
        decoder_layers.extend([
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()  # Output in [0, 1]
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store spatial size for reshaping
        self.spatial_size = input_size // (2 ** len(hidden_dims))
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
        """
        h = self.encoder(x)  # (B, C', H', W')
        h = h.flatten(1)  # (B, flatten_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.
        
        Args:
            z: Latent (B, latent_dim)
            
        Returns:
            Reconstructed image (B, C, H, W)
        """
        h = self.decoder_input(z)  # (B, flatten_size)
        h = h.view(-1, self.hidden_dims[-1], self.spatial_size, self.spatial_size)
        recon = self.decoder(h)
        return recon
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Dictionary with reconstruction and latent variables
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss.
        
        Args:
            x: Original images (B, C, H, W)
            recon: Reconstructed images (B, C, H, W)
            mu: Latent mean
            logvar: Latent log variance
            beta: KL weight (beta-VAE)
            
        Returns:
            Dictionary with loss components
        """
        batch_size = x.size(0)
        
        # Reconstruction loss (per-pixel MSE or BCE)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / batch_size
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def reconstruct_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error.
        
        This is the key for uncertainty estimation:
        High reconstruction error -> high uncertainty
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Reconstruction error per sample (B,)
        """
        with torch.no_grad():
            output = self.forward(x)
            recon = output['recon']
            
            # Per-sample MSE
            error = F.mse_loss(recon, x, reduction='none')
            error = error.view(x.size(0), -1).mean(dim=1)
            
        return error

