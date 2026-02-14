"""
Encoder for Slot Attention Model

Implements CNN encoders to extract features from images before Slot Attention.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Convolutional encoder that extracts features from images.
    
    This encoder uses a series of conv layers to downsample the input image
    and extract features that will be processed by Slot Attention.
    
    Args:
        in_channels (int): Number of input channels (3 for RGB). Default: 3.
        hidden_dims (Tuple[int, ...]): Hidden dimensions for conv layers.
            Default: (64, 64, 64, 64).
        out_dim (int): Output feature dimension. Default: 64.
        kernel_size (int): Kernel size for conv layers. Default: 5.
        stride (int): Stride for conv layers (downsampling). Default: 1.
    
    Input:
        x: torch.Tensor of shape (batch_size, in_channels, height, width)
    
    Output:
        features: torch.Tensor of shape (batch_size, num_features, out_dim)
            where num_features = (height // stride^n) * (width // stride^n)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        out_dim: int = 64,
        kernel_size: int = 5,
        stride: int = 1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_dim = out_dim
        
        # Build conv layers
        layers = []
        prev_dim = in_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(
                    prev_dim, hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Final projection to out_dim
        layers.append(
            nn.Conv2d(
                prev_dim, out_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to features.
        
        Args:
            x: Input image of shape (batch_size, in_channels, height, width).
            
        Returns:
            features: Encoded features of shape (batch_size, num_features, out_dim).
        """
        # x: (B, C, H, W)
        x = self.encoder(x)
        # x: (B, out_dim, H', W')
        
        batch_size, feature_dim, height, width = x.shape
        
        # Reshape to (B, H'*W', out_dim)
        features = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, feature_dim)
        
        return features


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder using torchvision's pretrained ResNet.
    
    This encoder uses a pretrained ResNet model to extract features.
    Useful for transfer learning and better feature extraction.
    
    Args:
        model_name (str): ResNet model name ('resnet18', 'resnet34', 'resnet50').
            Default: 'resnet18'.
        out_dim (int): Output feature dimension. Default: 64.
        pretrained (bool): Whether to use pretrained weights. Default: True.
        freeze_backbone (bool): Whether to freeze ResNet weights. Default: False.
    
    Input:
        x: torch.Tensor of shape (batch_size, 3, height, width)
    
    Output:
        features: torch.Tensor of shape (batch_size, num_features, out_dim)
    """
    
    def __init__(
        self,
        model_name: str = 'resnet18',
        out_dim: int = 64,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        import torchvision.models as models
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove final FC layer and avgpool
        layers = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection to out_dim
        self.proj = nn.Conv2d(backbone_dim, out_dim, kernel_size=1)
        
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to features using ResNet.
        
        Args:
            x: Input image of shape (batch_size, 3, height, width).
            
        Returns:
            features: Encoded features of shape (batch_size, num_features, out_dim).
        """
        # Extract features using ResNet backbone
        # x: (B, 3, H, W) -> (B, backbone_dim, H', W')
        x = self.backbone(x)
        
        # Project to out_dim
        # x: (B, backbone_dim, H', W') -> (B, out_dim, H', W')
        x = self.proj(x)
        
        batch_size, feature_dim, height, width = x.shape
        
        # Reshape to (B, H'*W', out_dim)
        features = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, feature_dim)
        
        return features


if __name__ == "__main__":
    # Test encoders
    print("Testing CNN Encoder...")
    
    batch_size = 4
    img_size = 64
    
    # CNN Encoder
    cnn_encoder = CNNEncoder(in_channels=3, out_dim=64)
    x = torch.randn(batch_size, 3, img_size, img_size)
    features = cnn_encoder(x)
    
    print(f"CNN Encoder:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {features.shape}")
    
    # ResNet Encoder
    print("\nTesting ResNet Encoder...")
    resnet_encoder = ResNetEncoder(model_name='resnet18', out_dim=64, pretrained=False)
    features = resnet_encoder(x)
    
    print(f"ResNet Encoder:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {features.shape}")
    
    print("\nEncoder tests passed!")

