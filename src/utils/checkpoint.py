"""
Checkpoint utilities for loading and saving model weights.

Supports:
    - Slot Attention checkpoints (AdaSlot .ckpt files)
    - Agent checkpoints (PyTorch .pth files)
    - Estimator checkpoints
    - Full system checkpoints
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


def load_slot_attention_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load pretrained Slot Attention weights.
    
    Supports:
        - AdaSlot PyTorch Lightning checkpoints (.ckpt)
        - Standard PyTorch checkpoints (.pth)
    
    Args:
        model: Slot Attention model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Strict mode for loading (if False, ignore missing keys)
        device: Device to load checkpoint to
    
    Returns:
        Model with loaded weights
    
    Example:
        >>> from src.models.slot_attention import SlotAttentionAutoEncoder
        >>> model = SlotAttentionAutoEncoder(num_slots=7, slot_dim=64)
        >>> model = load_slot_attention_checkpoint(
        ...     model,
        ...     './checkpoints/CLEVR10.ckpt',
        ...     strict=False
        ... )
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading Slot Attention checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Checkpoint loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint
            state_dict = checkpoint['state_dict']
            print("  Format: PyTorch Lightning (.ckpt)")
            
            # Remove 'model.' prefix if present
            state_dict = {
                k.replace('model.', ''): v
                for k, v in state_dict.items()
            }
        elif 'model_state_dict' in checkpoint:
            # Standard PyTorch checkpoint
            state_dict = checkpoint['model_state_dict']
            print("  Format: PyTorch (.pth)")
        else:
            # Assume it's just the state dict
            state_dict = checkpoint
            print("  Format: State dict only")
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    # Filter keys to match model
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print(f"⚠ Missing keys in checkpoint: {len(missing_keys)}")
        if strict:
            print(f"  Keys: {list(missing_keys)[:5]}...")
    
    if unexpected_keys:
        print(f"⚠ Unexpected keys in checkpoint: {len(unexpected_keys)}")
        if strict:
            print(f"  Keys: {list(unexpected_keys)[:5]}...")
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=strict)
        print(f"✓ Weights loaded successfully (strict={strict})")
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to load weights (strict mode): {e}")
        else:
            warnings.warn(f"Some weights could not be loaded: {e}")
            # Try to load what we can
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_keys
            }
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✓ Loaded {len(filtered_state_dict)}/{len(model_keys)} parameters")
    
    return model


def load_agent_checkpoint(
    student_agents: nn.ModuleList,
    teacher_agents: Optional[nn.ModuleList],
    checkpoint_path: str,
    strict: bool = False,
    device: str = 'cpu'
) -> tuple[nn.ModuleList, Optional[nn.ModuleList]]:
    """
    Load pretrained agent weights (from Phase 1 training).
    
    Args:
        student_agents: ModuleList of student agents
        teacher_agents: ModuleList of teacher agents (optional)
        checkpoint_path: Path to checkpoint file
        strict: Strict mode for loading
        device: Device to load checkpoint to
    
    Returns:
        (student_agents, teacher_agents) with loaded weights
    
    Example:
        >>> students, teachers = create_agent_pool(50, 64, 256)
        >>> students, teachers = load_agent_checkpoint(
        ...     students, teachers,
        ...     './checkpoints/agents_phase1_epoch10.pth'
        ... )
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading agent checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load student agents
    if 'student_agents' in checkpoint:
        student_state_dict = checkpoint['student_agents']
        student_agents.load_state_dict(student_state_dict, strict=strict)
        print(f"✓ Loaded {len(student_agents)} student agents")
    else:
        warnings.warn("No 'student_agents' found in checkpoint")
    
    # Load teacher agents (if available and requested)
    if teacher_agents is not None:
        if 'teacher_agents' in checkpoint:
            teacher_state_dict = checkpoint['teacher_agents']
            teacher_agents.load_state_dict(teacher_state_dict, strict=strict)
            print(f"✓ Loaded {len(teacher_agents)} teacher agents")
        else:
            print("⚠ No 'teacher_agents' in checkpoint, copying from students")
            # Copy student weights to teacher
            for student, teacher in zip(student_agents, teacher_agents):
                teacher.load_state_dict(student.state_dict())
    
    # Load DINO loss centers (if available)
    if 'dino_loss_centers' in checkpoint:
        centers = checkpoint['dino_loss_centers']
        print(f"✓ DINO loss centers available ({len(centers)} agents)")
        # Note: Caller should restore these to DINOLoss objects
    
    return student_agents, teacher_agents


def load_estimator_checkpoint(
    estimators: nn.ModuleList,
    checkpoint_path: str,
    strict: bool = False,
    device: str = 'cpu'
) -> nn.ModuleList:
    """
    Load pretrained estimator weights.
    
    Args:
        estimators: ModuleList of estimators
        checkpoint_path: Path to checkpoint file
        strict: Strict mode for loading
        device: Device to load checkpoint to
    
    Returns:
        estimators with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading estimator checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'estimators' in checkpoint:
        estimators.load_state_dict(checkpoint['estimators'], strict=strict)
        print(f"✓ Loaded {len(estimators)} estimators")
    else:
        raise KeyError("No 'estimators' found in checkpoint")
    
    return estimators


def save_slot_attention_checkpoint(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save Slot Attention checkpoint.
    
    Args:
        model: Slot Attention model
        save_path: Path to save checkpoint
        metadata: Optional metadata to include
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Slot Attention checkpoint saved to: {save_path}")


def save_agent_checkpoint(
    student_agents: nn.ModuleList,
    teacher_agents: nn.ModuleList,
    dino_losses: list,
    save_path: str,
    epoch: int,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save agent checkpoint (Phase 1 training).
    
    Args:
        student_agents: Student agents
        teacher_agents: Teacher agents
        dino_losses: List of DINOLoss objects
        save_path: Path to save checkpoint
        epoch: Current epoch
        metadata: Optional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract DINO loss centers
    centers = [loss.center.clone() for loss in dino_losses]
    
    checkpoint = {
        'student_agents': student_agents.state_dict(),
        'teacher_agents': teacher_agents.state_dict(),
        'dino_loss_centers': centers,
        'epoch': epoch,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Agent checkpoint saved to: {save_path}")


def save_estimator_checkpoint(
    estimators: nn.ModuleList,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save estimator checkpoint.
    
    Args:
        estimators: Estimators
        save_path: Path to save checkpoint
        metadata: Optional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'estimators': estimators.state_dict(),
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Estimator checkpoint saved to: {save_path}")


def save_full_checkpoint(
    slot_attention: nn.Module,
    student_agents: nn.ModuleList,
    teacher_agents: nn.ModuleList,
    estimators: nn.ModuleList,
    dino_losses: list,
    save_path: str,
    epoch: int,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save full system checkpoint (all components).
    
    Args:
        slot_attention: Slot Attention model
        student_agents: Student agents
        teacher_agents: Teacher agents
        estimators: Estimators
        dino_losses: DINO losses
        save_path: Path to save checkpoint
        epoch: Current epoch
        metadata: Optional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    centers = [loss.center.clone() for loss in dino_losses]
    
    checkpoint = {
        'slot_attention': slot_attention.state_dict(),
        'student_agents': student_agents.state_dict(),
        'teacher_agents': teacher_agents.state_dict(),
        'estimators': estimators.state_dict(),
        'dino_loss_centers': centers,
        'epoch': epoch,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Full checkpoint saved to: {save_path}")


def list_checkpoints(checkpoint_dir: str, pattern: str = "*.pth") -> list[Path]:
    """
    List all checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern (default: "*.pth")
    
    Returns:
        List of checkpoint paths, sorted by modification time (newest first)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return checkpoints


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pth") -> Optional[Path]:
    """
    Get the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = list_checkpoints(checkpoint_dir, pattern)
    return checkpoints[0] if checkpoints else None


# Example usage
if __name__ == "__main__":
    print("Checkpoint utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - load_slot_attention_checkpoint()")
    print("  - load_agent_checkpoint()")
    print("  - load_estimator_checkpoint()")
    print("  - save_slot_attention_checkpoint()")
    print("  - save_agent_checkpoint()")
    print("  - save_estimator_checkpoint()")
    print("  - save_full_checkpoint()")
    print("  - list_checkpoints()")
    print("  - get_latest_checkpoint()")


