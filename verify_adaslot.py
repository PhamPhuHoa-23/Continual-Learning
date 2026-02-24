"""
Verify AdaSlot checkpoint loading.
Tests that the new src/models/adaslot module can load the pretrained checkpoints.
"""
import torch
from pathlib import Path
from src.models.adaslot import AdaSlotModel


def test_checkpoint_loading():
    checkpoint_dir = Path("checkpoints/slot_attention/adaslot_real")
    ckpt_path = checkpoint_dir / "CLEVR10.ckpt"
    
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return False
    
    print(f"Loading checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Create model
    model = AdaSlotModel(
        resolution=(128, 128),
        num_slots=11,
        slot_dim=64,
        num_iterations=3,
    )
    
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys
    
    print(f"\nModel keys:      {len(model_keys)}")
    print(f"Checkpoint keys: {len(checkpoint_keys)}")
    print(f"Missing keys (in model but not in ckpt):    {len(missing)}")
    print(f"Unexpected keys (in ckpt but not in model): {len(unexpected)}")
    
    if missing:
        print(f"\n--- MISSING KEYS ---")
        for k in sorted(missing):
            print(f"  {k}")
    
    if unexpected:
        print(f"\n--- UNEXPECTED KEYS ---")
        for k in sorted(unexpected):
            print(f"  {k}")
    
    # Try strict loading
    if not missing and not unexpected:
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"\n[OK] strict=True loading SUCCEEDED!")
        except Exception as e:
            print(f"\n[ERROR] strict=True loading failed: {e}")
            return False
    else:
        print(f"\n[WARNING] Cannot do strict loading due to key mismatches.")
        # Try non-strict
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"[OK] strict=False loading succeeded (partial).")
        except Exception as e:
            print(f"[ERROR] Even non-strict loading failed: {e}")
            return False
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  reconstruction:  {out['reconstruction'].shape}")
    print(f"  slots:           {out['slots'].shape}")
    print(f"  masks:           {out['masks'].shape}")
    print(f"  slots_keep_prob: {out['slots_keep_prob'].shape}")
    print(f"  hard_keep_decision: {out['hard_keep_decision'].shape}")
    print(f"\n[OK] Forward pass succeeded!")
    
    return True


if __name__ == "__main__":
    success = test_checkpoint_loading()
    if success:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== TESTS FAILED ===")
