"""
Test script to check AdaSlot checkpoint compatibility.

Checks:
1. Can we load the checkpoint?
2. What's the structure?
3. Does it match our implementation?
"""

import torch
from pathlib import Path

# Available checkpoints
checkpoint_dir = Path("checkpoints/slot_attention/adaslot_real")
checkpoints = [
    "CLEVR10.ckpt",      # 11MB - CLEVR dataset
    "MOVi-C.ckpt",       # 374MB - MOVi-C dataset
    "MOVi-E.ckpt",       # 374MB - MOVi-E dataset  
    "COCO.ckpt",         # 473MB - COCO dataset
]

def inspect_checkpoint(ckpt_path):
    """Load and inspect checkpoint structure."""
    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Size: {ckpt_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*70}\n")
    
    try:
        # Load checkpoint (weights_only=False for PyTorch 2.6+)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"[OK] Checkpoint loaded successfully!\n")
        
        # Print top-level keys
        if isinstance(checkpoint, dict):
            print("Top-level keys:")
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, dict):
                    print(f"  - {key}: dict with {len(value)} keys")
                elif isinstance(value, torch.Tensor):
                    print(f"  - {key}: Tensor {tuple(value.shape)}")
                elif isinstance(value, (int, float, str)):
                    print(f"  - {key}: {type(value).__name__} = {value}")
                else:
                    print(f"  - {key}: {type(value).__name__}")
            
            # Check for state_dict
            if 'state_dict' in checkpoint:
                print(f"\nstate_dict structure:")
                state_dict = checkpoint['state_dict']
                
                # Group by module
                modules = {}
                for key in state_dict.keys():
                    module_name = key.split('.')[0]
                    if module_name not in modules:
                        modules[module_name] = []
                    modules[module_name].append(key)
                
                for module_name, keys in modules.items():
                    print(f"\n  Module: {module_name}/ ({len(keys)} parameters)")
                    # Show first 3 keys
                    for key in keys[:3]:
                        tensor = state_dict[key]
                        print(f"     - {key}: {tuple(tensor.shape)}")
                    if len(keys) > 3:
                        print(f"     ... and {len(keys) - 3} more")
                
                # Check for our expected modules
                print(f"\nModule compatibility check:")
                expected = {
                    'feature_extractor': False,
                    'perceptual_grouping': False,
                    'object_decoder': False,
                    'conditioning': False,
                }
                
                for key in state_dict.keys():
                    for module in expected.keys():
                        if key.startswith(module) or key.startswith(f"model.{module}"):
                            expected[module] = True
                
                for module, found in expected.items():
                    status = "[OK]" if found else "[MISSING]"
                    print(f"  {status} {module}")
                
                # Check for adaptive slot mechanism
                has_gumbel = any('gumbel' in k.lower() or 'single_gumbel' in k.lower() for k in state_dict.keys())
                status = "[OK]" if has_gumbel else "[MISSING]"
                print(f"\n  {status} Adaptive slot mechanism (Gumbel)")
                
                if has_gumbel:
                    print("\n  AdaSlot-specific parameters:")
                    for key in state_dict.keys():
                        if 'gumbel' in key.lower():
                            tensor = state_dict[key]
                            print(f"     - {key}: {tuple(tensor.shape)}")
            
            # Check hyperparameters
            if 'hyper_parameters' in checkpoint:
                print(f"\nHyperparameters:")
                hparams = checkpoint['hyper_parameters']
                important_keys = [
                    'object_dim', 'slot_dim', 'num_slots', 
                    'feature_dim', 'iters', 'n_heads'
                ]
                for key in important_keys:
                    if key in hparams:
                        print(f"  - {key}: {hparams[key]}")
        
        else:
            print(f"[WARNING] Checkpoint is not a dict, it's a {type(checkpoint)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_compatibility_with_our_code():
    """Check if checkpoint can be loaded into our implementation."""
    print(f"\n{'='*70}")
    print(f"Compatibility Test with Our Implementation")
    print(f"{'='*70}\n")
    
    try:
        from src.models.slot_attention import SlotAttentionAutoEncoder
        from src.utils.checkpoint import load_slot_attention_checkpoint
        
        print("[OK] Our modules imported successfully\n")
        
        # Try loading CLEVR10 (smallest)
        ckpt_path = checkpoint_dir / "CLEVR10.ckpt"
        
        print(f"Testing with: {ckpt_path.name}\n")
        
        # Create our model
        model = SlotAttentionAutoEncoder(
            num_slots=10,  # Max slots for adaptive
            slot_dim=64,
            hidden_dim=64,
            num_iterations=3
        )
        
        print(f"[OK] Our model created: {model.__class__.__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Try loading (non-strict, expect mismatches)
        try:
            model = load_slot_attention_checkpoint(
                model,
                str(ckpt_path),
                strict=False,
                device='cpu'
            )
            print(f"\n[OK] Checkpoint loaded into our model (non-strict mode)")
            print(f"   [WARNING] Some parameters may not match (expected for different architectures)")
            return True
        except Exception as e:
            print(f"\n[WARNING] Could not load checkpoint: {e}")
            return False
        
    except ImportError as e:
        print(f"[ERROR] Could not import our modules: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AdaSlot Checkpoint Inspector")
    print("="*70)
    
    # Check which checkpoints exist
    existing = []
    for ckpt_name in checkpoints:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            existing.append(ckpt_path)
    
    if not existing:
        print(f"\n[ERROR] No checkpoints found in: {checkpoint_dir}")
        print(f"\nExpected checkpoints:")
        for name in checkpoints:
            print(f"  - {name}")
        exit(1)
    
    print(f"\n[OK] Found {len(existing)} checkpoint(s):\n")
    for path in existing:
        print(f"  - {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Inspect first checkpoint (CLEVR10 if available)
    test_ckpt = existing[0]
    if any('CLEVR10' in str(p) for p in existing):
        test_ckpt = [p for p in existing if 'CLEVR10' in str(p)][0]
    
    inspect_checkpoint(test_ckpt)
    
    # Test compatibility
    check_compatibility_with_our_code()
    
    print(f"\n{'='*70}")
    print("[DONE] Inspection complete!")
    print(f"{'='*70}\n")
