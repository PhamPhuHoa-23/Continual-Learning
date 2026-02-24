"""Quick script to dump checkpoint keys and model keys for comparison."""
import torch
from pathlib import Path
from src.models.adaslot import AdaSlotModel

ckpt_path = Path("checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = checkpoint.get('state_dict', checkpoint)

model = AdaSlotModel(resolution=(128, 128), num_slots=11, slot_dim=64, num_iterations=3)

model_keys = sorted(model.state_dict().keys())
checkpoint_keys = sorted(state_dict.keys())

with open("keys_comparison.txt", "w") as f:
    f.write("=== CHECKPOINT KEYS ===\n")
    for k in checkpoint_keys:
        shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'N/A'
        f.write(f"  {k}: {shape}\n")
    
    f.write("\n=== MODEL KEYS ===\n")
    model_sd = model.state_dict()
    for k in model_keys:
        shape = model_sd[k].shape
        f.write(f"  {k}: {shape}\n")

print("Written to keys_comparison.txt")
