import torch
from pathlib import Path
from src.models.slot_attention import SlotAttentionAutoEncoder

if __name__ == "__main__":
    ckpt_path = Path("checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    model = SlotAttentionAutoEncoder(num_slots=10, slot_dim=64, hidden_dim=64, num_iterations=3)
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    with open("mismatch_output.txt", "w") as f:
        f.write("MISSING KEYS (In model but not in checkpoint):\n")
        f.write("\n".join(sorted(list(model_keys - checkpoint_keys))))
        f.write("\n\nUNEXPECTED KEYS (In checkpoint but not in model):\n")
        f.write("\n".join(sorted(list(checkpoint_keys - model_keys))))
        
    print("Mismatch output written.")
