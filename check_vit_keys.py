"""Inspect ViT feature extractor keys in COCO/MOVi checkpoints."""
import torch

ckpt = torch.load("checkpoints/slot_attention/adaslot_real/COCO.ckpt",
                  map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]

vit_keys = {k: v.shape for k, v in sd.items() if "feature_extractor" in k}
print(f"Total ViT keys: {len(vit_keys)}\n")
for k, v in vit_keys.items():
    print(f"  {k:70s} {str(list(v))}")
