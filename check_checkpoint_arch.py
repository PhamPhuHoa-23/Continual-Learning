import torch

checkpoints = {
    "CLEVR10": "checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt",
    "COCO": "checkpoints/slot_attention/adaslot_real/COCO.ckpt",
    "MOVi-C": "checkpoints/slot_attention/adaslot_real/MOVi-C.ckpt",
    "MOVi-E": "checkpoints/slot_attention/adaslot_real/MOVi-E.ckpt",
}

print("=" * 80)
print("CHECKPOINT ARCHITECTURE ANALYSIS")
print("=" * 80)

for name, path in checkpoints.items():
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ckpt["state_dict"].keys())
    n_keys = len(keys)

    # Check encoder type
    feature_keys = [k for k in keys if "feature_extractor" in k]

    if any("cls_token" in k or "pos_embed" in k or "patch_embed" in k for k in feature_keys):
        encoder_type = "ViT (Vision Transformer)"
    elif any("layers" in k for k in feature_keys):
        encoder_type = "CNN (Convolutional)"
    else:
        encoder_type = "Unknown"

    # Check num_slots from conditioning
    slots_mu_key = [k for k in keys if "conditioning.slots_mu" in k]
    if slots_mu_key:
        slots_mu = ckpt["state_dict"][slots_mu_key[0]]
        num_slots = slots_mu.shape[0]
    else:
        num_slots = "Unknown"

    print(f"\n{name}:")
    print(f"  Total keys    : {n_keys}")
    print(f"  Num slots     : {num_slots}")
    print(f"  Encoder       : {encoder_type}")
    print(f"  Sample keys   : {feature_keys[:3]}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("  CLEVR10, MOVi-C, MOVi-E → CNN encoder (original Slot Attention)")
print("  COCO                     → ViT encoder (different architecture!)")
print("=" * 80)
