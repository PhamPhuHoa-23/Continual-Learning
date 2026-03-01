"""
Compute reconstruction loss of pretrained AdaSlot checkpoints on random noise
(and real CIFAR data if available) to estimate target loss levels.
"""
from cont_src.models.adaslot_configs import build_adaslot_from_checkpoint, ADASLOT_CONFIGS
import sys
import torch
import os
sys.path.insert(0, '.')

for name, cfg in ADASLOT_CONFIGS.items():
    ckpt_path = cfg.default_ckpt_path
    if not os.path.exists(ckpt_path):
        print(f'{name}: checkpoint not found, skip')
        continue

    print(f'\n=== {name} ({cfg.source_dataset}) ===')
    model = build_adaslot_from_checkpoint(name)
    model.eval()

    B = 4
    H, W = cfg.resolution
    dummy = torch.rand(B, 3, H, W)

    with torch.no_grad():
        out = model(dummy)
        recon = out['reconstruction']
        mse = ((recon - dummy) ** 2).mean().item()
        mae = ((recon - dummy).abs()).mean().item()
        active = out['hard_keep_decision'].sum(1).float().mean().item()

    print(f'  MSE  (random input proxy) : {mse:.6f}')
    print(f'  MAE  (random input proxy) : {mae:.6f}')
    print(
        f'  Recon range              : [{recon.min():.3f}, {recon.max():.3f}]')
    print(f'  Avg active slots         : {active:.1f} / {cfg.num_slots}')
