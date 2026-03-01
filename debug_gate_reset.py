from src.models.adaslot.model import AdaSlotModel
import torch
import sys
import torch.nn as nn
sys.path.insert(0, '.')
b = AdaSlotModel((128, 128), 11, 64, 3, 64, 128, 1)
ck = torch.load('checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt',
                map_location='cpu', weights_only=False)
b.load_state_dict(ck['state_dict'], strict=True)
imgs = torch.randn(4, 3, 128, 128)
out = b(imgs)
print('BEFORE reset: skp=%.4f  active=%.1f/11' % (
    out['slots_keep_prob'].mean().item(), out['hard_keep_decision'].sum(1).float().mean().item()))
for nm, m in b.named_modules():
    if 'single_gumbel' in nm:
        for p in m.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
out2 = b(imgs)
print('AFTER  reset: skp=%.4f  active=%.1f/11' % (
    out2['slots_keep_prob'].mean().item(), out2['hard_keep_decision'].sum(1).float().mean().item()))
