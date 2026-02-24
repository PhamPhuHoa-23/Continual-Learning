"""Test if slots from encode_slots have gradient issues"""
import torch
import torch.nn.functional as F
from src.models.adaslot.model import AdaSlotModel
from torch.utils.data import TensorDataset, DataLoader

# Create fake adaslot
adaslot = AdaSlotModel(num_slots=7, slot_dim=64)
adaslot.eval()

# Freeze it
for p in adaslot.parameters():
    p.requires_grad = False

# Create fake data
X = torch.randn(8, 3, 32, 32)
y = torch.randint(0, 10, (8,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4)

print("=== Testing encode_slots pattern ===")
# Simulate encode_slots
with torch.no_grad():
    for X_batch, y_batch in loader:
        X_resized = F.interpolate(X_batch, size=(128, 128), mode='bilinear', align_corners=False)
        slots = adaslot.encode(X_resized)
        print(f"Slots from AdaSlot (no_grad context):")
        print(f"  Shape: {slots.shape}")
        print(f"  requires_grad: {slots.requires_grad}")
        print(f"  grad_fn: {slots.grad_fn}")
        
        # Move to CPU then back (like training code does)
        slots_cpu = slots.cpu()
        slots_back = slots_cpu.to('cpu')
        print(f"\nSlots after .cpu().to('cpu'):")
        print(f"  requires_grad: {slots_back.requires_grad}")
        print(f"  grad_fn: {slots_back.grad_fn}")
        
        # Test if we can create gradient with detach
        slots_detached = slots_back.detach()
        print(f"\nSlots after .detach():")
        print(f"  requires_grad: {slots_detached.requires_grad}")
        print(f"  grad_fn: {slots_detached.grad_fn}")
        
        # Now pass through a simple layer
        linear = torch.nn.Linear(64, 128)
        h = linear(slots_detached[0, 0])  # Process one slot
        print(f"\nOutput after linear layer:")
        print(f"  requires_grad: {h.requires_grad}")
        print(f"  grad_fn: {h.grad_fn}")
        
        # Try backward
        if h.grad_fn:
            loss = h.sum()
            loss.backward()
            print(f"\n✅ Backward works! Linear layer has gradient: {linear.weight.grad is not None}")
        else:
            print(f"\n❌ h has no grad_fn!")
        
        break
