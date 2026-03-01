"""
Quick test for contrastive loss integration with AdaSlot.
"""
import torch
from src.losses.contrastive import SlotClusteringLoss

def test_contrastive_loss():
    print("Testing SlotClusteringLoss...")
    
    # Simulated data
    batch_size = 16
    num_slots = 7
    slot_dim = 64
    num_classes = 10
    
    # Create loss function
    loss_fn = SlotClusteringLoss(
        loss_type='contrastive',
        temperature=0.07,
        num_classes=num_classes,
        embedding_dim=slot_dim,
        aggregation='mean'
    )
    
    # Simulated slots and labels
    slots = torch.randn(batch_size, num_slots, slot_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Compute loss
    loss = loss_fn(slots, labels)
    
    print(f"✓ Contrastive loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test prototype loss
    print("\nTesting Prototype loss...")
    loss_fn_proto = SlotClusteringLoss(
        loss_type='prototype',
        temperature=0.1,
        num_classes=num_classes,
        embedding_dim=slot_dim,
        aggregation='mean'
    )
    
    loss_proto = loss_fn_proto(slots, labels)
    print(f"✓ Prototype loss: {loss_proto.item():.4f}")
    assert loss_proto.item() > 0, "Loss should be positive"
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    slots.requires_grad = True
    loss = loss_fn(slots, labels)
    loss.backward()
    
    assert slots.grad is not None, "Gradients should exist"
    assert slots.grad.abs().sum() > 0, "Gradients should be non-zero"
    print(f"✓ Gradient norm: {slots.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")

if __name__ == '__main__':
    test_contrastive_loss()
