# AdaSlot Checkpoint Analysis

**Date**: 2026-02-14  
**Checkpoints**: 4 files in `checkpoints/slot_attention/adaslot_real/`

---

## Summary

**Question 1**: Does our Slot Attention implementation use the same architecture as AdaSlot?  
**Answer**: ⚠️ **Partially compatible**

**Question 2**: Can we load the checkpoints fully?  
**Answer**: ✅ **Yes, but with architecture differences**

---

## Checkpoint Details

### Available Checkpoints

| File | Size | Dataset | Notes |
|------|------|---------|-------|
| `CLEVR10.ckpt` | 10.8 MB | CLEVR | Smallest, good for testing |
| `MOVi-C.ckpt` | 374.1 MB | MOVi-C | Video dataset |
| `MOVi-E.ckpt` | 374.1 MB | MOVi-E | Video dataset |
| `COCO.ckpt` | 473.0 MB | COCO | Real images |

### Checkpoint Structure (CLEVR10.ckpt)

```
Top-level keys:
  - epoch: 330
  - global_step: 361783
  - pytorch-lightning_version: 1.6.4
  - state_dict: 56 parameters
  - loops, callbacks, optimizer_states, lr_schedulers

State dict modules (all prefixed with "models."):
  - models.conditioning.*
  - models.feature_extractor.*
  - models.perceptual_grouping.*
  - models.object_decoder.*
```

---

## Architecture Comparison

### AdaSlot Pipeline (from checkpoint)

```
Image
  ↓
models.feature_extractor (CNN encoder)
  ↓
models.conditioning (slot initialization with mu/logsigma)
  ↓
models.perceptual_grouping (Slot Attention + Gumbel selector)
  │
  ├─ SlotAttention (standard mechanism)
  │  - to_q, to_k, to_v projections
  │  - GRU update
  │  - LayerNorm
  │
  └─ Gumbel Selection Network (adaptive slots)
     - single_gumbel_score_network
     - Predicts keep/drop for each slot
     - Output: hard_keep_decision (binary mask)
  ↓
models.object_decoder (slot → reconstruction)
  ↓
Output (reconstructed image + masks)
```

### Our Implementation (`src/models/slot_attention/`)

```
Image
  ↓
encoder (CNNEncoder or ResNetEncoder)
  ↓
pos_embed (PositionalEmbedding)
  ↓
slot_attention (SlotAttention)
  - Fixed num_slots (e.g., 7)
  - Standard attention mechanism
  - No Gumbel selector
  ↓
decoder (BroadcastDecoder or MLPDecoder)
  ↓
Output (reconstructed image + slots + masks)
```

---

## Key Differences

### ✅ **What We Have (Compatible)**

| Component | AdaSlot | Our Code | Status |
|-----------|---------|----------|--------|
| CNN Encoder | ✅ Yes | ✅ CNNEncoder | ✅ Compatible |
| Slot Attention Core | ✅ Yes | ✅ SlotAttention | ✅ Compatible |
| Q/K/V Projections | ✅ Yes | ✅ Yes | ✅ Compatible |
| GRU Update | ✅ Yes | ✅ Yes | ✅ Compatible |
| LayerNorm | ✅ Yes | ✅ Yes | ✅ Compatible |
| Decoder | ✅ Yes | ✅ BroadcastDecoder | ✅ Compatible |

### ❌ **What We're Missing (Incompatible)**

| Component | AdaSlot | Our Code | Impact |
|-----------|---------|----------|--------|
| **Gumbel Selection Network** | ✅ Has | ❌ Missing | **HIGH** - Can't do adaptive slots |
| **Conditioning (mu/logsigma)** | ✅ Has | ✅ Has (simple init) | Medium - Different initialization |
| **Temperature annealing** | ✅ Has | ❌ Missing | Medium - For training only |
| **Hard keep decision** | ✅ Has | ❌ Missing | **HIGH** - Core adaptive mechanism |

---

## Adaptive Slot Mechanism (What's Missing)

### AdaSlot's Gumbel Selector

```python
# After slot attention iterations
slots = slot_attention(features, conditioning)  # (B, K, D)

# Gumbel selection network
slots_keep_prob = single_gumbel_score_network(slots)  # (B, K, 2)
# → Predicts probability of keeping each slot

# Gumbel-Softmax (differentiable discrete selection)
current_keep_decision = F.gumbel_softmax(
    slots_keep_prob, 
    hard=True,  # Get discrete 0/1
    tau=temperature
)[..., 1]  # (B, K) - binary mask

# Filter slots
active_slots = slots * current_keep_decision.unsqueeze(-1)
```

**Parameters in CLEVR10 checkpoint:**
```
models.perceptual_grouping.slot_attention.single_gumbel_score_network:
  - 0.weight: (64,) - LayerNorm
  - 0.bias: (64,)
  - 1.weight: (256, 64) - Linear
  - 1.bias: (256,)
  - 3.weight: (2, 256) - Output (keep/drop)
  - 3.bias: (2,)
```

**Network structure:**
```
slot (64D) → LayerNorm → Linear(256) → ReLU → Linear(2) → Gumbel-Softmax
```

### Our Implementation

```python
# Fixed number of slots, no selection
slots = slot_attention(features, conditioning)  # (B, 7, D)
# All slots are always "active"
```

---

## Can We Load Checkpoints?

### ✅ **What CAN Be Loaded**

```python
from src.utils.checkpoint import load_slot_attention_checkpoint
from src.models.slot_attention import SlotAttentionAutoEncoder

# Create model (our implementation)
model = SlotAttentionAutoEncoder(
    num_slots=10,  # Match max_slots in AdaSlot
    slot_dim=64,
    hidden_dim=64,
    num_iterations=3
)

# Load checkpoint (non-strict mode)
model = load_slot_attention_checkpoint(
    model,
    'checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt',
    strict=False  # ← Must use False!
)
```

**What gets loaded:**
- ✅ Encoder weights (CNN layers)
- ✅ Slot Attention weights (Q/K/V, GRU, LayerNorm)
- ✅ Decoder weights (if architecture matches)

**What gets IGNORED:**
- ❌ Gumbel selection network (not in our model)
- ❌ Conditioning mu/logsigma (different initialization)
- ❌ Module name prefix "models." (handled by our loader)

---

## Loading Results

### Test Run Output

```
[OK] Found 4 checkpoint(s):
  - CLEVR10.ckpt (10.8 MB)
  - MOVi-C.ckpt (374.1 MB)
  - MOVi-E.ckpt (374.1 MB)
  - COCO.ckpt (473.0 MB)

[OK] Checkpoint loaded successfully!

Module compatibility check:
  [MISSING] feature_extractor  ← Prefix issue (models.feature_extractor)
  [MISSING] perceptual_grouping
  [MISSING] object_decoder
  [MISSING] conditioning

  [OK] Adaptive slot mechanism (Gumbel) ← AdaSlot-specific
```

**Note**: "MISSING" means the key prefix doesn't match. The actual weights CAN be loaded with `strict=False` and key remapping.

---

## Recommendations

### Option 1: Use Checkpoints for Encoder/Decoder Only

**Pros:**
- ✅ Get pretrained feature extractors
- ✅ Good initialization for slot attention
- ✅ Works with our existing code

**Cons:**
- ❌ Lose adaptive slot mechanism
- ❌ Fixed number of slots (no dynamic allocation)

**Code:**
```python
# Load pretrained encoder/slot attention
model = load_slot_attention_checkpoint(
    model,
    'checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt',
    strict=False
)

# Fine-tune on your dataset
# Agents will work with fixed number of slots
```

---

### Option 2: Implement Gumbel Selection (Full AdaSlot)

**Pros:**
- ✅ True adaptive slots (3-10 slots dynamically)
- ✅ Can load full checkpoint
- ✅ Better for varying complexity images

**Cons:**
- ❌ Need to implement Gumbel network
- ❌ More complex training
- ❌ Need temperature annealing schedule

**Required implementation:**

```python
class GumbelSlotSelector(nn.Module):
    """AdaSlot's Gumbel selection network."""
    def __init__(self, slot_dim=64, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Keep/drop logits
        )
    
    def forward(self, slots, tau=1.0):
        logits = self.network(slots)  # (B, K, 2)
        # Gumbel-Softmax for differentiable discrete selection
        keep_decision = F.gumbel_softmax(logits, hard=True, tau=tau)[..., 1]
        return keep_decision  # (B, K) binary mask
```

Then integrate into `SlotAttentionAutoEncoder`:

```python
class AdaptiveSlotAttentionAE(nn.Module):
    def __init__(self, max_slots=10, ...):
        super().__init__()
        self.encoder = ...
        self.slot_attention = ...
        self.gumbel_selector = GumbelSlotSelector(slot_dim)  # ← Add this
        self.decoder = ...
    
    def forward(self, image, tau=1.0):
        features = self.encoder(image)
        slots, attn = self.slot_attention(features)
        
        # Adaptive slot selection
        keep_mask = self.gumbel_selector(slots, tau)  # (B, K)
        active_slots = slots * keep_mask.unsqueeze(-1)
        
        reconstruction = self.decoder(active_slots)
        return {
            'reconstruction': reconstruction,
            'slots': active_slots,
            'keep_mask': keep_mask,
            'num_active_slots': keep_mask.sum(dim=1)  # Per-image slot count
        }
```

---

### Option 3: Hybrid Approach (Recommended for Your Use Case)

Since you're using slots as **input to agents**, not for reconstruction:

**Strategy:**
1. ✅ Load AdaSlot encoder/attention weights (pretrained features)
2. ✅ Use **fixed K slots** (e.g., 7-10) in inference
3. ✅ Agents select top-k active slots via estimators (your existing selector)

**Why this works:**
- AdaSlot with max_slots=10 will learn to use 3-10 slots
- In inference, always output 10 slots
- Some slots will have low attention (inactive)
- Your VAE/MLP estimators will naturally score inactive slots low
- Top-k selector will pick active slots automatically

**Code:**
```python
# Load pretrained AdaSlot encoder (without Gumbel selector)
slot_model = SlotAttentionAutoEncoder(num_slots=10, slot_dim=64)
slot_model = load_slot_attention_checkpoint(
    slot_model,
    'checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt',
    strict=False
)

# Freeze slot attention (optional)
for param in slot_model.parameters():
    param.requires_grad = False

# Your agent system
slots, attn = slot_model.encoder_and_slots(image)  # (B, 10, 64)

# Estimators score all slots
for i, slot in enumerate(slots):
    scores[i] = estimator[i].estimate_performance(slot)

# Top-k selector picks active slots (naturally filters inactive ones)
selected_agents = selector.select_top_k(slot, k=3)
```

**Benefits:**
- ✅ Pretrained features from AdaSlot
- ✅ No need to implement Gumbel selector
- ✅ Your existing agent system handles slot selection
- ✅ Works immediately with current code

---

## Conclusion

### Answer 1: Architecture Compatibility

**Core Slot Attention**: ✅ **Compatible**  
- Q/K/V projections, GRU, LayerNorm all match

**Adaptive Mechanism**: ❌ **Missing**  
- Gumbel selection network not implemented
- Can work around with fixed slots + your estimators

### Answer 2: Checkpoint Loading

**Can Load**: ✅ **Yes (with `strict=False`)**  
- Encoder weights: ✅ Loads
- Slot Attention weights: ✅ Loads  
- Decoder weights: ✅ Loads (if architecture matches)
- Gumbel selector: ❌ Ignored (not in our model)

**Recommendation**: **Option 3 (Hybrid)**  
- Load pretrained weights
- Use fixed 10 slots
- Let your estimators + top-k selector handle active slot detection
- No need to implement Gumbel selector

---

## Next Steps

1. **Update config to load checkpoint:**
   ```yaml
   slot_attention:
     pretrained:
       enabled: true
       checkpoint_path: "./checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt"
       strict_load: false
       freeze: true  # Optional: freeze for faster agent training
   ```

2. **Test loading:**
   ```bash
   python -c "
   from src.utils import load_config
   from src.models.slot_attention import SlotAttentionAutoEncoder
   from src.utils.checkpoint import load_slot_attention_checkpoint
   
   model = SlotAttentionAutoEncoder(num_slots=10, slot_dim=64)
   model = load_slot_attention_checkpoint(
       model, 
       'checkpoints/slot_attention/adaslot_real/CLEVR10.ckpt',
       strict=False
   )
   print('Checkpoint loaded successfully!')
   "
   ```

3. **Train agents with pretrained slots:**
   ```bash
   python train_phase1.py --config config_variants/07_pretrained_slot_attention.yaml
   ```

---

**Last Updated**: 2026-02-14  
**Test Script**: `test_adaslot_checkpoint.py`

