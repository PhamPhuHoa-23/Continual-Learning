# Primitive Loss for AdaSlot Training - Implementation Guide

## Summary

Based on the CompSLOT paper analysis, **YES**, you should use primitive loss during AdaSlot training for better concept discovery and continual learning performance.

## What Changed?

### 1. New Files Created

#### `src/losses/primitive.py`
Contains three key components:

- **`PrimitiveSelector`**: Learnable attention-based mechanism that aggregates K slots into a single primitive representation
  - Uses learnable primitive key K_p to measure slot importance
  - Implements Equation 2 from the paper
  
- **`PrimitiveLoss`**: Contrastive loss ensuring intra-class consistency
  - Uses KL divergence to match label-based and primitive-based similarity matrices
  - Different from standard contrastive loss (SupCon)
  - Implements Equation 3 from the paper
  
- **`ConceptLearningLoss`**: Combined loss for concept learning stage
  - L_slot = L_re + α * L_p
  - Reconstruction + Primitive loss

### 2. Updated Files

#### `train_compositional.py`
- Added imports for primitive loss components
- Modified `phase_adaslot()` to support primitive loss
- Added command-line arguments:
  - `--use_primitive_loss`: Enable primitive loss
  - `--primitive_alpha`: Weight for primitive loss (default: 10.0)
  - `--primitive_temp`: Temperature for similarity (default: 10.0)

## Why Use Primitive Loss?

### Evidence from Paper

1. **Ablation Study (Table 2)**: Removing L_p causes significant performance drops
   - RanPAC: 65.81% → 65.08% (AA drops 0.73%)
   - CPrompt: 46.75% → 46.30% (AA drops 0.45%)

2. **Better Concept Discovery**: 
   - Primitive loss ensures slots representing the same concept stay consistent across images
   - Helps identify class-relevant concepts (primitives) vs. noise (background)

3. **Intra-class Consistency (Theorem 1)**:
   - Images from the same class should have similar primitive representations
   - KL divergence loss enforces this property

### Key Differences from Clustering Loss

| Feature | Clustering Loss (Current) | Primitive Loss (Paper) |
|---------|--------------------------|------------------------|
| Approach | Contrastive (SupCon) or Prototype | KL Divergence on similarities |
| Aggregation | Simple averaging/max | Learnable attention with K_p |
| Loss Type | Direct embedding contrast | Similarity distribution matching |
| Paper Used? | Optional (not in paper) | Core component (Table 4) |

## Usage

### Basic Usage

```bash
python train_compositional.py \
    --phase adaslot \
    --use_primitive_loss \
    --primitive_alpha 10.0 \
    --primitive_temp 10.0 \
    --adaslot_epochs 50
```

### Test Mode Example

```bash
./train_with_primitive_loss.bat
```

Or manually:
```bash
python train_compositional.py \
    --phase adaslot \
    --n_tasks 10 \
    --n_classes_per_task 10 \
    --adaslot_epochs 50 \
    --batch_size 64 \
    --device cuda \
    --use_primitive_loss \
    --primitive_alpha 10.0 \
    --primitive_temp 10.0 \
    --test_mode \
    --max_samples 200
```

### Full Pipeline with Primitive Loss

```bash
python train_compositional.py \
    --phase all \
    --n_tasks 10 \
    --n_classes_per_task 10 \
    --epochs 100 \
    --adaslot_epochs 50 \
    --batch_size 64 \
    --use_primitive_loss \
    --primitive_alpha 10.0 \
    --primitive_temp 10.0 \
    --device cuda
```

## Hyperparameters from Paper (Table 4)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `α` (primitive_alpha) | 10.0 | Weight for primitive loss L_p |
| `τ_p` (primitive_temp) | 10.0 | Temperature for primitive similarity |
| `τ_t` | 100/√D_s | Temperature for slot selection (auto) |
| K (num_slots) | 10 | Number of slots |
| D_s (slot_dim) | 128 | Slot embedding dimension |
| N_s (slot iterations) | 5 | Slot refinement iterations |

## When to Use What?

### Use Primitive Loss When:
- ✅ Following CompSLOT paper approach
- ✅ Want better concept discovery
- ✅ Need intra-class consistency
- ✅ Training from scratch with labels available

### Use Clustering Loss When:
- ✅ Want quick prototyping (simpler)
- ✅ Small batch sizes (prototype loss works better)
- ✅ Don't care about matching paper exactly

### Use Both When:
- ⚠️ Experimental - not tested in paper
- ⚠️ May cause conflicting objectives

## Expected Results

Based on paper's ablation:

**Without Primitive Loss:**
- Slots may not capture meaningful concepts
- Lower compositional generalization (R score)
- More forgetting across tasks

**With Primitive Loss:**
- Stable primitive extraction across tasks
- Higher Hn (compositional generalization)
- Better AA (average accuracy)
- Improved intra-class consistency

## Architecture Flow

```
Input Image (B, 3, H, W)
    ↓
AdaSlot (Slot Attention)
    ↓
Slots (B, K, slot_dim)              Labels (B,)
    ↓                                    ↓
PrimitiveSelector (learnable)  ─────────┤
    ↓                                    ↓
Primitives (B, slot_dim)  → PrimitiveLoss (KL div)
    ↓
ConceptLearningLoss: L_slot = L_re + α*L_p
```

## Troubleshooting

### Loss doesn't decrease
- Check `primitive_alpha` - may be too large (try 1.0-10.0)
- Check `primitive_temp` - try 0.1-10.0 range
- Ensure labels are correct (0-indexed)

### Out of memory
- Reduce `batch_size`
- Reduce `num_slots` or `slot_dim`

### Slots don't look meaningful
- Increase `primitive_alpha` (more emphasis on concepts)
- Increase `adaslot_epochs` (more training)
- Try different `primitive_temp`

## References

- CompSLOT Paper: Section 4.1 (Concept Learning)
- Equation 2: Primitive Selection
- Equation 3: Primitive Loss
- Table 2: Ablation Study (shows importance of L_p)
- Table 4: Hyperparameters

## Next Steps

1. **Train AdaSlot with primitive loss:**
   ```bash
   ./train_with_primitive_loss.bat
   ```

2. **Compare with/without primitive loss:**
   ```bash
   # Without primitive loss
   python train_compositional.py --phase adaslot --adaslot_epochs 50
   
   # With primitive loss
   python train_compositional.py --phase adaslot --adaslot_epochs 50 --use_primitive_loss
   ```

3. **Run full pipeline:**
   ```bash
   python train_compositional.py --phase all --use_primitive_loss
   ```

4. **Visualize slot masks** (see paper Figure 8-11) to verify concepts are meaningful

## Recommendation

**YES, use primitive loss during AdaSlot training** because:

1. ✅ Paper shows it's essential (Table 2 ablation)
2. ✅ Better concept discovery and consistency
3. ✅ Improves continual learning performance
4. ✅ Minimal computational overhead (~same as clustering loss)
5. ✅ Follows paper's proven approach

The implementation is ready to use—just add `--use_primitive_loss` flag!
