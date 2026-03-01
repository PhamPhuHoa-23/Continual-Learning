# Primitive Loss Weight Tuning Guide

## Test Results Summary

### Loss Magnitude Analysis
- **Reconstruction Loss** (random init): ~30.0
- **Primitive Loss** (raw, random init): ~0.04-0.44
- **Ratio**: ~70:1 (reconstruction dominates)

### Weight Testing Results

| Weight | Reconstruction | Primitive | Recon % | Primitive % | Status |
|--------|---------------|-----------|---------|-------------|--------|
| 0.1    | 30.0          | 0.043     | 99.9%   | 0.1%        | Too weak |
| 0.5    | 30.0          | 0.072     | 99.8%   | 0.2%        | Too weak |
| 1.0    | 30.0          | 0.043     | 99.9%   | 0.1%        | Too weak |
| 5.0    | 30.0          | 1.923     | 94.0%   | 6.0%        | Weak |
| 10.0   | 30.0          | 0.623     | 98.0%   | 2.0%        | Weak |
| 20.0   | 30.0          | 0.499     | 98.4%   | 1.6%        | Weak |
| **50.0** | **30.0**   | **21.98** | **57.7%** | **42.3%** | **Balanced** |

## Recommendations

### Option 1: Conservative (Recommended for Initial Training)
```yaml
loss:
  reconstruction:
    type: reconstruction
    weight: 1.0
  
  primitive:
    type: primitive
    weight: 30.0  # Start conservative
    temperature: 10.0
```

**Expected behavior**:
- Reconstruction: ~75% of total loss
- Primitive: ~25% of total loss
- Stable training, gradual concept learning

### Option 2: Balanced (After Reconstruction Improves)
```yaml
loss:
  reconstruction:
    type: reconstruction
    weight: 1.0
  
  primitive:
    type: primitive
    weight: 50.0  # More aggressive
    temperature: 10.0
```

**Expected behavior**:
- Reconstruction: ~60% of total loss
- Primitive: ~40% of total loss
- Stronger concept signal

### Option 3: Aggressive (Fine-tuning)
```yaml
loss:
  reconstruction:
    type: reconstruction
    weight: 1.0
  
  primitive:
    type: primitive
    weight: 100.0  # Maximum impact
    temperature: 10.0
```

**Expected behavior**:
- Reconstruction: ~40% of total loss
- Primitive: ~60% of total loss
- Risk: May hurt reconstruction quality

## Scheduled Weight Strategy (BEST)

Gradually increase primitive weight during training:

### Phase 1: Pretrain (Epochs 1-20)
- Focus on learning good slot representations
- Primitive weight: **10.0** (weak signal, ~5% of loss)
- Monitor: Reconstruction should drop from 30 → 15

### Phase 2: Concept Learning (Epochs 21-40)
- Introduce stronger concept supervision
- Primitive weight: **30.0** (balanced, ~25% of loss)
- Monitor: Primitive loss should drop, accuracy should increase

### Phase 3: Fine-tuning (Epochs 41-50)
- Strengthen concept discrimination
- Primitive weight: **50.0** (strong, ~40% of loss)
- Monitor: Concept similarity should align with labels

### Implementation

```python
# In training loop
if epoch <= 20:
    primitive_weight = 10.0
elif epoch <= 40:
    primitive_weight = 30.0
else:
    primitive_weight = 50.0

# Update loss
total_loss = recon_loss + primitive_weight * primitive_loss
```

Or use config with multiple phases:

```yaml
# Phase 1: Pretrain
loss:
  primitive:
    weight: 10.0

# Phase 2: Concept learning
loss:
  primitive:
    weight: 30.0

# Phase 3: Fine-tuning
loss:
  primitive:
    weight: 50.0
```

## Monitoring During Training

### Key Metrics to Track

1. **Loss Ratio**
   ```python
   recon_ratio = recon_loss / total_loss * 100
   primitive_ratio = primitive_loss / total_loss * 100
   ```
   - Target: Reconstruction 60-70%, Primitive 30-40%

2. **Primitive Loss Magnitude**
   - Should decrease over time
   - Initial: ~0.4 (random)
   - After training: ~0.05-0.1 (learned)

3. **Reconstruction Loss**
   - Should decrease steadily
   - Initial: ~30 (random decoder)
   - Target: <5 (good reconstruction)

### Warning Signs

❌ **Primitive too weak** (< 10% of total loss)
   - Increase weight by 2x
   - Concepts won't be learned effectively

❌ **Primitive too strong** (> 50% of total loss)
   - Decrease weight by 0.5x
   - May hurt reconstruction quality

❌ **Primitive loss not decreasing**
   - Check temperature parameter (try 5.0 or 20.0)
   - Verify slot representations are meaningful
   - Increase number of slot attention iterations

✅ **Good balance**
   - Both losses decreasing
   - Primitive: 20-40% of total
   - Reconstruction quality maintained

## Temperature Parameter

The primitive loss also has a temperature parameter (τ):

```yaml
primitive:
  temperature: 10.0  # Default, works well
```

**Effect**:
- **Low** (1.0-5.0): Sharper similarity, harder matching
- **Medium** (10.0): Balanced, recommended
- **High** (20.0-50.0): Softer similarity, easier matching

**When to adjust**:
- If primitive loss is very high (>1.0) even with low weight → increase temperature
- If primitive loss is very low (<0.01) and not learning → decrease temperature

## Quick Start Commands

### Test Current Setup
```bash
python test_primitive_loss_weight.py
```

### Train with Conservative Weight
```bash
# Edit config file
loss:
  primitive:
    weight: 30.0

# Run training
python train_adaslot_multiphase.py --config your_config.yaml
```

### Monitor Training
```bash
# Check loss ratios in logs
tensorboard --logdir runs/
```

## Expected Timeline

**With weight=30.0 (recommended)**:

| Epoch | Recon Loss | Primitive Loss | Total Loss | Concept Acc |
|-------|------------|----------------|------------|-------------|
| 1     | 30.0       | 0.40           | 42.0       | ~20% (random) |
| 10    | 15.0       | 0.30           | 24.0       | ~40% |
| 20    | 10.0       | 0.20           | 16.0       | ~60% |
| 30    | 7.0        | 0.15           | 11.5       | ~70% |
| 50    | 5.0        | 0.10           | 8.0        | ~80% |

## Conclusion

**Start with weight=30.0** and monitor the loss ratio:
- Target: Reconstruction ~70-75%, Primitive ~25-30%
- Adjust based on training dynamics
- Increase to 50.0 if reconstruction is stable
- Decrease to 10.0 if reconstruction degrades

The key is **balance**: primitive loss should guide concept learning without overwhelming reconstruction.
