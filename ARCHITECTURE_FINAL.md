# Final Architecture: Slot Multi-Agent System for Continual Learning

## Overview

A continual learning system that chains five major modules:

1. **AdaSlot** – Adaptive object-centric decomposition with Gumbel-based slot pruning
2. **Atomic Agents (DINO SSL)** – Pool of 50 specialized agents trained self-supervisedly on slot tokens
3. **Performance Estimators (VAE + MLP)** – Lightweight fast-filter scoring each agent–slot pair; eliminates expensive all-agent inference
4. **UCB Weighted MoE** – Weighted committee of top-K filtered agents; weights learned via UCB bandit (all K members contribute)
5. **CRP Expert Aggregator** – Chinese Restaurant Process + Gradient Projection for incremental class-learning without forgetting

Default dataset: **CIFAR-100 continual benchmark** (split into tasks, resolution 128x128).

---

## Architecture Diagram

```
INPUT IMAGE (B x 3 x 128 x 128)
         |
+--------+---------+
|   AdaSlot        |  CNN -> RandomConditioning -> SlotAttention (3 iters)
|   Model          |  + Gumbel Pruning -> hard_keep_decision (B, 11) in {0,1}
+--------+---------+
         |  slots (B, S, 64)   S <= 11
         |
   [For each slot independently]
         |
+--------+--------------------------------------------------+
| Estimators  (trained in Phase 2.5, frozen in Phase 3)     |
|   VAE_i(slot) reconstructs slot; score = f(recon error)   |
|   MLP(slot, agent_id) -> quality score                    |
|   hybrid_score[50] = 0.5 x VAE + 0.5 x MLP               |
|   -> top-10 agent IDs (filter_k=10)                       |
+--------+--------------------------------------------------+
         |  filtered_ids [K]
+--------+--------------------------------------------------+
| UCB Weighted MoE                                          |
|   UCB(i) = mu_i + c * sqrt(ln t / n_i)                   |
|   burn-in (t < 100): uniform weights                      |
|   post burn-in: weights = softmax(UCB[K] / temp)          |
|   weighted_out = sum_i w_i * agent_i(slot)  -> (256,)     |
+--------+--------------------------------------------------+
         |  (256,) per slot
         |  concat S slots -> 11 x 256 = 2816
+--------+--------------------------------------------------+
| CRP Expert Aggregator                                     |
|   Score(k) = Similarity x Alignment x Capacity           |
|   Route to existing expert OR create new expert           |
|   Expert: 2-layer MLP + Gradient Projection (GPM)        |
|   -> prediction -> reward -> ucb_moe.update_batch()       |
+--------+--------------------------------------------------+
         |
   FINAL PREDICTION (class in [0, 99])
```

---

## Component Details

### 1. AdaSlot

**Source:** `src/models/adaslot/model.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_slots` | 11 | Max slots; Gumbel reduces at runtime |
| `slot_dim` | 64 | Slot embedding dimension |
| `kvq_dim` | 128 | Key/Query/Value projection dim |
| `num_iterations` | 3 | Slot Attention refinement rounds |
| `low_bound` | 1 | Minimum kept slots |

Gumbel score network applies hard binary keep/drop to each slot (differentiable via Gumbel-Softmax). Only kept slots pass downstream.

**Pre-trained checkpoint:** `checkpoints/slot_attention/adaslot_real/AdaSlotCkpt/CLEVR10_Custom.ckpt`

---

### 2. Atomic Agents (ResidualMLPAgent + DINO)

**Source:** `src/slot_multi_agent/atomic_agent.py`

```
slot (B, 64)
  -> Linear(64->256) + LayerNorm
  -> 3x ResidualBlock(256)
  -> DINO head: Linear(256->256->128->256)   bottleneck projection
  -> softmax -> hidden label (B, 256)
```

Pool: **50 student + 50 teacher agents**.
DINO: student_temp=0.1, teacher_temp=0.07, EMA momentum=0.996.
Each step randomly samples 5/50 agents; trained on all active slots from batch.

---

### 3. Performance Estimators (Phase 2.5)

**Source:** `src/slot_multi_agent/estimators.py`

**Quality signal:**
```
quality_i(slot) = 1 - H(agent_i(slot)) / log(256)
```
Values: peaked output -> quality ~1;  uniform output -> quality ~0.

**VAEEstimator** (one per agent, 50 total):
- Encoder: slot(64) -> fc_mu(16) / fc_logvar(16)
- Decoder: z(16) -> recon(64)
- Loss: `sum_i quality_i * MSE(recon_i, slot)  +  0.5 * KL`
- Score: `sigmoid(threshold - MSE_recon)` in [0,1]

**MLPEstimator** (shared, all agents):
- Input: `concat[slot(64), agent_embedding(32)] = 96-dim`
- MLP: `Linear(96->128) -> LN -> ReLU -> Linear(128->64) -> LN -> ReLU -> Linear(64->1) -> Sigmoid`
- Loss: `MSE(predicted_quality, true_quality)`

**Hybrid score at inference:**
```
hybrid_score[i] = 0.5 * vae_score[i] + 0.5 * mlp_score[i]
filtered_K = argsort(hybrid_score, descending=True)[:filter_k]
```

**Checkpoint:** `checkpoints/estimators/estimators_final.pth`

---

### 4. UCB Weighted MoE (Phase 3)

**Source:** `src/slot_multi_agent/bandit_selector.py` -> `UCBWeightedMoE`

```
UCB(i) = mu_i + c * sqrt(ln t / n_i)
   mu_i = empirical mean reward
   n_i  = rounds participated
   t    = total rounds,  c = sqrt(2) default

Burn-in (t < 100):  weights = uniform
Post burn-in:       weights = softmax(UCB[K] / temperature)

weighted_out = sum_i w_i * agent_i(slot)   ->  (256,)

Reward update per sample:
  reward = 1.0 if pred correct else 0.0
  ucb_moe.update_batch(filtered_ids, weights, reward)
  # each agent i: local_reward = reward * w_i
```

UCB state saved to `checkpoints/ucb_moe_state.npz`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_constant` | 1.414 | UCB exploration bonus c |
| `temperature` | 1.0 | Softmax sharpness |
| `burn_in` | 100 | Uniform-weight warmup rounds |

---

### 5. CRP Expert Aggregator — MoE Cross-Attention

**Source:** `src/slot_multi_agent/aggregator.py`

**Design:** Each expert holds a set of **learnable query embeddings** (not prototype
centroids). Expert routing uses attention entropy as an OOD signal; a high-entropy
response means no expert "owns" this input → CRP spawns a new expert. Classification
uses per-class learnable query vectors that attend to the aggregated representation.

```
LearnableExpert:
    queries  ∈ ℝ^{n_queries × d}             # expert identity (learnable)
    key_proj : Linear(agent_dim → d)
    val_proj : Linear(agent_dim → d)

    forward(H: B×S×D):
        K = key_proj(H)                       # B×S×d
        V = val_proj(H)                       # B×S×d
        A = softmax(queries @ K^T / √d)       # n_q × S
        z = mean(A @ V)                       # (d,) aggregated repr
        H_e = -(A * log A+ε).sum()            # entropy (OOD signal)

CRPExpertAggregator:
    class_queries ∈ ℝ^{num_classes × d}       # per-class learnable codes

    routing (inference):
        all_z, all_scores, all_H = forward_all_experts(H)
        repr = MoE_aggregate(all_z, all_scores)   # softmax-gated sum
        logits = repr @ class_queries^T

    CRP trigger (training):
        if mean(entropy) > threshold AND Bernoulli(α / (N + α)):
            create new LearnableExpert()

    loss:
        CE(logits[seen_classes], y)  — backprop through cross-attention
        old class_queries frozen at task boundary (freeze_old_classes)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_slots` | `S` | Number of slots (=sequence length for cross-attention) |
| `agent_dim` | `proto_dim` | Agent output dimension (=key/value input dim) |
| `embed_dim` | 256 | Cross-attention hidden size |
| `n_queries` | 8 | Learnable queries per expert |
| `alpha` | 1.0 | CRP concentration |
| `max_experts` | 30 | Hard expert cap |
| `entropy_threshold` | 3.0 | nats — above this triggers CRP check |
| `num_classes` | — | Total class capacity |

---

## Training Pipeline (4 Phases)

### Phase 1 – AdaSlot Pretraining

```
Loss = MSE(recon, image)  +  10.0 * mean(hard_keep_decision)
Adam lr=4e-4  |  linear warmup 10k -> exp decay  |  500,000 steps
Output: checkpoints/adaslot/adaslot_final.pth
```

### Phase 2 – Agent DINO Training

```
AdaSlot FROZEN.  5/50 agents random per step.
AdamW lr=1e-3, wd=0.04  |  warmup 5k -> exp decay  |  100,000 steps
Output: checkpoints/agents/agents_final.pth
```

### Phase 2.5 – Estimator Training (NEW)

```
AdaSlot + Agents FROZEN.  10/50 agents sampled per step.

quality_i(slot) = 1 - H(agent_i(slot)) / log(256)

VAE loss : sum_i quality_i * MSE(recon_i, slot)  +  0.5 * KL
MLP loss : MSE(MLPEstimator(slot, agent_id), quality_i)

Adam lr=1e-3  |  20,000 steps
Output: checkpoints/estimators/estimators_final.pth
```

### Phase 3 – Filter -> UCB MoE -> CRP (Continual)

```
AdaSlot + Agents + Estimators FROZEN.  UCBWeightedMoE updated online.

Per slot:
  hybrid_score[50] = 0.5*VAE_score + 0.5*MLP_score
  filtered_K = top-10
  weights[K]  = UCBWeightedMoE.get_weights(filtered_K)
  out(slot)   = sum_i w_i * agent_i(slot)        ->  (256,)

Feature = concat(out[0..S-1])                     ->  2816-dim

CRP predict -> reward (1.0/0.0)
ucb_moe.update_batch(filtered_ids, weights, reward)
CRP learn  -> expert routing + GPM projection

Continual loop:
  for task_id in range(n_tasks):
    train  -> train_loaders[task_id]
    eval   -> cumulative test_loaders[0..task_id]

UCB state: checkpoints/ucb_moe_state.npz
```

---

## Feature Dimension Summary

| Mode | Formula | Dim |
|------|---------|-----|
| **Pipeline (UCB MoE)** | 11 x 256 | **2816** |
| Legacy first-k concat | 7 x 3 x 128 | 2688 |

Mode auto-selected: pipeline if estimators + ucb_moe passed to `train_phase3_crp`, otherwise legacy.

---

## Directory Structure

```
src/
  models/adaslot/
    model.py               AdaSlotModel
    train.py               4-phase training CLI
  slot_multi_agent/
    atomic_agent.py        ResidualMLPAgent + DINOLoss
    estimators.py          VAEEstimator, MLPEstimator
    bandit_selector.py     UCBWeightedMoE + legacy selectors
    aggregator.py          CRPExpertAggregator + ExpertModule
    system.py              Inference wrapper

checkpoints/
  adaslot/adaslot_final.pth                Phase 1 output
  agents/agents_final.pth                  Phase 2 output
  estimators/estimators_final.pth          Phase 2.5 output
  ucb_moe_state.npz                        Phase 3 UCB state
  slot_attention/adaslot_real/             Pre-trained AdaSlot
```

---

## CLI Reference (train.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase` | `all` | `1`, `2`, `2.5`, `3`, or `all` |
| `--device` | `cuda` | Device |
| `--adaslot_ckpt` | — | AdaSlot checkpoint path |
| `--agent_ckpt` | — | Agent pool checkpoint path |
| `--estimator_ckpt` | — | Estimators checkpoint (for Phase 3) |
| `--steps` | `500000` | Phase 1 steps |
| `--agent_steps` | `100000` | Phase 2 steps |
| `--p2b_steps` | `20000` | Phase 2.5 steps |
| `--p2b_lr` | `1e-3` | Phase 2.5 learning rate |
| `--filter_k` | `10` | Top-K after estimator filtering |
| `--ucb_exploration` | `1.414` | UCB exploration constant c |
| `--ucb_burn_in` | `100` | Burn-in rounds (uniform weights) |
| `--num_classes` | `20` | Classes per continual task |
| `--batch_size` | `64` | Batch size |

---

## Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| Estimator training | Not trained | Phase 2.5 quality-weighted |
| Agent selection | First k=3 (hardcoded) | VAE/MLP filter -> top-10 |
| Committee decision | Hard concat of k outputs | Weighted MoE sum |
| Bandit | Not connected | UCB Weighted MoE + reward |
| CRP input dim | 2688 (7x3x128) | 2816 (11x256) |
| Phase count | 3 | 4 |
| UCB persistence | None | ucb_moe_state.npz |

---

## Implementation Status

**Completed:**
- AdaSlot with adaptive Gumbel slot pruning
- 50-agent DINO SSL training
- Phase 2.5: VAE/MLP estimator training with quality signal
- UCBWeightedMoE with online reward feedback
- Phase 3 full pipeline: filter -> UCB MoE -> CRP
- CRP Expert Aggregator with GPM
- 4-phase main() with all new CLI args
- UCB state save/load
- Continual multi-task loop with cumulative evaluation

**To Do:**
- Ablation studies (pipeline vs legacy, filter_k sensitivity)
- UCB hyperparameter tuning
- Class-IL / Task-IL benchmark evaluation

---

## References

1. AdaSlot: https://github.com/amazon-science/adaSlot
2. DINOv2: Oquab et al. (2023)
3. UCB Bandit: Auer et al. (2002)
4. Weighted MoE / UCB combo: omoe-codebase/theta_wmv.py, combo_bandit.py
5. CRP: Aldous (1985); Pitman (2006)
6. GPM: Saha et al. (2021)
7. Expert Gate: Aljundi et al. (2017)
8. CL Scenarios: van de Ven & Tolias (2019)

---
*Last Updated: 2026-02-23*
