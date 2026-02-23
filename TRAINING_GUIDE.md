# Training Guide: Slot Multi-Agent System

## Overview

The pipeline trains in **4 sequential phases**. Each phase freezes earlier components and builds on them.

```
Phase 1    AdaSlot pretraining (unsupervised)       ~500k steps   -> checkpoints/adaslot/adaslot_final.pth
Phase 2    Agent DINO training (self-supervised)    ~100k steps   -> checkpoints/agents/agents_final.pth
Phase 2.5  Estimator training (VAE + MLP)           ~20k steps    -> checkpoints/estimators/estimators_final.pth
Phase 3    Filter -> UCB MoE -> CRP (continual)     1 pass/task   -> checkpoints/ucb_moe_state.npz
```

All phases use the same entry point: `src/models/adaslot/train.py`.

---

## Quick Start

```bash
# Full pipeline (all 4 phases):
python src/models/adaslot/train.py --phase all --device cuda --num_classes 20

# Individual phases:
python src/models/adaslot/train.py --phase 1
python src/models/adaslot/train.py --phase 2   --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth
python src/models/adaslot/train.py --phase 2.5 --agent_ckpt checkpoints/agents/agents_final.pth
python src/models/adaslot/train.py --phase 3   --estimator_ckpt checkpoints/estimators/estimators_final.pth
```

---

## Phase 1: AdaSlot Pretraining

**What trains:** AdaSlot (CNN encoder + SlotAttention + Gumbel pruning + decoder).
**Supervision:** Reconstruction loss (fully unsupervised, no labels).

### Objective

```
Loss = MSE(reconstruction, original_image) / batch_size
     + sparsity_weight * mean(hard_keep_decision)

sparsity_weight = 10.0   (encourages pruning inactive slots)
```

### Optimiser

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Learning rate | 4e-4 |
| Scheduler | linear warmup (10k steps) then exp decay |
| Grad clip | max_norm=1.0 |
| Default steps | 500,000 |
| Batch size | 64 |

### Key Outputs

- Input: 3-channel image at 128x128
- Outputs: `slots (B, S, 64)` and `hard_keep_decision (B, num_slots)` in {0,1}
- Slots with `hard_keep_decision == 0` are **dropped** in all downstream phases

### Checkpoint

```
checkpoints/adaslot/adaslot_final.pth
  Keys: model_state_dict, optimizer_state_dict, scheduler_state_dict, loss, step
```

### CLI

```bash
python src/models/adaslot/train.py \
  --phase 1 \
  --steps 500000 \
  --batch_size 64 \
  --lr 4e-4
```

---

## Phase 2: Agent DINO Training

**What trains:** 50 ResidualMLPAgent student+teacher pairs via DINO self-supervised loss.
**AdaSlot is FROZEN.**

### Objective

```
DINO loss between student and teacher outputs on slot tokens.
Teacher updated via EMA (not gradient): theta_teacher = 0.996 * theta_teacher + 0.004 * theta_student
```

### Architecture

```
slot (B, 64)
  -> Linear(64->256) + LayerNorm
  -> 3x ResidualBlock(256)
  -> DINO head: Linear(256->256->128->256)
  -> softmax -> hidden label (B, 256)
```

### Efficiency Trick

Each step randomly samples **5 out of 50 agents** to update. After 100k steps all agents are
trained substantially even with partial sampling.

### Optimiser

| Setting | Value |
|---------|-------|
| Optimiser | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.04 |
| Scheduler | warmup (5k) then exp decay (rate 0.5 per 50k) |
| EMA momentum | 0.996 |
| student_temp | 0.1 |
| teacher_temp | 0.07 |
| center_momentum | 0.9 |
| Default steps | 100,000 |

### Checkpoint

```
checkpoints/agents/agents_final.pth
  Keys: student_agents_state_dict, teacher_agents_state_dict, optimizer_state_dict, step
```

### CLI

```bash
python src/models/adaslot/train.py \
  --phase 2 \
  --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth \
  --agent_steps 100000
```

---

## Phase 2.5: Estimator Training (NEW)

**What trains:** 50 VAEEstimators (one per agent) + 1 shared MLPEstimator.
**AdaSlot + Agents are FROZEN.**

### Why Estimators?

In Phase 3, for each slot we need to select the most relevant agents from a pool of 50.
Running all 50 agents per slot is expensive. The estimators learn a **fast proxy score**
that approximates how well each agent will perform on a given slot, using only a
forward pass through a tiny VAE/MLP — much cheaper than running the agent.

### Quality Signal

The supervision target for training is the **normalised confidence** of each agent's output:

```
quality_i(slot) = 1 - H(agent_i(slot)) / log(num_prototypes)

H(p) = -sum_k p_k * log(p_k + 1e-8)   (entropy)

Interpretation:
  peaked softmax -> low entropy -> high quality (~1.0)
  uniform softmax -> max entropy -> low quality (~0.0)
```

### VAEEstimator (50 instances)

```
Architecture:
  Encoder: Linear(64->64) -> ReLU -> Linear(64->32) -> ReLU -> fc_mu(16) + fc_logvar(16)
  Decoder: Linear(16->32) -> ReLU -> Linear(32->64) -> ReLU -> Linear(64->64)

Loss (quality-weighted):
  L_vae = sum_i quality_i * MSE(recon_i, slot)  +  beta * KL(z || N(0,1))
  beta = 0.5

Score at inference:
  score = sigmoid(threshold - MSE(recon, slot))   in [0,1]
  Low reconstruction error -> agent understands this slot -> high score
```

### MLPEstimator (1 shared instance)

```
Input: concat[slot(64), Embedding(agent_id)(32)] = 96-dim
MLP:
  Linear(96->128) -> LayerNorm -> ReLU -> Dropout(0.1)
  Linear(128->64) -> LayerNorm -> ReLU -> Dropout(0.1)
  Linear(64->1)   -> Sigmoid

Loss: MSE(predicted_quality, true_quality)
```

### Training Loop

```python
# For each step:
sampled_agents = random.sample(range(50), k=10)   # 10/50 per step

for agent_id in sampled_agents:
    with torch.no_grad():
        slot_output = agents[agent_id](active_slots)
    quality = 1 - entropy(slot_output) / log(256)

    # VAE
    recon, mu, logvar = vae_estimators[agent_id](active_slots)
    L_vae = (quality * MSE(recon, active_slots)).mean() + 0.5 * KL

    # MLP
    pred_quality = mlp_estimator(active_slots, agent_id)
    L_mlp = MSE(pred_quality, quality)
```

### Optimiser

| Setting | Value |
|---------|-------|
| Optimiser (VAE) | Adam, lr=1e-3 |
| Optimiser (MLP) | Adam, lr=1e-3 |
| Default steps | 20,000 |
| Agents per step | 10 / 50 |

### Checkpoint

```
checkpoints/estimators/estimators_final.pth
  Keys: vae_estimators (list of state_dicts), mlp_estimator (state_dict), step
```

### CLI

```bash
python src/models/adaslot/train.py \
  --phase 2.5 \
  --agent_ckpt checkpoints/agents/agents_final.pth \
  --p2b_steps 20000 \
  --p2b_lr 1e-3
```

---

## Phase 3: Filter -> UCB MoE -> CRP (Continual Learning)

**What trains/updates:** CRPExpertAggregator (online per-task) + UCBWeightedMoE (online per-sample).
**AdaSlot + Agents + Estimators are FROZEN.**

### Full Pipeline Per Slot

```
slot (64)
  -> hybrid_score[50] = 0.5*VAE_score + 0.5*MLP_score
  -> top-10 filtered_ids          (top-K fast filter)
  -> weights[10] = UCBWeightedMoE.get_weights(filtered_ids)
  -> weighted_out = sum_i w_i * agent_i(slot)    ->  (256,)
```

### Feature Assembly

```python
# For each image:
outputs = []
for s in range(num_active_slots):
    scores = _estimate_agent_scores(slots[s], vae_estimators, mlp_estimator, num_agents=50)
    _, topk_ids = torch.topk(scores, k=filter_k)           # filter_k=10

    agent_ids, weights = ucb_moe.get_weights(topk_ids.tolist())
    weighted_out = sum(w * agents[i](slot_s) for i, w in zip(agent_ids, weights))
    outputs.append(weighted_out)

feature = torch.cat(outputs)   # (S * 256,) = 2816-dim
```

### UCB Weighted MoE

```
UCB(i) = mu_i + c * sqrt(ln t / n_i)

  mu_i = empirical mean reward   (initialised 0)
  n_i  = rounds agent i participated in
  t    = total rounds
  c    = exploration_constant = sqrt(2) = 1.414

Burn-in (t < burn_in=100):
  weights = [1/K for each filtered agent]   (pure exploration)

Post burn-in:
  weights = softmax(UCB_scores / temperature)   (sum to 1)
```

### CRP Routing

```
Score(k) = Similarity(feature, prototype_k)
         * Alignment(grad_new, grad_memory_k)
         * Capacity(k)

Capacity(k) = exp(-beta * n_classes_k / ideal_classes_per_expert)

If best_score < score_threshold (0.05):
    With prob alpha / (N + alpha): create new expert
    Else: assign to best existing expert
```

### Reward Feedback

```python
# After each prediction:
reward = 1.0 if (pred_label == true_label) else 0.0
ucb_moe.update_batch(filtered_ids, weights, reward)
# Agent i receives: local_reward = reward * w_i
# This credits agents proportional to their committee weight
```

### Continual Multi-Task Loop

```python
aggregator = CRPExpertAggregator(...)   # single object across all tasks
ucb_moe    = UCBWeightedMoE(...)        # single object across all tasks

for task_id in range(n_tasks):
    # Train on new task
    for batch in train_loaders[task_id]:
        feature = _extract_weighted_features(batch, ...)
        pred, reward = aggregator.predict_and_learn(feature, label)
        ucb_moe.update_batch(...)

    # Evaluate cumulatively
    for eval_task in range(task_id + 1):
        acc = evaluate(aggregator, test_loaders[eval_task], ...)
        print(f"Task {eval_task} acc: {acc:.4f}")

# Save UCB state
ucb_moe.save("checkpoints/ucb_moe_state.npz")
```

### Task Scheduling (CIFAR-100)

| `--num_classes` | Tasks (n_tasks) | Classes per task |
|-----------------|-----------------|------------------|
| 10 | 10 | 10 |
| 20 | 5 | 20 |
| 50 | 2 | 50 |
| 100 | 1 | 100 |

### Why `test_loaders[0..task_id]` is cumulative?

Each `test_loaders[i]` contains all classes from tasks 0 through i. This tests:
- **Plasticity**: accuracy on the current task
- **Stability**: accuracy on old tasks (catastrophic forgetting check)

### CLI

```bash
python src/models/adaslot/train.py \
  --phase 3 \
  --estimator_ckpt checkpoints/estimators/estimators_final.pth \
  --filter_k 10 \
  --ucb_exploration 1.414 \
  --ucb_burn_in 100 \
  --num_classes 20
```

---

## Full Pipeline Command

```bash
python src/models/adaslot/train.py \
  --phase all \
  --device cuda \
  --num_classes 20 \
  --steps 500000 \
  --agent_steps 100000 \
  --p2b_steps 20000 \
  --p2b_lr 1e-3 \
  --filter_k 10 \
  --ucb_exploration 1.414 \
  --ucb_burn_in 100 \
  --batch_size 64
```

---

## Complete Parameter Reference

| Parameter | Default | Phase | Description |
|-----------|---------|-------|-------------|
| `--phase` | `all` | — | Which phase(s) to run |
| `--device` | `cuda` | — | Compute device |
| `--batch_size` | `64` | 1/2/2.5/3 | Batch size |
| `--steps` | `500000` | 1 | AdaSlot training steps |
| `--lr` | `4e-4` | 1 | AdaSlot learning rate |
| `--num_slots` | `11` | 1 | Max slot count |
| `--slot_dim` | `64` | 1 | Slot embedding dimension |
| `--adaslot_ckpt` | — | 2/2.5/3 | AdaSlot checkpoint to load |
| `--num_agents` | `50` | 2/2.5/3 | Agent pool size |
| `--num_prototypes` | `256` | 2/2.5/3 | DINO output dimension |
| `--agent_steps` | `100000` | 2 | Agent DINO training steps |
| `--agent_ckpt` | — | 2.5/3 | Agent pool checkpoint to load |
| `--p2b_steps` | `20000` | 2.5 | Estimator training steps |
| `--p2b_lr` | `1e-3` | 2.5 | Estimator learning rate |
| `--estimator_ckpt` | — | 3 | Estimator checkpoint to load |
| `--filter_k` | `10` | 3 | Top-K agents after fast filtering |
| `--ucb_exploration` | `1.414` | 3 | UCB exploration constant c |
| `--ucb_burn_in` | `100` | 3 | Burn-in rounds (uniform weights) |
| `--num_classes` | `20` | 3 | Classes per continual task |
| `--alpha` | `1.0` | 3 | CRP concentration parameter |
| `--max_experts` | `30` | 3 | Max CRP experts |

---

## Architecture Flow

```
IMAGE
  |
  v
[Phase 1] AdaSlot (unsupervised, frozen after)
  | slots (S, 64)
  v
[Phase 2] Agent Pool (DINO SSL, frozen after)
  | hidden labels (50 agents x 256)
  v
[Phase 2.5] Estimators (quality-signal supervised, frozen after)
  | quality scores (50,)
  v
[Phase 3] UCB Weighted MoE (online bandit update)
  | weighted feature (S x 256 = 2816)
  v
[Phase 3] CRP Expert Aggregator (online continual learning)
  |
  v
PREDICTION
```

---

## Estimated Training Times (GPU)

| Phase | Steps | Approx. Time |
|-------|-------|-------------|
| Phase 1 | 500k | 4-6 hours |
| Phase 2 | 100k | 1-2 hours |
| Phase 2.5 | 20k | 20-30 minutes |
| Phase 3 | 1 epoch/task | 10-30 minutes total |
| **Total** | — | **~7-10 hours** |

Times assume NVIDIA V100/A100, batch_size=64, CIFAR-100 at 128x128.

---

## Checkpoint Loading Examples

```python
from src.models.adaslot.train import (
    train_phase1_adaslot,
    train_phase2_agents,
    train_phase2b_estimators,
    train_phase3_crp
)
from src.slot_multi_agent import UCBWeightedMoE

# Load trained estimators for Phase 3
ckpt = torch.load("checkpoints/estimators/estimators_final.pth")
for i, vae in enumerate(vae_estimators):
    vae.load_state_dict(ckpt["vae_estimators"][i])
mlp_estimator.load_state_dict(ckpt["mlp_estimator"])

# Create UCB MoE
ucb_moe = UCBWeightedMoE(num_agents=50, exploration_constant=1.414, burn_in=100)
# Or load saved state:
ucb_moe.load("checkpoints/ucb_moe_state.npz")
```
