# Training Guide: Slot Multi-Agent System

## Overview

The pipeline trains in **4 sequential phases**. The default mode (`--phase all`) follows the
**CompSLOT protocol**: for every continual task, all four phases run in order on that task's
data before moving to the next task. Weights carry forward across tasks with no
re-initialisation.

```
────────────────────────────────────────────────────────────────
 CompSLOT mode  (--phase all, default)
────────────────────────────────────────────────────────────────
 for each task t = 0 … T-1:
   Phase 1    fine-tune AdaSlot on task t data     (--task_p1_steps, default 2000)
   Phase 2    fine-tune Agents on task t data      (--task_p2_steps, default 2000)
   Phase 2.5  fine-tune Estimators on task t data  (--task_p2b_steps, default 1000)
   Phase 3    update CRP aggregator on task t data (online, 1 pass)
   → eval on all tasks 0 … t
   → save checkpoints/task{t}/
────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────
 Single-phase mode  (--phase 1 / 2 / 2.5 / 3)
────────────────────────────────────────────────────────────────
 Phase 1    AdaSlot pretraining over ALL tasks  (~500k steps)  → checkpoints/adaslot/adaslot_final.pth
 Phase 2    Agent DINO training over ALL tasks  (~100k steps)  → checkpoints/agents/agents_final.pth
 Phase 2.5  Estimator training over ALL tasks   (~20k steps)   → checkpoints/estimators/estimators_final.pth
 Phase 3    CRP aggregator — per-task loop                     → checkpoints/ucb_moe_state.npz
────────────────────────────────────────────────────────────────
```

All phases share the same entry point: `src/models/adaslot/train.py`.

---

## Quick Start

```bash
# ── CompSLOT: per-task sequential fine-tuning (recommended) ──────────
python src/models/adaslot/train.py \
  --phase all \
  --device cuda \
  --num_classes 10 \
  --task_p1_steps 2000 \
  --task_p2_steps 2000 \
  --task_p2b_steps 1000

# ── Single-phase runs (original behaviour) ───────────────────────────
python src/models/adaslot/train.py --phase 1
python src/models/adaslot/train.py --phase 2   --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth
python src/models/adaslot/train.py --phase 2.5 --agent_ckpt   checkpoints/agents/agents_final.pth
python src/models/adaslot/train.py --phase 3   --estimator_ckpt checkpoints/estimators/estimators_final.pth
```

---

## Phase 1: AdaSlot Pretraining / Fine-Tuning

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
| Learning rate | 4e-4 (`--p1_lr`) |
| Scheduler | linear warmup (10k steps) then exp decay |
| Grad clip | max_norm=1.0 |
| Steps — single-phase | 500,000 (`--p1_steps`) |
| Steps — per-task (CompSLOT) | 2,000 (`--task_p1_steps`) |

### Key Outputs

- Input: 3-channel image at 128×128
- Outputs: `slots (B, S, 64)` and `hard_keep_decision (B, S)` in {0,1}
- Slots with `hard_keep_decision == 0` are **dropped** in all downstream phases

### Checkpoints

```
Single-phase:   checkpoints/adaslot/adaslot_final.pth
CompSLOT task t: checkpoints/adaslot/task{t}/adaslot_final.pth
                 checkpoints/task{t}/adaslot.pth          (combined save)
```

---

## Phase 2: Agent DINO Training / Fine-Tuning

**What trains:** 50 ResidualMLPAgent student+teacher pairs via DINO self-supervised loss.  
**AdaSlot is FROZEN.**

### Objective

```
DINO loss between student and teacher on slot tokens.
Teacher EMA: θ_teacher = 0.996 * θ_teacher + 0.004 * θ_student
```

### Architecture

```
slot (B, 64)
  -> Linear(64->256) + LayerNorm
  -> 3× ResidualBlock(256)
  -> DINO head: Linear(256->256->128->256)
  -> softmax -> hidden label (B, 256)
```

### Optimiser

| Setting | Value |
|---------|-------|
| Optimiser | AdamW |
| Learning rate | 1e-3 (`--p2_lr`) |
| Weight decay | 0.04 |
| Scheduler | warmup (5k) then exp decay |
| EMA momentum | 0.996 |
| Steps — single-phase | 100,000 (`--p2_steps`) |
| Steps — per-task (CompSLOT) | 2,000 (`--task_p2_steps`) |
| Agents per step | 5 / 50 (random sample) |

### Checkpoints

```
Single-phase:    checkpoints/agents/agents_final.pth
CompSLOT task t: checkpoints/agents/task{t}/agents_final.pth
                 checkpoints/task{t}/agents.pth
```

---

## Phase 2.5: Estimator Training / Fine-Tuning

**What trains:** 50 VAEEstimators + 1 shared MLPEstimator.  
**AdaSlot + Agents are FROZEN.**

### Quality Signal

```
quality_i(slot) = 1 - H(agent_i(slot)) / log(num_prototypes)

peaked softmax -> low entropy -> high quality (~1.0)
uniform softmax -> max entropy -> low quality (~0.0)
```

### VAEEstimator (50 instances)

```
Encoder: Linear(64->64) -> ReLU -> Linear(64->32) -> ReLU -> fc_mu(16) + fc_logvar(16)
Decoder: Linear(16->32) -> ReLU -> Linear(32->64) -> ReLU -> Linear(64->64)

Loss: sum_i quality_i * MSE(recon_i, slot) + 0.5 * KL(z || N(0,1))
Score at inference: sigmoid(threshold - MSE(recon, slot))
```

### MLPEstimator (1 shared)

```
Input: concat[slot(64), Embedding(agent_id)(32)] = 96-dim
MLP:   Linear(96->128) -> LN -> ReLU -> Linear(128->64) -> LN -> ReLU -> Linear(64->1) -> Sigmoid
Loss:  MSE(predicted_quality, true_quality)
```

### Optimiser

| Setting | Value |
|---------|-------|
| Optimiser | Adam, lr=1e-3 (`--p2b_lr`) |
| Steps — single-phase | 20,000 (`--p2b_steps`) |
| Steps — per-task (CompSLOT) | 1,000 (`--task_p2b_steps`) |
| Agents per step | 10 / 50 |

### Checkpoints

```
Single-phase:    checkpoints/estimators/estimators_final.pth
CompSLOT task t: checkpoints/estimators/task{t}/estimators_final.pth
                 checkpoints/task{t}/estimators.pth
```

---

## Phase 3: Filter → UCB MoE → Cross-Attention CRP (Continual Learning)

**What trains/updates:** `CRPExpertAggregator` (online, per-task) + `UCBWeightedMoE` (online, per-sample).  
**AdaSlot + Agents + Estimators are FROZEN in this phase.**

### Feature Extraction Per Slot

```
slot (64)
  -> hybrid_score[50] = 0.5*VAE_score + 0.5*MLP_score
  -> top-K filtered_ids                               (--filter_k, default 10)
  -> weights[K] = UCBWeightedMoE.get_weights(filtered_ids)
  -> weighted_out = Σ w_i * agent_i(slot)   ->  (256,)

Feature = stack(weighted_out[0 .. S-1])   shape: (S, 256)   <- input to cross-attention
```

### UCB Weighted MoE

```
UCB(i) = μ_i + c * √(ln t / n_i)
  c = exploration_constant (--ucb_exploration, default √2 = 1.414)

Burn-in (t < burn_in):   uniform weights
Post burn-in:            weights = softmax(UCB_scores)
```

### CRP Expert Aggregator — MoE Cross-Attention

Each expert holds **learnable query embeddings** (not prototype centroids).
Routing uses attention entropy as an out-of-distribution signal.

```
LearnableExpert:
  queries  ∈ ℝ^{n_queries × d}          # expert identity (learnable parameter)
  key_proj : Linear(agent_dim → d)
  val_proj : Linear(agent_dim → d)

  forward(H: S×D):
    K = key_proj(H)                      # S×d
    V = val_proj(H)                      # S×d
    A = softmax(queries @ Kᵀ / √d)       # n_queries × S
    z = mean(A @ V)                      # (d,)
    entropy = -Σ A·logA                  # OOD signal

CRPExpertAggregator:
  class_queries ∈ ℝ^{num_classes × d}   # per-class learnable codes

  Routing (inference):
    all_z, scores, entropy = forward_all_experts(H)
    repr = softmax(scores) · all_z       # MoE-gated aggregation
    logits = repr @ class_queriesᵀ

  CRP trigger (training):
    if mean(entropy) > threshold AND Bernoulli(α / (N + α)):
        create new LearnableExpert()

  Loss:
    CE(logits[seen_classes], y)  — backprop through cross-attention
    old class_queries frozen at task boundary (freeze_old_classes)
```

### Task Boundary

At the start of each task (CompSLOT mode) the aggregator's old class query vectors are frozen
so past knowledge is preserved while new classes can still be learned:

```python
aggregator._agg.freeze_old_classes(set(new_task_class_ids))
```

### Reward Feedback to UCB

```python
reward = 1.0 if pred == true_label else 0.0
ucb_moe.update_batch(filtered_ids, weights, reward)
# agent i receives: local_reward = reward * w_i
```

### Continual Task Loop

```python
# CompSLOT mode — simplified pseudo-code
aggregator = None
ucb_moe    = UCBWeightedMoE(...)

for task_id, task_loader in enumerate(train_loaders):
    # ── per-task fine-tuning ──────────────────────────────
    train_phase1_adaslot(adaslot, task_loader, steps=task_p1_steps)
    train_phase2_agents(adaslot, agents, task_loader, steps=task_p2_steps)
    train_phase2b_estimators(adaslot, agents, task_loader, steps=task_p2b_steps)

    # ── freeze old classes, then learn new ones ───────────
    if aggregator:
        aggregator._agg.freeze_old_classes(class_order[task_id])

    aggregator = train_phase3_crp(
        adaslot_model=adaslot,
        student_agents=agents,
        dataloader=task_loader,
        aggregator=aggregator,   # None on first task → creates fresh
        ...
    )

    # ── cumulative evaluation ─────────────────────────────
    for eval_task in range(task_id + 1):
        acc = evaluate(aggregator, test_loaders[eval_task])
```

### Task Scheduling (CIFAR-100 default)

| `--num_classes` | Tasks (n_tasks) | Classes per task |
|-----------------|-----------------|------------------|
| 10 | 10 | 10 |
| 20 | 5 | 20 |
| 50 | 2 | 50 |
| 100 | 1 | 100 |

---

## Full Pipeline Commands

### CompSLOT (per-task sequential — recommended)

```bash
python src/models/adaslot/train.py \
  --phase all \
  --device cuda \
  --num_classes 10 \
  --task_p1_steps 2000 \
  --task_p2_steps 2000 \
  --task_p2b_steps 1000 \
  --p1_lr 4e-4 \
  --p2_lr 1e-3 \
  --p2b_lr 1e-3 \
  --filter_k 10 \
  --ucb_exploration 1.414 \
  --ucb_burn_in 100 \
  --batch_size 8
```

### Original (all tasks combined, then phase 3 loop)

```bash
python src/models/adaslot/train.py --phase 1 --p1_steps 500000 --device cuda
python src/models/adaslot/train.py --phase 2 --p2_steps 100000 \
    --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth
python src/models/adaslot/train.py --phase 2.5 --p2b_steps 20000 \
    --agent_ckpt checkpoints/agents/agents_final.pth
python src/models/adaslot/train.py --phase 3 \
    --estimator_ckpt checkpoints/estimators/estimators_final.pth \
    --filter_k 10 --num_classes 10
```

---

## Complete Parameter Reference

| Parameter | Default | Mode | Description |
|-----------|---------|------|-------------|
| `--phase` | `all` | — | `all`=CompSLOT, `1/2/2.5/3`=single-phase |
| `--device` | auto | — | `cuda` / `mps` / `cpu` |
| `--batch_size` | `8` | all | Batch size |
| `--num_slots` | `11` | all | Max slot count |
| `--slot_dim` | `64` | all | Slot embedding dimension |
| `--num_agents` | `50` | all | Agent pool size |
| `--num_prototypes` | `256` | all | DINO output dimension |
| `--num_classes` | `10` | 3/all | Classes per continual task |
| `--resolution` | `128` | all | Input image resolution |
| `--pretrained` | `CLEVR10` | all | Pretrained AdaSlot weights |
| `--adaslot_ckpt` | — | 2/2.5/3 | Path to AdaSlot checkpoint |
| `--agent_ckpt` | — | 2.5/3 | Path to agent checkpoint |
| `--estimator_ckpt` | — | 3 | Path to estimator checkpoint |
| **Single-phase steps** | | | |
| `--p1_steps` | `500000` | `--phase 1` | AdaSlot training steps |
| `--p2_steps` | `100000` | `--phase 2` | Agent DINO training steps |
| `--p2b_steps` | `20000` | `--phase 2.5` | Estimator training steps |
| `--p1_lr` | `4e-4` | 1 | AdaSlot learning rate |
| `--p2_lr` | `1e-3` | 2 | Agent learning rate |
| `--p2b_lr` | `1e-3` | 2.5 | Estimator learning rate |
| **Per-task steps (CompSLOT)** | | | |
| `--task_p1_steps` | `2000` | `--phase all` | Phase 1 fine-tune steps per task |
| `--task_p2_steps` | `2000` | `--phase all` | Phase 2 fine-tune steps per task |
| `--task_p2b_steps` | `1000` | `--phase all` | Phase 2.5 fine-tune steps per task |
| **Phase 3 / Aggregator** | | | |
| `--filter_k` | `10` | 3/all | Top-K agents after fast filtering |
| `--ucb_exploration` | `1.414` | 3/all | UCB exploration constant c |
| `--ucb_burn_in` | `100` | 3/all | Burn-in rounds (uniform weights) |

---

## Architecture Flow

```
IMAGE
  │
  ▼
[Phase 1] AdaSlot — fine-tuned on each task (CompSLOT) or pretrained once
  │ slots (S, 64) — inactive slots pruned by Gumbel gate
  ▼
[Phase 2] Agent Pool — fine-tuned on each task (CompSLOT) or trained once
  │ hidden labels  (50 agents × 256)
  ▼
[Phase 2.5] Estimators — fine-tuned on each task (CompSLOT) or trained once
  │ quality scores (50,)
  ▼
[Phase 3] UCB Weighted MoE — online bandit, persists across tasks
  │ weighted feature (S, 256)
  ▼
[Phase 3] CRP Expert Aggregator — learnable cross-attention experts
  │   LearnableExpert: queries ∈ ℝ^{n_q × d}  ×  cross-attention over (S, 256)
  │   class_queries  ∈ ℝ^{C × d}   (old classes frozen at task boundary)
  │   routing: attention entropy → CRP new-expert trigger
  ▼
PREDICTION  (logits = repr @ class_queriesᵀ)
```

---

## Estimated Training Times (GPU)

| Mode | Phase | Steps/task | Approx. time |
|------|-------|------------|-------------|
| Single-phase | Phase 1 | 500k total | 4–6 h |
| Single-phase | Phase 2 | 100k total | 1–2 h |
| Single-phase | Phase 2.5 | 20k total | 20–30 min |
| Single-phase | Phase 3 | 1 epoch/task | 10–30 min |
| **Single-phase total** | | | **~7–10 h** |
| CompSLOT | Phase 1 ft | 2k/task | ~2 min/task |
| CompSLOT | Phase 2 ft | 2k/task | ~2 min/task |
| CompSLOT | Phase 2.5 ft | 1k/task | ~1 min/task |
| CompSLOT | Phase 3 | 1 epoch/task | ~5 min/task |
| **CompSLOT total (10 tasks)** | | | **~1–1.5 h** |

Times assume NVIDIA V100/A100, batch_size=8, CIFAR-100 at 128×128.

---

## Checkpoint Loading Examples

```python
import torch
from src.models.adaslot.train import (
    train_phase1_adaslot,
    train_phase2_agents,
    train_phase2b_estimators,
    train_phase3_crp,
)

# Resume CompSLOT from task checkpoint
task_id = 3
ckpt_dir = f"checkpoints/task{task_id}"

adaslot.load_state_dict(
    torch.load(f"{ckpt_dir}/adaslot.pth")["model"])
student_agents.load_state_dict(
    torch.load(f"{ckpt_dir}/agents.pth")["student_agents"])

est = torch.load(f"{ckpt_dir}/estimators.pth")
vae_estimators.load_state_dict(est["vae_estimators"])
mlp_estimator.load_state_dict(est["mlp_estimator"])

# Load UCB state
from src.slot_multi_agent.bandit_selector import UCBWeightedMoE
ucb_moe = UCBWeightedMoE(num_agents=50, exploration_constant=1.414, burn_in=100)
ucb_moe.load("checkpoints/ucb_moe_state.npz")
```

