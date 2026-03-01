# Research Prompt: Performance Estimation for Slot-Agent Matching

## 🎯 Research Question

**"How to estimate the performance of a specialized agent on an object slot WITHOUT actually running the agent?"**

## 📋 Context (Very Specific)

### System Architecture

```
Image
  ↓
Slot Attention → Decompose into N slots (e.g., 7 object representations)
  ↓
Problem: Which agent should process which slot?
  
Given:
- M specialized agents (e.g., 10 agents)
- All agents have SAME architecture but DIFFERENT weights
- Each agent is specialized for certain object types/patterns
- N slots per image (e.g., 7 slots)

Challenge:
- Cannot run all M agents on all N slots (M × N = 70 forward passes!)
- Need to SELECT top-k agents per slot (k << M, e.g., k=3)
- Selection must be FAST (real-time) and CHEAP (resource-constrained)
```

### Key Constraint: Pre-Execution Estimation

**Critical requirement**: Estimate agent performance on slot using a **lightweight sub-network** (NOT running the actual agent).

```
Current Ideas:
1. VAE-based: Reconstruction error of slot → difficulty/uncertainty
2. MLP-based: Direct score prediction (learn mapping: slot → agent performance)

Need: What other methods exist in literature?
```

## 🔍 What to Research

### 1. **Performance Prediction Networks (Meta-Learning)**

**Keywords to search**:
- "Performance prediction networks"
- "Meta-learning for model selection"
- "Hyper-networks predict model performance"
- "Learned optimizer selection"
- "Neural architecture search performance prediction"

**Questions**:
- How do NAS methods predict architecture performance without full training?
- Can hyper-networks predict how well a network will perform?
- Meta-learning approaches for model selection?

### 2. **Amortized Inference for Computational Cost**

**Keywords**:
- "Amortized inference"
- "Fast inference without execution"
- "Learned inference cost estimation"
- "Early exit networks performance prediction"

**Questions**:
- How to learn direct mapping: input → expected performance?
- Can we amortize the cost of running multiple models?

### 3. **Uncertainty-Based Selection (Our Context: Object Slots)**

**Keywords**:
- "Uncertainty estimation object-centric"
- "VAE reconstruction for difficulty estimation"
- "Prototype matching uncertainty"
- "Out-of-distribution detection object representations"

**Questions**:
- How effective is VAE reconstruction error for estimating difficulty?
- Prototype-based matching: distance to prototypes → performance?
- Other uncertainty measures suitable for object slots?

### 4. **Multi-Armed Bandits & Online Learning**

**Keywords**:
- "Contextual bandits model selection"
- "Thompson sampling neural networks"
- "Online learning agent selection"
- "Explore-exploit for model routing"

**Questions**:
- Can we frame this as contextual bandits? (Context = slot, Arm = agent)
- How to balance exploration (try new agents) vs exploitation (use best known)?

### 5. **Mixture of Experts (MoE) Gating**

**Keywords**:
- "Mixture of experts gating network"
- "Sparse mixture of experts routing"
- "Switch transformer routing"
- "Learned routing for specialized models"

**Questions**:
- How do MoE gating networks decide which expert to use?
- Can gating mechanisms be adapted for our slot-agent matching?
- Top-k routing in MoE: how is it implemented efficiently?

### 6. **Object-Centric Learning & Slot Matching**

**Keywords**:
- "Object-centric slot assignment"
- "Slot attention routing mechanisms"
- "Compositional learning object specialization"
- "Object-centric continual learning"

**Questions**:
- Existing work on matching slots to specialized processors?
- How does compositional learning handle specialization?

### 7. **Continual Learning Task Selection**

**Keywords**:
- "Task selection continual learning"
- "Model selection incremental learning"
- "Agent selection multi-task learning"
- "Forward transfer prediction"

**Questions**:
- How to predict which model will perform best on new data?
- Forward transfer estimation methods?
- Catastrophic forgetting prediction?

## 🎓 Specific Papers to Look For

### Highly Relevant Areas

1. **Neural Architecture Search (NAS) - Performance Prediction**
   - How NAS predicts architecture performance without full training
   - Zero-shot NAS, one-shot NAS predictors
   
2. **Meta-Learning for Model Selection**
   - Learning to select models based on input characteristics
   - MAML, Prototypical Networks applied to model selection

3. **Conditional Computation & Early Exit**
   - BranchyNet, MSDNet (multi-scale dense networks)
   - How they decide when to exit early

4. **Mixture of Experts**
   - Switch Transformers, GShard (Google)
   - Expert selection and routing mechanisms

5. **Object-Centric Learning**
   - Slot Attention follow-up papers
   - Compositional learning with specialized components

## 📊 Comparison Criteria

When evaluating methods, consider:

| Criterion | Importance | Notes |
|-----------|------------|-------|
| **Speed** | ⭐⭐⭐⭐⭐ | Must be < 5% overhead of running agent |
| **Accuracy** | ⭐⭐⭐⭐ | Estimation should correlate with true performance |
| **Lightweight** | ⭐⭐⭐⭐⭐ | Sub-network must be small (< 1M params) |
| **Online Learning** | ⭐⭐⭐ | Can improve over time? |
| **Continual Learning** | ⭐⭐⭐⭐ | Works with new tasks/classes? |
| **Object-Centric** | ⭐⭐⭐ | Suitable for slot representations? |

## 🔬 Research Template

For each method found, document:

```markdown
### Method Name: [e.g., "Hyper-Critic Performance Prediction"]

**Source**: [Paper, Conference, Year]

**Core Idea**: [One sentence]

**How it works**:
- Input: ?
- Output: ?
- Training: ?

**Pros**:
- 

**Cons**:
- 

**Applicability to our problem** (1-5): ⭐⭐⭐⭐⭐

**Implementation complexity** (1-5): ⭐⭐⭐

**Notes**:
```

## 🎯 Our Specific Constraints (Important!)

### Input
- **Slot representation**: (slot_dim,) e.g., 64-dimensional vector
- Object-centric, from Slot Attention
- NOT raw image

### Output Needed
- **Performance score**: Scalar [0, 1] or similar
- Represents: "How well will Agent_i perform on this slot?"

### Constraints
- **Lightweight**: Sub-network must be small (e.g., < 100K parameters)
- **Fast**: Inference time < 1ms per slot-agent pair
- **No ground truth during inference**: Cannot run agent to get true performance
- **Training time**: Have access to historical data (slot, agent_id, true_performance)

### Current Baseline Ideas

**1. VAE Reconstruction Error**
```python
class VAEEstimator:
    def estimate(self, slot):
        recon_error = vae.reconstruct_error(slot)
        # High error → difficult → might need specialized agent
        return score_function(recon_error)
```

**2. Simple MLP**
```python
class MLPEstimator:
    def estimate(self, slot, agent_id):
        # Direct learned mapping
        concat = torch.cat([slot, agent_embedding[agent_id]])
        score = mlp(concat)  # → [0, 1]
        return score
```

**Question**: What's better than these baselines?

## 📚 Suggested Search Queries

### Google Scholar
```
1. "performance prediction neural networks meta-learning"
2. "amortized inference model selection"
3. "mixture of experts gating mechanism"
4. "early exit networks confidence estimation"
5. "object-centric compositional learning"
6. "slot attention routing mechanism"
7. "continual learning task selection"
8. "contextual bandits model selection"
9. "zero-shot neural architecture search"
10. "learned routing for specialized models"
```

### ArXiv
```
Recent papers (2020-2026) on:
- cs.LG + "performance prediction"
- cs.LG + "model selection meta-learning"
- cs.LG + "mixture of experts"
- cs.CV + "object-centric" + "routing"
```

### GitHub
```
Search for:
- "mixture of experts pytorch"
- "slot attention routing"
- "model selection meta learning"
- "performance prediction network"
```

## 🎯 Expected Output

After research, provide:

1. **Top 3-5 promising methods** with justifications
2. **Comparison table** of methods
3. **Recommended approach** for our specific problem
4. **Implementation plan** (if choosing a method)

## 💡 Bonus: Related Topics

- **Neural Program Synthesis**: Learning to select programs/models
- **AutoML**: Automated model selection
- **Active Learning**: Selecting which model to query
- **Ensemble Selection**: Choosing subset of models
- **Resource-Aware ML**: Computational budgeting

---

## 📝 Summary

**Goal**: Find the best method to estimate "Agent_i performance on Slot_j" using a lightweight sub-network, without running Agent_i.

**Context**: Object-centric continual learning with multiple specialized agents.

**Constraints**: Fast, lightweight, works with slots, supports continual learning.

**Current ideas**: VAE reconstruction error, simple MLP predictor.

**Need**: Literature review to find better/SOTA methods! 🚀

