# Đề xuất các phương pháp chấm điểm agent thay thế VAE + MLP

## Vấn đề với cơ chế hiện tại

Hiện tại pipeline dùng hai tín hiệu:

| Estimator | Tín hiệu | Vấn đề |
|-----------|----------|--------|
| VAE | Reconstruction error thấp = agent quen slot này | Reconstruction ≠ classification. Agent quen slot theo VAE chưa chắc phân loại đúng. |
| MLP | Học regression `(slot, agent_id) → entropy` | Entropy thấp ≠ accuracy cao. Agent tự tin sai vẫn được chọn (confident-but-wrong). |

**Vấn đề gốc**: cả hai đều dùng **entropy của chính agent** làm ground truth cho quality — đây là tín hiệu circular, không gắn với performance thực tế.

---

## Phương pháp 1 — Distilled Tester (Shadow Agent)

### Ý tưởng

Mỗi agent đầy đủ (`3 ResidualBlocks + DINO head`) đi kèm một **shadow** nhỏ hơn (1 block, hidden_dim/2). Shadow được distill từ full agent → khi chạy shadow ta thu được xấp xỉ output của full agent với chi phí ~4–8× thấp hơn.

Tại scoring step: **chỉ chạy shadow**, nhận output xấp xỉ, dùng entropy của shadow làm proxy cho confidence của full agent. So với VAE, signal này trung thực hơn vì shadow *thực sự là agent*, chỉ nhỏ hơn.

### Kiến trúc

```
Full Agent (frozen sau Phase 2):
  slot (64) → InputProj(256) → Block×3 → DINOHead → logits (256)

Shadow Agent (mới, trainable):
  slot (64) → InputProj(128) → Block×1 → ShadowHead(128→256) → logits (256)
```

### Training (Phase 2.5 thay thế)

```python
# Loss = KL divergence giữa shadow output và full agent output
teacher_probs = softmax(full_agent(slot) / τ_teacher)
student_log   = log_softmax(shadow(slot) / τ_student)
distill_loss  = -( teacher_probs * student_log ).sum(dim=-1).mean()
```

Không cần label thật. Shadow học "copy logic" của full agent — nếu full agent thấy slot lạ và output uniform, shadow cũng sẽ output uniform (entropy cao) và ngược lại.

### Scoring tại inference

```python
with torch.no_grad():
    shadow_logits = shadow_i(slot)
    entropy = -(softmax(shadow_logits) * log_softmax(shadow_logits)).sum(-1)
    score_i = 1.0 - entropy / log(num_prototypes)
```

### Ưu / nhược điểm

| | |
|---|---|
| ✅ | Signal trực tiếp từ logic agent (không phải VAE reconstruction) |
| ✅ | Không cần Phase 2.5 riêng — shadow được distill ngay trong Phase 2 hoặc sau |
| ✅ | Chi phí nhỏ: shadow ~1/4 tham số full agent |
| ✅ | Tự cập nhật: khi fine-tune full agent qua task mới, re-distill shadow |
| ⚠️ | Vẫn dùng entropy (confident-but-wrong vẫn xảy ra) — nhưng đúng hơn VAE |
| ⚠️ | Cần thêm 50 shadow models vào memory |

### Sơ đồ

```
slot
 ├─→ shadow_1 → entropy_1 → score_1 ─┐
 ├─→ shadow_2 → entropy_2 → score_2  │  topK filter
 │   ...                              ├─────────────→ top-K agent ids
 └─→ shadow_50 → entropy_50 → score_50 ┘
         (50 shadow forward passes, không grad)
```

---

## Phương pháp 2 — Reasoning Budget & Self-Consistency Scoring

### Ý tưởng

Thay vì dùng một forward pass để đánh giá, cho phép agent **"suy nghĩ"** bằng cách rollout K lần với stochastic sampling (MC Dropout hoặc temperature > 1). Điểm của agent = **consistency across K samples**: nếu K lần chạy đều ra cùng prototype → agent chắc chắn → điểm cao; nếu K lần ra khác nhau → agent bối rối → điểm thấp.

Đây là ý tưởng của **Self-Consistency** (Wang et al., 2022) áp dụng vào slot-agent matching thay cho LLMs.

### Cơ chế

```python
# Reasoning budget = K rollouts
K = budget   # ví dụ K = 5 (cheap) hoặc K = 20 (accurate)

# Bật dropout tại inference
agent.train()   # dropout active
with torch.no_grad():
    outputs = [softmax(agent(slot)) for _ in range(K)]   # K × (proto_dim,)

# Đo consistency = 1 - variance trung bình
outputs_stacked = torch.stack(outputs)          # (K, proto_dim)
mean_output     = outputs_stacked.mean(dim=0)   # (proto_dim,)
variance        = ((outputs_stacked - mean_output)**2).mean()
score           = 1.0 - variance.clamp(0, 1)
```

Hoặc dùng **Jensen-Shannon divergence** giữa các sample để đo mức độ bất đồng:

$$\text{score}_i(\text{slot}) = 1 - \text{JSD}\left(\{p_k\}_{k=1}^K\right), \quad p_k = \text{agent}_i(\text{slot}, \text{dropout}=\text{on})$$

### Dynamic Budget Allocation

Chi phí được kiểm soát: không cần rollout đủ K cho tất cả 50 agent. Dùng **Early Stopping**:

```
K_max = 20, threshold_confidence = 0.05

for agent_i in all_agents:
    for k in range(1, K_max):
        run one rollout
        if variance_so_far < threshold_confidence:
            break   # đã đủ tự tin, không rollout thêm
    score_i = 1 - variance
```

Agent chắc chắn (variance nhỏ ngay từ đầu) → dừng sớm. Agent bối rối → tiêu thêm budget → cuối cùng loại bỏ.

### Ưu / nhược điểm

| | |
|---|---|
| ✅ | Không cần module estimator riêng — dùng luôn agent gốc |
| ✅ | Không cần Phase 2.5 training |
| ✅ | Dễ tune: tăng K → chính xác hơn, giảm K → nhanh hơn |
| ✅ | Self-consistency là tín hiệu tốt hơn entropy đơn lẻ |
| ⚠️ | Vẫn cần chạy 50 agents (dù mỗi agent chỉ K lần thay vì 1 lần) |
| ⚠️ | K=5 → tổng 250 forward passes/slot thay vì 50 |
| ⚠️ | Cần MC Dropout: model phải có dropout, teacher không dùng được |

### Kết hợp với Shadow (Hybrid)

```
shadow_i(slot) × K rollouts → consistency score_i
```

Shadow nhỏ (1 block) + K=5 rollouts ≈ chi phí của 5/4 = 1.25 full agent forward pass cho cả 50 agents.

---

## Phương pháp 3 — Contrastive Compatibility (Supervision Đúng)

### Ý tưởng

Giải quyết vấn đề gốc: dạy estimator bằng **accuracy thực** thay vì entropy.

Dùng một **memory buffer nhỏ** (reservoir sampling, ~200 samples/task) lưu lại `(slot, correct_label)`. Sau mỗi task, với mỗi agent chạy trên buffer → đo accuracy thực → dùng làm label cho estimator.

```python
# Buffer: [(slot_1, label_1), ..., (slot_N, label_N)]
for agent_i in agents:
    preds_i       = agent_i.predict(buffer_slots)
    true_accuracy = (preds_i == buffer_labels).float().mean()
    # Dùng true_accuracy làm target thay vì entropy
```

Estimator giờ học mapping `(slot_prototype, agent_id) → true_accuracy`, không phải entropy.

### Kiến trúc estimator

Thay vì MLP regression thông thường, dùng **Contrastive Learning**:

- Positive pair: `(slot_type, agent_i)` nếu agent_i có accuracy cao trên slot type đó
- Negative pair: `(slot_type, agent_j)` nếu agent_j có accuracy thấp

```python
# Contrastive loss (InfoNCE)
slot_embed  = slot_encoder(slot)           # (d,)
agent_embed = agent_embeddings[agent_id]   # (d,)

pos_score = dot(slot_embed, good_agent_embed)
neg_score = dot(slot_embed, bad_agent_embed)
loss = -log(exp(pos_score) / (exp(pos_score) + exp(neg_score)))
```

### Ưu / nhược điểm

| | |
|---|---|
| ✅ | Ground truth thực sự là accuracy, không phải proxy |
| ✅ | Buffer nhỏ (200 samples) → memory overhead thấp |
| ✅ | Tương thích với continual learning: buffer cập nhật qua task |
| ⚠️ | Cần buffer — vi phạm strict no-replay setting |
| ⚠️ | Phức tạp hơn khi implement |

---

## Phương pháp 4 — Online Slot Clustering + Beta-Bandit Per Cluster

### Ý tưởng

Thay vì score từng cặp `(slot, agent)` riêng lẻ, **cluster slots** thành C loại online (GMM hoặc K-means streaming), rồi maintain một **Beta distribution** `Beta(α, β)` cho mỗi cặp `(cluster, agent)`:

- $\alpha$ = số lần agent chọn đúng trên slot thuộc cluster này
- $\beta$ = số lần agent chọn sai

Scoring bằng **Thompson Sampling**: sample $p \sim \text{Beta}(\alpha, \beta)$ → score của agent trên slot thuộc cluster này = p.

```
slot → Online GMM → cluster_id c
Thompson sample: score_{i,c} ~ Beta(α_{i,c}, β_{i,c})
Top-K filter by score
After prediction: update α hoặc β dựa trên đúng/sai
```

### Ưu / nhược điểm

| | |
|---|---|
| ✅ | Không cần training riêng — fully online |
| ✅ | Exploration-exploitation tốt hơn UCB (uncertainty tự nhiên qua Beta) |
| ✅ | Slot cluster = prior knowledge về "loại object slot" |
| ⚠️ | C cluster cần tune (C quá nhỏ → không discriminative, quá lớn → sparse data) |
| ⚠️ | Online GMM khó ổn định khi feature space drift qua task |

---

## Tổng hợp & Khuyến nghị

### So sánh tổng quan

| Phương pháp | Training cần? | Chi phí inference | Quality signal | Continual-friendly |
|-------------|--------------|-------------------|----------------|-------------------|
| VAE + MLP *(hiện tại)* | Phase 2.5 riêng | 50 VAE + 50 MLP | Entropy (proxy, circular) | ✅ |
| **Shadow Distillation** | Trong Phase 2 | 50 shadow (nhanh) | Entropy (nhưng accurate hơn) | ✅ |
| **Reasoning Budget** | Không | K×50 rollouts | Self-consistency | ✅ |
| **Contrastive + Buffer** | Phase 2.5 mới | 1 MLP | Accuracy thực | ⚠️ (cần buffer) |
| **Beta-Bandit / Cluster** | Không | 1 GMM lookup | Reward online | ✅ |

### Đề xuất lộ trình triển khai

**Ngắn hạn (ít thay đổi nhất):**
Thay `VAE + MLP` bằng **Shadow Distillation**:
1. Thêm `ShadowAgent` (1 block, hidden=128) vào `atomic_agent.py`
2. Distill shadow từ full agent cuối Phase 2, ngay trước Phase 3
3. Bỏ hoàn toàn Phase 2.5 (VAE + MLP)
4. Scoring = entropy của shadow forward pass

**Trung hạn (cải thiện signal):**
Kết hợp **Shadow + Reasoning Budget**:
- Shadow × K=5 rollouts (MC Dropout) → consistency score
- Chi phí: 50 shadow × 5 = 250 forward passes/slot, mỗi shadow ~1/4 full agent → ~62.5× full agent equivalent
- So với hiện tại: 50 VAE + 50 MLP forward passes mỗi slot

**Dài hạn (accuracy-grounded):**
Thêm **Contrastive Buffer** nhỏ (100 examples/task, không vi phạm tinh thần CL):
- Dùng accuracy thực trên buffer để calibrate shadow score
- `final_score = shadow_score × calibration_factor_{cluster}`

### Sơ đồ kiến trúc đề xuất (Shadow + Reasoning Budget)

```
                        ┌────────────────────────────────┐
slot (64-dim)           │  PER-AGENT SHADOW SCORING       │
    │                   │                                 │
    ├──→ shadow_1 ──×K──┤ consistency_1 = 1 - JSD(K runs) │
    ├──→ shadow_2 ──×K──┤ consistency_2                   │
    │    ...            │  ...                            │
    └──→ shadow_50 ─×K──┤ consistency_50                  │
                        └──────────────┬─────────────────┘
                                        ↓
                               top-K filter (K=10)
                                        ↓
                               UCB Weighted MoE
                          (dùng reward thực từ Phase 3)
                                        ↓
                               full_agent_i(slot)  ×K selected
                                        ↓
                               weighted slot output
```

---

## Tham khảo liên quan

- **Self-Consistency**: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS 2014
- **MC Dropout as Bayesian Approximation**: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
- **Thompson Sampling**: Russo et al., "A Tutorial on Thompson Sampling", FnTML 2018
- **InfoNCE Contrastive**: Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018
