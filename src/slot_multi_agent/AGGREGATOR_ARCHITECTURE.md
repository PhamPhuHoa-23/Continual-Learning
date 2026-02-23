# CRP-based Expert Assignment Aggregator

## Tổng quan

Aggregator nhận **output từ các Agents** (hidden labels) và đưa ra **dự đoán class cuối cùng**. Thay vì dùng một cây quyết định (Hoeffding Tree) duy nhất, kiến trúc mới sử dụng hệ thống **Expert động** dựa trên:

1. **Chinese Restaurant Process (CRP)** — tự tạo Expert mới khi cần
2. **Gradient Alignment Scoring** — đo mức tương thích giữa dữ liệu mới và Expert
3. **Gradient Projection (GPM-lite)** — chống representation drift khi cập nhật Expert

## Pipeline

```
Agent Outputs (hidden labels)
        │
        ▼
┌──────────────────────────────────┐
│   CRPExpertAggregator            │
│                                  │
│  1. Similarity: cosine(x, proto) │
│     → route tới Expert quen      │
│                                  │
│  2. Alignment: cos(g_new, g_old) │
│     → gradient tương thích?      │
│                                  │
│  3. Capacity: exp(-β·C_k/C_ideal)│
│     → Expert quá tải? → phạt!    │
│                                  │
│  Score = Sim × Align × Capacity  │
│                                  │
│  4. CRP Decision:                │
│     • Score cao → gán Expert cũ  │
│     • Score thấp → tạo mới       │
│       (P = α/(N+α))             │
│                                  │
│  5. Train Expert (GPM-lite)      │
└──────────────────────────────────┘
        │
        ▼
  Predicted Class Label
```

## Công thức Score (Balanced — không rich-get-richer)

$$Score(k) = \underbrace{cos(x, proto_k)}_{Similarity} \times \underbrace{cos(g_{new}, g_{old}^k)}_{Alignment} \times \underbrace{e^{-\beta \cdot C_k / C_{ideal}}}_{Capacity}$$

| Thành phần | Ý nghĩa | Tại sao cần? |
|------------|---------|-------------|
| **Similarity** `cos(x, prototype_k)` | Input giống prototype Expert → ưu tiên route tới | Đảm bảo Expert xử lý đúng loại dữ liệu nó quen |
| **Alignment** `cos(g_new, g_old^k)` | Gradient mới cùng hướng gradient cũ → tương thích | Tránh catastrophic forgetting |
| **Capacity** `exp(-β × C_k / C_ideal)` | Expert đã ôm nhiều class → bị phạt nặng | **Chống ôm đồm class** |

> [!IMPORTANT]
> **Tại sao không dùng Popularity (CRP gốc)?**
> CRP gốc: `P(k) = N_k / (N + α)` → Expert đông → nhận thêm → đông hơn (rich-get-richer).
> Trong continual learning, điều này khiến **1 Expert ôm hết tất cả class**, các Expert khác bị bỏ rơi.
> Thay vào đó, Capacity penalty **phạt Expert overload** theo số class đa dạng, lấy cảm hứng từ
> Switch Transformer load balancing và Expert Gate.

## 3 Class chính

### 1. `ExpertModule(nn.Module)`

Một Expert nhỏ gọn, gồm:

```
Input (feature_dim) → Linear → LayerNorm → GELU → Linear → Logits (num_classes)
```

Mỗi Expert lưu trữ:
- **`prototype`** — Running mean (EMA) của các input đã được gán, dùng cho cosine similarity matching
- **`gradient_memory`** — Trung bình gradient tích lũy (EMA), dùng cho Alignment scoring
- **`projection_bases`** — Ma trận SVD cơ sở cho Gradient Projection (chống drift)
- **`class_counts`** — Mapping class → count, phục vụ Capacity penalty (đếm số class đa dạng)

### 2. `CRPExpertAggregator(nn.Module)`

Quản lý danh sách Expert động (`nn.ModuleList`). Các method chính:

| Method | Chức năng |
|--------|-----------|
| `assign_expert(x, label)` | CRP + Score → chọn Expert hoặc tạo mới |
| `learn_one(x, label)` | Online learning: assign → train → update prototype & gradient memory |
| `predict_one(x)` | Tìm Expert gần nhất (cosine) → forward → argmax |
| `predict_proba_one(x)` | Tương tự, trả softmax distribution |
| `update_all_projection_bases()` | Cập nhật SVD bases (gọi tại task boundary) |

### 3. `BatchCRPAggregator`

Wrapper batch cho `CRPExpertAggregator`, cung cấp `learn_batch` / `predict_batch` / `predict_proba_batch`.

## Gradient Projection (GPM-lite)

Khi cập nhật Expert, gradient được **project lên không gian vuông góc** với các subspace quan trọng trước đó:

```
grad_projected = grad - bases @ bases^T @ grad
```

- `bases` được tính bằng **SVD** trên buffer activations
- Giữ tối đa `projection_rank` singular vectors
- Gọi `update_all_projection_bases()` tại mỗi task boundary

## Smart Expert Initialization

Khi tạo Expert mới, weights được **copy từ Expert gần nhất + noise nhỏ**:

```python
new_expert.classifier = copy(nearest_expert.classifier) + N(0, 0.01)
```

Giúp Expert mới không phải học từ đầu, mà kế thừa knowledge có sẵn.

## Hyperparameters

| Param | Default | Mô tả |
|-------|---------|-------|
| `alpha` | 1.0 | CRP concentration — cao hơn = dễ tạo Expert mới hơn |
| `max_experts` | 30 | Giới hạn số Expert tối đa |
| `score_threshold` | 0.05 | Score dưới ngưỡng này → xem xét tạo Expert mới |
| `expert_lr` | 1e-3 | Learning rate cho classifier MLP của Expert |
| `capacity_beta` | 1.5 | Sức mạnh penalty capacity — cao hơn = phạt nặng hơn khi Expert ôm nhiều class |
| `ideal_classes_per_expert` | 5 | Số class lý tưởng mỗi Expert (CIFAR-100 / 20 experts ≈ 5) |
| `projection_rank` | 10 | Số SVD bases giữ lại cho GPM |
| `prototype_momentum` | 0.95 | EMA momentum cho prototype update |
| `gradient_momentum` | 0.9 | EMA momentum cho gradient memory |
| `buffer_size` | 100 | Số activations lưu cho SVD |

## Sử dụng

```python
from src.slot_multi_agent import create_aggregator

# Tạo CRP aggregator
agg = create_aggregator('crp', feature_dim=2688, num_classes=100)

# Online learning
info = agg.learn_one(hidden_labels, label=5)
# info = {'expert_idx': 2, 'is_new_expert': False, 'num_experts': 7}

# Prediction
pred = agg.predict_one(hidden_labels)  # → 5
proba = agg.predict_proba_one(hidden_labels)  # → {5: 0.87, 3: 0.08, ...}

# Tại task boundary
agg.update_all_projection_bases()
```
