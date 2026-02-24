# %% [markdown]
# # Continual Learning Pipeline — CompSLOT Protocol
#
# ## Quy trình training
#
# **CompSLOT mode** (`--phase all`, mặc định):
# Với **mỗi task**, pipeline chạy tuần tự **4 giai đoạn** trên dữ liệu của task đó,
# sau đó mới sang task tiếp theo. Trọng số giữ nguyên qua các task (không reset).
#
# ```
# for task t = 0 … T-1:
#     Phase 1   → fine-tune AdaSlot       (--task_p1_steps  bước)
#     Phase 2   → fine-tune Agents        (--task_p2_steps  bước)
#     Phase 2.5 → fine-tune Estimators    (--task_p2b_steps bước)
#     Phase 3   → cập nhật CRP Aggregator (online, 1 pass)
#     → đánh giá tích lũy task 0 … t
#     → lưu checkpoints/task{t}/
# ```
#
# **Single-phase mode** (`--phase 1 / 2 / 2.5 / 3`):
# Chạy từng phase riêng lẻ trên toàn bộ dữ liệu (hành vi gốc).

# %% [markdown]
# ## 1. Cài đặt môi trường

# %%
# Clone repo (bỏ qua nếu đã có)
!git clone -b fq https://github.com/PhamPhuHoa-23/Continual-Learning.git
%cd Continual-Learning

# Cài dependencies
!pip install -q einops wandb river pytest gdown

# Tải pretrained AdaSlot checkpoint (CLEVR10)
!mkdir -p checkpoints/slot_attention/adaslot_real/AdaSlotCkpt
!gdown "1lcBTpdEFKotrMQjE_xuH23AfgzxXHVd1" \
    -O "checkpoints/slot_attention/adaslot_real/AdaSlotCkpt/CLEVR10.ckpt"

# %% [markdown]
# ## 2. CompSLOT — training tuần tự theo task *(khuyến nghị)*
#
# Với mỗi task trên CIFAR-100 (mặc định 10 tasks × 10 classes):
# - Phase 1: fine-tune AdaSlot 2 000 bước
# - Phase 2: fine-tune Agents 2 000 bước
# - Phase 2.5: fine-tune Estimators 1 000 bước
# - Phase 3: cập nhật CRP aggregator (cross-attention, entropy-based routing)
#
# Điều chỉnh `--task_p*_steps` để tăng/giảm mức độ fine-tune mỗi task.

# %%
!python src/models/adaslot/train.py \
    --phase all \
    --device cuda \
    --num_classes 100 \
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

# %% [markdown]
# ## 3. (Tuỳ chọn) Tiếp tục từ checkpoint task đã lưu
#
# Nếu quá trình bị gián đoạn, load checkpoint của task cuối cùng đã hoàn thành
# rồi chạy lại. Ví dụ tiếp tục từ task 4:

# %%
RESUME_TASK = 4   # task cuối đã hoàn thành

!python src/models/adaslot/train.py \
    --phase all \
    --device cuda \
    --num_classes 100 \
    --task_p1_steps 2000 \
    --task_p2_steps 2000 \
    --task_p2b_steps 1000 \
    --adaslot_ckpt   checkpoints/task{RESUME_TASK}/adaslot.pth \
    --agent_ckpt     checkpoints/task{RESUME_TASK}/agents.pth \
    --estimator_ckpt checkpoints/task{RESUME_TASK}/estimators.pth

# %% [markdown]
# ## 4. (Tuỳ chọn) Single-phase — chạy từng giai đoạn riêng
#
# Dùng khi muốn pretrain toàn bộ dữ liệu trước, sau đó mới chạy CRP.

# %% [markdown]
# ### Phase 1: Pretrain AdaSlot

# %%
!python src/models/adaslot/train.py \
    --phase 1 \
    --device cuda \
    --p1_steps 500000 \
    --p1_lr 4e-4 \
    --batch_size 64

# %% [markdown]
# ### Phase 2: Train Agents (DINO SSL)

# %%
!python src/models/adaslot/train.py \
    --phase 2 \
    --device cuda \
    --p2_steps 100000 \
    --p2_lr 1e-3 \
    --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth

# %% [markdown]
# ### Phase 2.5: Train Estimators (VAE + MLP)

# %%
!python src/models/adaslot/train.py \
    --phase 2.5 \
    --device cuda \
    --p2b_steps 20000 \
    --p2b_lr 1e-3 \
    --agent_ckpt checkpoints/agents/agents_final.pth

# %% [markdown]
# ### Phase 3: CRP Aggregator (continual loop only)

# %%
!python src/models/adaslot/train.py \
    --phase 3 \
    --device cuda \
    --num_classes 100 \
    --filter_k 10 \
    --ucb_exploration 1.414 \
    --ucb_burn_in 100 \
    --estimator_ckpt checkpoints/estimators/estimators_final.pth
