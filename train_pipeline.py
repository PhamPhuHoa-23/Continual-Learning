# %% [markdown]
# # Continual Learning Pipeline (AdaSlot & Agents) 
# This script sets up the environment by checking out the `fq` branch of the 
# Continual-Learning repository, installs required dependencies, and runs all 3 training phases.
# 
# *Note for Kaggle/Colab: You can copy and paste this code into an empty Notebook or run it directly.*

# %% [markdown]
# ## 1. Setup & Installation
# Clone the repository, switch to the `fq` branch, and install dependencies.

# %%
!git clone -b fq https://github.com/PhamPhuHoa-23/Continual-Learning.git
%cd Continual-Learning

# Install pip packages (assuming torch/torchvision are already in the environment if using Colab)
!pip install -q einops wandb river pytest gdown

# Download pre-trained AdaSlot checkpoint from Google Drive
!mkdir -p checkpoints/slot_attention/adaslot_real/AdaSlotCkpt
!gdown "1lcBTpdEFKotrMQjE_xuH23AfgzxXHVd1" -O "checkpoints/slot_attention/adaslot_real/AdaSlotCkpt/CLEVR10.ckpt"

# %% [markdown]
# ## 2. Phase 1: Train AdaSlot (Pretraining)
# Train the AdaSlot extraction model. We'll specify paths to save checkpoints.
# Note: Since this phase takes a long time, we might use a predefined pretrained model instead.
# To do so automatically, `train.py` handles the `--pretrained` argument to download models if absent.

# %%
!python src/models/adaslot/train.py --phase 1 --device cuda
# %% [markdown]
# ## 3. Phase 2: Train Atomic Agents (DINO SSL)
# Train the pool of generic experts on the extracted slots in an offline manner.
# We utilize the checkpoints trained from Phase 1. 
# Here we use `--pretrained CLEVR10` just for demonstration so it will auto-download pre-trained weights if phase 1 was skipped.
# %%
!python src/models/adaslot/train.py --phase 2 --adaslot_ckpt checkpoints/adaslot/adaslot_final.pth

# %%
!python src/models/adaslot/train.py --phase 2.5 \
  --agent_ckpt checkpoints/agents/agents_final.pth

# %% [markdown]
# ## 4. Phase 3: Incremental Aggregator Training
# Train the CRP-based Aggregator on the outputs of the Atomic Agents.
# This runs the CIFAR-100 continual benchmark by default.

# %%
!python src/models/adaslot/train.py --phase 3 \
  --estimator_ckpt checkpoints/estimators/estimators_final.pth
