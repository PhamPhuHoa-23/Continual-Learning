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
!gdown "1lcBTpdEFKotrMQjE_xuH23AfgzxXHVd1" -O "checkpoints/slot_attention/adaslot_real/AdaSlotCkpt/CLEVR10_Custom.ckpt"

# %% [markdown]
# ## 2. Phase 1: Train AdaSlot (Pretraining)
# Train the AdaSlot extraction model. We'll specify paths to save checkpoints.
# Note: Since this phase takes a long time, we might use a predefined pretrained model instead.
# To do so automatically, `train.py` handles the `--pretrained` argument to download models if absent.

# %%
!python -m src.models.adaslot.train --phase 1 --p1_steps 1000 --save_dir checkpoints/adaslot

# %% [markdown]
# ## 3. Phase 2: Train Atomic Agents (DINO SSL)
# Train the pool of generic experts on the extracted slots in an offline manner.
# We utilize the checkpoints trained from Phase 1. 
# Here we use `--pretrained CLEVR10` just for demonstration so it will auto-download pre-trained weights if phase 1 was skipped.

# %%
!python -m src.models.adaslot.train --phase 2 --p1_steps 2 --pretrained CLEVR10_Custom --p2_steps 2000

# %% [markdown]
# ## 4. Phase 3: Incremental Aggregator Training
# Train the CRP-based Aggregator on the outputs of the Atomic Agents.
# This runs the CIFAR-100 continual benchmark by default.

# %%
!python -m src.models.adaslot.train --phase 3 --p1_steps 2 --pretrained CLEVR10_Custom --num_classes 100
