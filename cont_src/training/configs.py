"""
Training Configurations
=======================
One dataclass per training phase.  All fields have sensible defaults so a
caller only needs to override what differs from the baseline.

Adding a new phase?
  1. Add a @dataclass here (subclass PhaseConfig or just standalone).
  2. Add it to PipelineConfig.phases if it should run by default.
  3. Write the matching Trainer in its own file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """Fields shared by every training phase."""

    # Optimisation
    lr: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0          # 0 → disabled

    # Duration  (steps take priority over epochs when both are set)
    max_steps: int = 0              # 0 → use max_epochs
    max_epochs: int = 1

    # Checkpointing
    save_every_n_steps: int = 500   # 0 → only at end
    checkpoint_dir: str = "checkpoints/continual"

    # Logging
    log_every_n_steps: int = 50

    # Device  ("" → auto-detect cuda/cpu)
    device: str = ""


# ---------------------------------------------------------------------------
# Phase 0  –  AdaSlot pre-training
# ---------------------------------------------------------------------------

@dataclass
class AdaSlotTrainerConfig(PhaseConfig):
    """
    Train (or fine-tune) the SlotAttention backbone with:
        L_total = w_recon · L_recon  +  w_sparse · L_sparse  +  w_prim · L_prim

    Reconstruction loss uses AdaSlot's original mse_sum convention:
        L_recon = F.mse_loss(pred, tgt, reduction="sum") / B

    Both recon and sparse weights here are applied on top of that raw value.
    The recommended defaults keep all three terms in a similar magnitude range.
    """

    # --- Loss weights ---
    w_recon: float = 2.03e-5        # 1/(C·H·W) normalises mse_sum to per-pixel
    w_sparse: float = 10.0          # linear sparse coefficient (AdaSlot default)
    w_prim: float = 5.0             # primitive loss weight

    # --- Primitive loss ---
    prim_temperature: float = 10.0

    # --- Slot attention freeze strategy ---
    # "none"   → train everything
    # "frozen" → freeze slot-attention, only train downstream modules
    freeze_slot_attn: str = "none"

    # --- Optimiser ---
    lr: float = 4e-4
    max_steps: int = 200
    max_epochs: int = 0             # ignored when max_steps > 0

    # --- Input ---
    image_size: int = 128           # used for mse_sum normalisation check


# ---------------------------------------------------------------------------
# Phase Init  –  Clustering initialisation
# ---------------------------------------------------------------------------

@dataclass
class ClusterInitConfig:
    """
    Extract slots from the (frozen) backbone, run a chosen clustering
    algorithm (via CLUSTERING_REGISTRY), and spawn one SlotVAE + one Agent
    per cluster.

    ``method`` selects the algorithm from CLUSTERING_REGISTRY.
    Registered names (built-in):
        "kmeans"           – KMeansClustering   (needs n_clusters)
        "minibatch_kmeans" – MiniBatchKMeans     (needs n_clusters)
        "dbscan"           – DBSCANClustering    (needs eps, min_samples)
        "hdbscan"          – HDBSCANClustering   (needs min_cluster_size, min_samples)
        "agglomerative"    – AgglomerativeClustering
        "bayesian_gmm"     – GaussianMixtureClustering (soft, prunes components)

    ``method_kwargs`` are passed directly to the algorithm's constructor,
    so any algorithm-specific parameter can be set without changing this class.
    """

    # ------- Algorithm selection -------
    method: str = "hdbscan"          # any key registered in CLUSTERING_REGISTRY

    # Common / shared params
    n_clusters: int = 8              # ignored by density-based methods
    random_state: int = 42

    # Method-specific kwargs passed straight to the algorithm constructor.
    # Example for HDBSCAN:  {"min_cluster_size": 20, "min_samples": 5}
    # Example for DBSCAN:   {"eps": 0.8, "min_samples": 5}
    # Example for KMeans:   {"n_init": 10, "max_iter": 300}
    method_kwargs: dict = None       # None → use algorithm defaults

    def __post_init__(self):
        if self.method_kwargs is None:
            self.method_kwargs = {}

    # ------- Data collection -------
    # 0 → use entire dataloader
    max_batches_for_clustering: int = 200

    # ------- VAE training per cluster -------
    vae_latent_dim: int = 16
    vae_epochs: int = 20
    vae_beta: float = 1.0
    vae_lr: float = 1e-3

    # Scoring mode for spawned SlotVAEs
    # One of: "generative" | "mahal_z" | "mahal_slot"
    scoring_mode: str = "generative"

    device: str = ""


# Backward-compat alias — old code that says KMeansInitConfig still works
KMeansInitConfig = ClusterInitConfig


# ---------------------------------------------------------------------------
# Phase A  –  Agent warm-up (L_agent only)
# ---------------------------------------------------------------------------

@dataclass
class PhaseAConfig(PhaseConfig):
    """
    Warm-up agents with only the agent reconstruction loss.
    SlotVAEs are frozen; routing is hard (argmax).

    L = γ · L_agent
    """

    # Loss weight
    gamma: float = 1.0              # multiplier for L_agent

    # Duration
    lr: float = 3e-4
    max_steps: int = 50
    max_epochs: int = 0

    # Hard vs soft routing during Phase A
    routing_mode: str = "hard"      # "hard" | "soft"
    routing_temperature: float = 1.0


# ---------------------------------------------------------------------------
# Phase B  –  Full agent training with label losses + soft routing
# ---------------------------------------------------------------------------

@dataclass
class PhaseBConfig(PhaseConfig):
    """
    Full agent training:
        L = γ·L_agent  +  α·L_prim  +  β·L_SupCon

    Routing uses temperature-annealed softmax so gradients flow during
    training; hard argmax is used at inference.
    """

    # Loss weights
    gamma: float = 1.0              # L_agent
    alpha: float = 0.3              # L_prim  (primitive / KL)
    beta: float = 0.3               # L_SupCon (supervised contrastive)

    # Primitive loss
    prim_temperature: float = 10.0

    # SupCon loss
    supcon_temperature: float = 0.07

    # Soft routing temperature schedule
    init_temperature: float = 2.0   # start temperature
    final_temperature: float = 0.1  # end temperature
    temp_anneal: str = "cosine"     # "cosine" | "linear" | "constant"

    # Duration
    lr: float = 2e-4
    max_steps: int = 200
    max_epochs: int = 0

    # Aggregator mode: "attention" (→ SLDA) or "concat" (→ classifier)
    aggregator_mode: str = "attention"

    # Freeze VAE routers during Phase B?
    freeze_routers: bool = True


# ---------------------------------------------------------------------------
# Phase C  –  SLDA incremental fitting
# ---------------------------------------------------------------------------

@dataclass
class SLDAConfig:
    """
    StreamLDA incremental fitting.  No gradient steps — this is a one-pass
    closed-form update over the frozen feature extractor output.
    """

    feature_dim: int = 64           # dimension of the aggregated H vector
    n_classes: int = 100            # total classes across all tasks

    # Covariance regularisation (ridge)
    shrinkage: float = 1e-4

    # 0 → full dataset pass; N → use at most N batches
    max_batches: int = 0

    device: str = ""


# ---------------------------------------------------------------------------
# Pipeline orchestration config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Top-level config that wires all phase configs together.

    phases: ordered list of phase names that will run for each new task.
    Available names (built-in):
        "adaslot"   → AdaSlotTrainer   (Task 1 only by default)
        "kmeans"    → KMeansInitialiser (Task 1 only by default)
        "phase_a"   → AgentPhaseATrainer
        "phase_b"   → AgentPhaseBTrainer
        "slda"      → SLDATrainer

    Custom trainers can be registered via ContinualPipeline.register_phase().
    """

    # Which phases run for Task 1?
    task1_phases: List[str] = field(
        default_factory=lambda: ["adaslot", "kmeans", "phase_a", "phase_b", "slda"]
    )
    # Which phases run for Tasks 2+?
    taskN_phases: List[str] = field(
        default_factory=lambda: ["phase_a", "phase_b", "slda"]
    )

    # Sub-configs  (populated with defaults; override as needed)
    adaslot: AdaSlotTrainerConfig = field(default_factory=AdaSlotTrainerConfig)
    kmeans:  ClusterInitConfig    = field(default_factory=ClusterInitConfig)
    phase_a: PhaseAConfig         = field(default_factory=PhaseAConfig)
    phase_b: PhaseBConfig         = field(default_factory=PhaseBConfig)
    slda:    SLDAConfig           = field(default_factory=SLDAConfig)

    # Global checkpoint root (individual phases write into sub-folders)
    checkpoint_root: str = "checkpoints/continual"

    # Seed
    seed: int = 42
