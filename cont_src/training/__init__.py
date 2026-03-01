"""
Training module
================
All phase trainers, the Cluster initialiser (multi-algorithm), SLDA,
and the pipeline orchestrator.  Everything you need to drive a continual
learning run:

    from cont_src.training import ContinualPipeline, PipelineConfig
    from cont_src.training import AdaSlotTrainer, AdaSlotTrainerConfig
    from cont_src.training import (
        AgentPhaseATrainer, PhaseAConfig,
        AgentPhaseBTrainer, PhaseBConfig,
        SLDATrainer, SLDAConfig, StreamLDA,
    )
    # Cluster init — pick any method from CLUSTERING_REGISTRY
    from cont_src.training import ClusterInitialiser, ClusterInitConfig, extract_slots
    from cont_src.training import BaseTrainer, MetricAccumulator
"""

# ---------- Configs ----------
from cont_src.training.configs import (
    PhaseConfig,
    AdaSlotTrainerConfig,
    ClusterInitConfig,
    ClusterInitConfig as KMeansInitConfig,   # backward compat
    PhaseAConfig,
    PhaseBConfig,
    SLDAConfig,
    PipelineConfig,
)

# ---------- Base ----------
from cont_src.training.base_trainer import BaseTrainer, MetricAccumulator

# ---------- Phase 0: AdaSlot ----------
from cont_src.training.adaslot_trainer import AdaSlotTrainer, mse_sum

# ---------- Init: clustering (all methods via CLUSTERING_REGISTRY) ----------
from cont_src.training.cluster_init import (
    ClusterInitialiser,
    ClusterInitialiser as KMeansInitialiser,   # backward compat
    extract_slots,
)

# ---------- Phase A: agent warm-up ----------
from cont_src.training.agent_phase_a import AgentPhaseATrainer

# ---------- Phase B: full agent training ----------
from cont_src.training.agent_phase_b import AgentPhaseBTrainer

# ---------- Phase C: SLDA ----------
from cont_src.training.slda_trainer import SLDATrainer, StreamLDA

# ---------- Pipeline ----------
from cont_src.training.pipeline import ContinualPipeline

__all__ = [
    # configs
    "PhaseConfig",
    "AdaSlotTrainerConfig",
    "ClusterInitConfig",
    "KMeansInitConfig",          # backward compat alias
    "PhaseAConfig",
    "PhaseBConfig",
    "SLDAConfig",
    "PipelineConfig",
    # base
    "BaseTrainer",
    "MetricAccumulator",
    # trainers
    "AdaSlotTrainer",
    "mse_sum",
    "ClusterInitialiser",
    "KMeansInitialiser",         # backward compat alias
    "extract_slots",
    "AgentPhaseATrainer",
    "AgentPhaseBTrainer",
    "SLDATrainer",
    "StreamLDA",
    # pipeline
    "ContinualPipeline",
]
