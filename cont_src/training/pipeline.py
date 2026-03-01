"""
Continual Pipeline Orchestrator
================================
Wires every phase trainer together and manages shared state across tasks.

The pipeline stores the full set of live components (backbone, VAEs, agents,
aggregator, SLDA) and drives them through the configured phase sequence for
each new task.

Extension points
----------------
1. **Add a phase**: write a Trainer class (or plain callable), then call
   ``pipeline.register_phase("my_phase", builder_fn)`` where
   ``builder_fn(pipeline, dataloader) → None`` runs your phase.
2. **Swap a phase**: re-register under the same name to override.
3. **Task-specific phases**: modify ``pipeline_config.task1_phases`` /
   ``pipeline_config.taskN_phases`` before calling ``run_task()``.

Usage
-----
    from cont_src.training.configs   import PipelineConfig
    from cont_src.training.pipeline  import ContinualPipeline

    cfg      = PipelineConfig()
    pipeline = ContinualPipeline(cfg, slot_model, aggregator)

    # Task 1
    pipeline.run_task(task_id=1, train_loader=loader1, test_loader=test1)

    # Task 2
    pipeline.run_task(task_id=2, train_loader=loader2, test_loader=test2)

    # Evaluate all tasks
    pipeline.evaluate_all(test_loaders={1: test1, 2: test2})
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cont_src.training.configs import PipelineConfig
from cont_src.training.adaslot_trainer import AdaSlotTrainer
from cont_src.training.cluster_init import ClusterInitialiser, extract_slots
from cont_src.training.agent_phase_a import AgentPhaseATrainer
from cont_src.training.agent_phase_b import AgentPhaseBTrainer
from cont_src.training.slda_trainer import SLDATrainer, StreamLDA

logger = logging.getLogger(__name__)

# Type alias for a phase builder function
PhaseBuilder = Callable[["ContinualPipeline", DataLoader], None]


class ContinualPipeline:
    """
    Continual learning pipeline orchestrator.

    Manages the shared state (backbone, VAEs, agents, aggregator, SLDA) and
    provides ``run_task()`` to drive all phases for one task in sequence.

    Parameters
    ----------
    config : PipelineConfig
    slot_model : nn.Module
        Pre-loaded AdaSlot backbone.  Phase 0 may fine-tune it; subsequent
        tasks leave it frozen.
    aggregator : nn.Module
        Aggregates (B, K, D_h) → (B, D_h) for SLDA and label losses.
    primitive_predictor : nn.Module, optional
        Used by AdaSlotTrainer for L_prim.  Can be None if backbone
        already provides "primitives" in its output dict.
    """

    def __init__(
        self,
        config: PipelineConfig,
        slot_model: nn.Module,
        aggregator: nn.Module,
        primitive_predictor: Optional[nn.Module] = None,
    ):
        self.config = config
        self.slot_model = slot_model
        self.aggregator = aggregator
        self.primitive_predictor = primitive_predictor

        # Shared mutable state updated across tasks
        self.vaes:   List[Any] = []    # List[SlotVAE]
        self.agents: List[nn.Module] = []    # List[ResidualMLPAgent]
        self.slda:   StreamLDA = StreamLDA(
            n_classes=config.slda.n_classes,
            feature_dim=config.slda.feature_dim,
            shrinkage=config.slda.shrinkage,
        )

        self._task_history: List[Dict] = []

        # Phase registry: name → builder(pipeline, loader)
        self._phase_builders: Dict[str, PhaseBuilder] = {}
        self._register_builtin_phases()

        logger.info("[ContinualPipeline] Initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Run all configured phases for ``task_id``.

        Parameters
        ----------
        task_id : int
            1-indexed.  Determines which phase sequence is used
            (task1_phases vs taskN_phases).
        train_loader : DataLoader
        test_loader  : DataLoader, optional
            If provided, accuracy is evaluated after SLDA fitting.

        Returns
        -------
        dict with per-phase metrics and optional evaluation accuracy.
        """
        cfg = self.config
        phases = cfg.task1_phases if task_id == 1 else cfg.taskN_phases

        logger.info(
            f"\n{'='*60}\n"
            f"  Task {task_id}  —  phases: {phases}\n"
            f"{'='*60}"
        )

        task_metrics: Dict[str, Any] = {"task_id": task_id}

        for phase_name in phases:
            builder = self._phase_builders.get(phase_name)
            if builder is None:
                raise KeyError(
                    f"Phase '{phase_name}' is not registered.  "
                    f"Available: {list(self._phase_builders.keys())}"
                )
            logger.info(f"\n--- Phase: {phase_name} (task {task_id}) ---")
            result = builder(self, train_loader)
            if result is not None:
                task_metrics[phase_name] = result

        # Optional evaluation after SLDA
        if test_loader is not None:
            metrics = self._run_evaluation(test_loader)
            task_metrics["eval"] = metrics

        self._task_history.append(task_metrics)
        self._save_task_checkpoint(task_id)

        return task_metrics

    def evaluate_all(
        self, test_loaders: Dict[int, DataLoader]
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate SLDA on multiple task test sets."""
        results = {}
        for tid, loader in test_loaders.items():
            m = self._run_evaluation(loader)
            results[tid] = m
            logger.info(f"[ContinualPipeline] Task {tid} eval: {m}")
        return results

    def register_phase(self, name: str, builder: PhaseBuilder) -> None:
        """
        Register (or override) a phase by name.

        Parameters
        ----------
        name : str
            Phase identifier, referenced in ``PipelineConfig.task1_phases``.
        builder : Callable[[ContinualPipeline, DataLoader], Any]
            Function that executes the phase and optionally returns metrics.
        """
        self._phase_builders[name] = builder
        logger.info(f"[ContinualPipeline] Phase '{name}' registered.")

    # ------------------------------------------------------------------
    # Built-in phase builders
    # ------------------------------------------------------------------

    def _register_builtin_phases(self) -> None:
        self.register_phase("adaslot", _build_adaslot)
        self.register_phase("kmeans",  _build_kmeans)
        self.register_phase("phase_a", _build_phase_a)
        self.register_phase("phase_b", _build_phase_b)
        self.register_phase("slda",    _build_slda)

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _run_evaluation(self, test_loader: DataLoader) -> Dict[str, float]:
        trainer = SLDATrainer(
            config=self.config.slda,
            slot_model=self.slot_model,
            agents=self.agents,
            aggregator=self.aggregator,
            slda=self.slda,
            vaes=self.vaes,
        )
        return trainer.evaluate(test_loader)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_task_checkpoint(self, task_id: int) -> None:
        root = self.config.checkpoint_root
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"pipeline_task{task_id}.pt")
        state = {
            "task_id":    task_id,
            "slot_model": self.slot_model.state_dict(),
            "aggregator": self.aggregator.state_dict(),
            "agents":     [a.state_dict() for a in self.agents],
            "slda":       self.slda.state_dict(),
            # VAEs have their own state_dict method
            "vaes":       [v.state_dict() for v in self.vaes],
        }
        torch.save(state, path)
        logger.info(f"[ContinualPipeline] Checkpoint saved → {path}")

    def load_task_checkpoint(self, task_id: int) -> None:
        """Restore pipeline state from a task checkpoint."""
        path = os.path.join(
            self.config.checkpoint_root, f"pipeline_task{task_id}.pt"
        )
        state = torch.load(path, weights_only=False)
        self.slot_model.load_state_dict(state["slot_model"])
        self.aggregator.load_state_dict(state["aggregator"])
        for agent, sd in zip(self.agents, state["agents"]):
            agent.load_state_dict(sd)
        self.slda.load_state_dict(state["slda"])
        for vae, sd in zip(self.vaes, state["vaes"]):
            vae.load_state_dict(sd)
        logger.info(
            f"[ContinualPipeline] Restored checkpoint task {task_id} from {path}"
        )


# ---------------------------------------------------------------------------
# Built-in phase builder functions
# (free functions so they can be easily replaced / patched)
# ---------------------------------------------------------------------------

def _build_adaslot(pipeline: ContinualPipeline, loader: DataLoader):
    """Phase 0: AdaSlot fine-tuning."""
    cfg = pipeline.config.adaslot
    trainer = AdaSlotTrainer(
        config=cfg,
        slot_model=pipeline.slot_model,
        primitive_predictor=pipeline.primitive_predictor,
    )
    metrics = trainer.train(loader)
    return metrics


def _build_kmeans(pipeline: ContinualPipeline, loader: DataLoader):
    """Init: slot extraction → KMeans → spawn agents & VAEs."""
    cfg = pipeline.config.kmeans
    slots = extract_slots(pipeline.slot_model, loader, cfg)

    init = ClusterInitialiser(cfg)
    vaes, agents, result = init.run(
        slots,
        agent_input_dim=pipeline.config.slda.feature_dim,
        agent_output_dim=pipeline.config.slda.feature_dim,
    )

    pipeline.vaes = vaes
    pipeline.agents = agents

    return {"n_clusters": result.n_clusters, "method": cfg.method}


def _build_phase_a(pipeline: ContinualPipeline, loader: DataLoader):
    """Phase A: agent warm-up with L_agent only."""
    if not pipeline.agents:
        logger.warning("[Pipeline] No agents available — skipping Phase A.")
        return {}

    cfg = pipeline.config.phase_a
    trainer = AgentPhaseATrainer(
        config=cfg,
        slot_model=pipeline.slot_model,
        vaes=pipeline.vaes,
        agents=pipeline.agents,
    )
    metrics = trainer.train(loader)
    return metrics


def _build_phase_b(pipeline: ContinualPipeline, loader: DataLoader):
    """Phase B: full agent training with soft routing + label losses."""
    if not pipeline.agents:
        logger.warning("[Pipeline] No agents available — skipping Phase B.")
        return {}

    cfg = pipeline.config.phase_b
    trainer = AgentPhaseBTrainer(
        config=cfg,
        slot_model=pipeline.slot_model,
        vaes=pipeline.vaes,
        agents=pipeline.agents,
        aggregator=pipeline.aggregator,
    )
    metrics = trainer.train(loader)

    # Freeze agents after Phase B
    for agent in pipeline.agents:
        if hasattr(agent, "freeze"):
            agent.freeze()
        else:
            for p in agent.parameters():
                p.requires_grad_(False)

    return metrics


def _build_slda(pipeline: ContinualPipeline, loader: DataLoader):
    """Phase C: incremental SLDA fitting."""
    cfg = pipeline.config.slda
    trainer = SLDATrainer(
        config=cfg,
        slot_model=pipeline.slot_model,
        agents=pipeline.agents,
        aggregator=pipeline.aggregator,
        slda=pipeline.slda,
        vaes=pipeline.vaes,
    )
    trainer.fit(loader)
    return {"slda_samples": pipeline.slda._n_total}
