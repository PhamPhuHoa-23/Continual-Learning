"""
kmeans_init.py  –  backward-compatibility shim
================================================
All logic has moved to ``cluster_init.py`` which supports every algorithm
in ``CLUSTERING_REGISTRY``.  This file just re-exports everything so that
old ``from cont_src.training.kmeans_init import ...`` still works.
"""
from cont_src.training.cluster_init import (   # noqa: F401
    ClusterInitialiser,
    ClusterInitialiser as KMeansInitialiser,
    extract_slots,
)
from cont_src.training.configs import (
    ClusterInitConfig,
    ClusterInitConfig as KMeansInitConfig,
)

__all__ = [
    "ClusterInitialiser",
    "KMeansInitialiser",
    "extract_slots",
    "ClusterInitConfig",
    "KMeansInitConfig",
]

# ---- original docstring kept below for reference ----
"""
KMeans Initialiser  –  Between Phase 0 and Phase A
====================================================
Extracts slot embeddings from the *frozen* backbone, clusters them with
KMeans, then spawns one ``SlotVAE`` + one ``ResidualMLPAgent`` per cluster.

Pipeline
--------
    1. ``extract_slots(model, loader)``  →  ``slots_np`` of shape (N, D)
    2. ``KMeansInitialiser.fit(slots_np)``  →  cluster labels + centres
    3. ``KMeansInitialiser.spawn()``
           → ``List[SlotVAE]``   (one per cluster, VAE trained)
           → ``List[ResidualMLPAgent]``  (one per cluster, untrained)

Usage
-----
    from cont_src.training.configs     import KMeansInitConfig
    from cont_src.training.kmeans_init import KMeansInitialiser, extract_slots

    cfg   = KMeansInitConfig(n_clusters=8, scoring_mode="generative")
    slots = extract_slots(slot_model, train_loader, cfg)
    init  = KMeansInitialiser(cfg)
    vaes, agents, labels = init.run(slots)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cont_src.models.agents.residual_mlp_agent import ResidualMLPAgent
from cont_src.models.routers.slot_vae import SlotVAE, ScoringMode
from cont_src.training.configs import KMeansInitConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: slot extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_slots(
    slot_model: nn.Module,
    dataloader: DataLoader,
    config: KMeansInitConfig,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Run the frozen slot model over ``dataloader`` and collect all slot
    embeddings into a single NumPy array of shape ``(N_total_slots, D)``.

    The slot model's ``forward`` must return a dict with key ``"slots"``
    of shape ``(B, K, D)``.

    Parameters
    ----------
    slot_model : nn.Module
    dataloader : DataLoader
    config : KMeansInitConfig
    device : torch.device, optional

    Returns
    -------
    np.ndarray  shape (N_total_slots, D)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slot_model.eval()
    slot_model.to(device)

    all_slots: List[np.ndarray] = []
    max_batches = config.max_batches_for_clustering

    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        # Unpack — handle tuple (images, labels) or dict
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        elif isinstance(batch, dict):
            images = batch["image"]
        else:
            images = batch

        images = images.to(device)
        out = slot_model(images)
        slots = out["slots"]  # (B, K, D)

        B, K, D = slots.shape
        all_slots.append(slots.reshape(B * K, D).cpu().numpy())

    if not all_slots:
        raise RuntimeError("No slots extracted — check dataloader and model.")

    result = np.concatenate(all_slots, axis=0)   # (N, D)
    logger.info(f"[KMeansInit] Extracted {result.shape[0]} slot embeddings (dim={result.shape[1]})")
    return result


# ---------------------------------------------------------------------------
# Step 2/3: cluster + spawn
# ---------------------------------------------------------------------------

class KMeansInitialiser:
    """
    Clusters slots with KMeans, then spawns one VAE + one Agent per cluster.

    Parameters
    ----------
    config : KMeansInitConfig
    """

    def __init__(self, config: KMeansInitConfig):
        self.config = config
        self.device = self._resolve_device(config.device)

        self._labels: Optional[np.ndarray] = None
        self._centres: Optional[np.ndarray] = None
        self._cluster_slots: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, slots: np.ndarray) -> np.ndarray:
        """
        Run KMeans on ``slots`` (N, D).

        Returns
        -------
        np.ndarray  shape (N,)  — cluster assignment for each slot
        """
        from sklearn.cluster import KMeans  # lazy import

        cfg = self.config
        logger.info(
            f"[KMeansInit] Fitting KMeans  K={cfg.n_clusters}  "
            f"n_samples={slots.shape[0]}"
        )
        km = KMeans(
            n_clusters=cfg.n_clusters,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
        self._labels  = km.fit_predict(slots)
        self._centres = km.cluster_centers_           # (K, D)

        # partition slots by cluster for VAE training
        self._cluster_slots = [
            slots[self._labels == k] for k in range(cfg.n_clusters)
        ]

        sizes = [len(cs) for cs in self._cluster_slots]
        logger.info(f"[KMeansInit] Cluster sizes: {sizes}")
        return self._labels

    def spawn(
        self,
        *,
        agent_input_dim: Optional[int] = None,
        agent_output_dim: Optional[int] = None,
        agent_num_blocks: int = 2,
    ) -> Tuple[List[SlotVAE], List[ResidualMLPAgent]]:
        """
        Spawn and train one SlotVAE + one ResidualMLPAgent per cluster.

        Must be called after ``fit()``.

        Parameters
        ----------
        agent_input_dim : int, optional
            Slot dimension.  Inferred from cluster centres if not given.
        agent_output_dim : int, optional
            Hidden dimension of agent output.  Defaults to input_dim.

        Returns
        -------
        vaes   : List[SlotVAE]           length = n_clusters
        agents : List[ResidualMLPAgent]  length = n_clusters
        """
        if self._cluster_slots is None:
            raise RuntimeError("Call fit() before spawn().")

        cfg = self.config
        slot_dim = self._centres.shape[1]
        in_dim  = agent_input_dim  if agent_input_dim  is not None else slot_dim
        out_dim = agent_output_dim if agent_output_dim is not None else in_dim

        vaes:   List[SlotVAE]             = []
        agents: List[ResidualMLPAgent]    = []

        _valid_modes = {"generative", "mahal_z", "mahal_slot"}
        assert cfg.scoring_mode in _valid_modes, \
            f"scoring_mode must be one of {_valid_modes}, got '{cfg.scoring_mode}'"
        scoring_mode: ScoringMode = cfg.scoring_mode  # type: ignore[assignment]

        for k, cluster_slots in enumerate(self._cluster_slots):
            logger.info(
                f"[KMeansInit] Spawning cluster {k}/{cfg.n_clusters}  "
                f"n_slots={len(cluster_slots)}"
            )

            # ---- SlotVAE ----
            vae = SlotVAE(
                slot_dim=slot_dim,
                latent_dim=cfg.vae_latent_dim,
            )
            slots_t = torch.tensor(cluster_slots, dtype=torch.float32).to(self.device)
            vae.train_vae(slots_t, epochs=cfg.vae_epochs, beta=cfg.vae_beta)
            vae.update_stats(slots_t)          # initialise Welford stats
            vaes.append(vae)

            # ---- Agent ----
            agent = ResidualMLPAgent(
                input_dim=in_dim,
                output_dim=out_dim,
                num_blocks=agent_num_blocks,
                use_decoder=True,
            ).to(self.device)
            agents.append(agent)

        logger.info(
            f"[KMeansInit] Spawned {len(vaes)} VAEs and {len(agents)} agents."
        )
        return vaes, agents

    # ------------------------------------------------------------------
    # Convenience: run fit + spawn in one call
    # ------------------------------------------------------------------

    def run(
        self,
        slots: np.ndarray,
        **spawn_kwargs,
    ) -> Tuple[List[SlotVAE], List[ResidualMLPAgent], np.ndarray]:
        """
        fit + spawn.

        Returns
        -------
        vaes, agents, cluster_labels
        """
        labels = self.fit(slots)
        vaes, agents = self.spawn(**spawn_kwargs)
        return vaes, agents, labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def centres(self) -> Optional[np.ndarray]:
        return self._centres

    @property
    def labels(self) -> Optional[np.ndarray]:
        return self._labels

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
