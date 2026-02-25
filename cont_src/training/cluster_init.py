"""
Cluster Initialiser  –  Between Phase 0 and Phase A
=====================================================
Extracts slot embeddings from the *frozen* backbone, clusters them using
**any algorithm in ``CLUSTERING_REGISTRY``**, then spawns one ``SlotVAE``
+ one ``ResidualMLPAgent`` per cluster.

Supported methods (selected via ``ClusterInitConfig.method``):
    "kmeans"           – KMeansClustering       (needs n_clusters)
    "minibatch_kmeans" – MiniBatchKMeansClustering
    "dbscan"           – DBSCANClustering        (auto cluster count)
    "hdbscan"          – HDBSCANClustering       (auto cluster count, best default)
    "agglomerative"    – AgglomerativeClustering (needs n_clusters or threshold)
    "bayesian_gmm"     – GaussianMixtureClustering (soft/auto-prunes)

Adding your own algorithm:
    @CLUSTERING_REGISTRY.register("my_algo")
    class MyClustering(BaseClustering): ...

    cfg = ClusterInitConfig(method="my_algo", method_kwargs={...})

Pipeline
--------
    1. ``extract_slots(model, loader, cfg)``  →  ``slots_np`` (N, D)
    2. ``ClusterInitialiser.fit(slots_np)``   →  ``ClusteringResult``
    3. ``ClusterInitialiser.spawn()``
           → ``List[SlotVAE]``            (one per cluster, VAE trained)
           → ``List[ResidualMLPAgent]``   (one per cluster, untrained)

Usage
-----
    from cont_src.training.configs      import ClusterInitConfig
    from cont_src.training.cluster_init import ClusterInitialiser, extract_slots

    # HDBSCAN (auto-discovers cluster count)
    cfg   = ClusterInitConfig(method="hdbscan",
                              method_kwargs={"min_cluster_size": 20, "min_samples": 5})
    slots = extract_slots(slot_model, train_loader, cfg)
    init  = ClusterInitialiser(cfg)
    vaes, agents, result = init.run(slots)
    print(f"Discovered {result.n_clusters} clusters")

    # KMeans (fixed count)
    cfg2  = ClusterInitConfig(method="kmeans", n_clusters=8)
    ...
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cont_src.clustering import CLUSTERING_REGISTRY, ClusteringResult
from cont_src.models.agents.residual_mlp_agent import ResidualMLPAgent
from cont_src.models.routers.slot_vae import SlotVAE, ScoringMode
from cont_src.training.configs import ClusterInitConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slot extraction  (shared across all clustering methods)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_slots(
    slot_model: nn.Module,
    dataloader: DataLoader,
    config: ClusterInitConfig,
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
    config : ClusterInitConfig
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

        if isinstance(batch, (list, tuple)):
            images = batch[0]
        elif isinstance(batch, dict):
            images = batch["image"]
        else:
            images = batch

        images = images.to(device)
        out    = slot_model(images)
        slots  = out["slots"]          # (B, K, D)

        # Only keep active slots (hard_keep_decision > 0.5) when available
        mask = out.get("hard_keep_decision", out.get("mask"))  # (B, K) or None
        B, K, D = slots.shape
        flat = slots.reshape(B * K, D).cpu().numpy()
        if mask is not None:
            active = (mask > 0.5).reshape(B * K).cpu().numpy().astype(bool)
            flat = flat[active]

        if flat.shape[0] > 0:
            all_slots.append(flat)

    if not all_slots:
        raise RuntimeError("No slots extracted — check dataloader and model.")

    result = np.concatenate(all_slots, axis=0)   # (N, D)

    # Subsample if over the cap
    max_s = config.max_slots_for_clustering
    n_before = result.shape[0]
    if max_s > 0 and n_before > max_s:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_before, size=max_s, replace=False)
        result = result[idx]
        logger.info(
            f"[ClusterInit] Subsampled {n_before} active slots → {max_s}"
        )
    logger.info(
        f"[ClusterInit] Extracted {result.shape[0]} slot embeddings "
        f"(dim={result.shape[1]})  method={config.method}"
    )
    return result


# ---------------------------------------------------------------------------
# Cluster + spawn
# ---------------------------------------------------------------------------

class ClusterInitialiser:
    """
    Builds a clustering algorithm from ``CLUSTERING_REGISTRY``, fits it on
    slot embeddings, then spawns one ``SlotVAE`` + one ``ResidualMLPAgent``
    per discovered cluster.

    Parameters
    ----------
    config : ClusterInitConfig
    """

    def __init__(self, config: ClusterInitConfig):
        self.config = config
        self.device = self._resolve_device(config.device)

        self._result:          Optional[ClusteringResult] = None
        self._cluster_slots:   Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, slots: np.ndarray) -> ClusteringResult:
        """
        Cluster ``slots`` (N, D) using the configured algorithm.

        Returns
        -------
        ClusteringResult  — labels, centres, n_clusters, scores, metadata
        """
        cfg = self.config

        # Build algorithm kwargs
        # For fixed-count algorithms, inject n_clusters if not overridden
        kwargs = dict(cfg.method_kwargs)   # shallow copy
        if "n_clusters" not in kwargs and cfg.n_clusters > 0:
            # Only inject for algorithms that accept n_clusters
            _fixed_count = {"kmeans", "minibatch_kmeans", "agglomerative", "bayesian_gmm"}
            if cfg.method in _fixed_count:
                kwargs["n_clusters"] = cfg.n_clusters

        if "random_state" not in kwargs:
            kwargs["random_state"] = cfg.random_state

        log_kw = {k: v for k, v in kwargs.items() if k != "random_state"}
        logger.info(
            f"[ClusterInit] Fitting '{cfg.method}'  "
            f"n_samples={slots.shape[0]}  params={log_kw}"
        )

        clusterer = CLUSTERING_REGISTRY.build(cfg.method, **kwargs)
        slots_t   = torch.tensor(slots, dtype=torch.float32)
        result    = clusterer.fit_predict(slots_t)

        # ── Fallback: density-based methods may find 0 clusters (all noise) ──
        # Fall back to KMeans with cfg.n_clusters so the pipeline never stalls.
        if result.n_clusters == 0:
            fallback_k = max(cfg.n_clusters, 2)
            logger.warning(
                f"[ClusterInit] '{cfg.method}' found 0 clusters (all noise). "
                f"Falling back to kmeans with n_clusters={fallback_k}."
            )
            fb_clusterer = CLUSTERING_REGISTRY.build(
                "kmeans",
                n_clusters=fallback_k,
                random_state=cfg.random_state,
            )
            result = fb_clusterer.fit_predict(slots_t)

        self._result = result

        # Partition slots per cluster (noise = -1 is dropped)
        self._cluster_slots = []
        for k in range(result.n_clusters):
            mask = result.labels == k
            self._cluster_slots.append(slots[mask])

        sizes = [len(cs) for cs in self._cluster_slots]
        logger.info(
            f"[ClusterInit] '{cfg.method}' → {result.n_clusters} clusters  "
            f"sizes={sizes}  "
            f"noise={int((result.labels == -1).sum())}"
        )
        if result.scores:
            sil = result.scores.get("silhouette", None)
            if sil is not None:
                logger.info(f"[ClusterInit] Silhouette score: {sil:.4f}")

        return result

    def spawn(
        self,
        *,
        agent_input_dim:  Optional[int] = None,
        agent_output_dim: Optional[int] = None,
        agent_num_blocks: int = 2,
    ) -> Tuple[List[SlotVAE], List[ResidualMLPAgent]]:
        """
        Spawn and train one SlotVAE + one ResidualMLPAgent per cluster.

        Must be called after ``fit()``.

        Parameters
        ----------
        agent_input_dim : int, optional
            Slot dimension — inferred from cluster centres if not given.
        agent_output_dim : int, optional
            Agent hidden dim.  Defaults to input_dim.

        Returns
        -------
        vaes   : List[SlotVAE]
        agents : List[ResidualMLPAgent]
        """
        if self._cluster_slots is None:
            raise RuntimeError("Call fit() before spawn().")

        cfg      = self.config
        slot_dim = self._result.centers.shape[1] if self._result.centers is not None else (
            self._cluster_slots[0].shape[1] if self._cluster_slots else 64
        )
        in_dim  = agent_input_dim  if agent_input_dim  is not None else slot_dim
        out_dim = agent_output_dim if agent_output_dim is not None else in_dim

        vaes:   List[SlotVAE]          = []
        agents: List[ResidualMLPAgent] = []
        _valid_modes = {"generative", "mahal_z", "mahal_slot"}
        assert cfg.scoring_mode in _valid_modes, \
            f"scoring_mode must be one of {_valid_modes}, got '{cfg.scoring_mode}'"
        scoring_mode: ScoringMode = cfg.scoring_mode  # type: ignore[assignment]

        cluster_bar = tqdm(
            enumerate(self._cluster_slots),
            total=len(self._cluster_slots),
            desc="[ClusterInit] Training VAEs",
            unit="cluster",
        )
        for k, cluster_slots in cluster_bar:
            n = len(cluster_slots)
            cluster_bar.set_postfix(cluster=k, n_slots=n)
            logger.info(
                f"[ClusterInit] Spawning cluster {k}/{self._result.n_clusters}  "
                f"n_slots={n}"
            )

            if n == 0:
                logger.warning(f"[ClusterInit] Cluster {k} is empty — skipping.")
                continue

            # ---- SlotVAE ----
            vae     = SlotVAE(
                slot_dim    = slot_dim,
                latent_dim  = cfg.vae_latent_dim,
            )
            slots_t = torch.tensor(cluster_slots, dtype=torch.float32).to(self.device)
            vae.train_vae(slots_t, epochs=cfg.vae_epochs, beta=cfg.vae_beta)
            vae.update_stats(slots_t)
            vaes.append(vae)

            # ---- Agent ----
            agent = ResidualMLPAgent(
                input_dim  = in_dim,
                output_dim = out_dim,
                num_blocks = agent_num_blocks,
                use_decoder= True,
            ).to(self.device)
            agents.append(agent)

        logger.info(
            f"[ClusterInit] Spawned {len(vaes)} VAEs and {len(agents)} agents."
        )
        return vaes, agents

    def run(
        self,
        slots: np.ndarray,
        **spawn_kwargs,
    ) -> Tuple[List[SlotVAE], List[ResidualMLPAgent], ClusteringResult]:
        """
        Convenience: fit + spawn in one call.

        Returns
        -------
        vaes, agents, clustering_result
        """
        result = self.fit(slots)
        vaes, agents = self.spawn(**spawn_kwargs)
        return vaes, agents, result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def result(self) -> Optional[ClusteringResult]:
        return self._result

    @property
    def cluster_slots(self) -> Optional[List[np.ndarray]]:
        return self._cluster_slots

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

# Old code that used KMeansInitialiser still works; it will now use
# whatever method is in the config (HDBSCAN by default).
KMeansInitialiser = ClusterInitialiser
