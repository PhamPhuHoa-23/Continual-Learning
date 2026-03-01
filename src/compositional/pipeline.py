"""
Compositional Sub-Concept Routing Pipeline.

Implements Algorithm 1 from 'Compositional Sub-Concept Routing for
Exemplar-Free Class-Incremental Learning' (prototype_training-pipeline.tex).

Components:
    agents      — dict of AgentNet (ResidualMLP encoder + lightweight decoder)
    aggregator  — BlockDiagonalAggregator (constant d_h output)
    router      — VAERouter (Mahalanobis routing + Welford stats)
    slda        — StreamingLDAAggregator (analytic incremental classifier)
    novel_buf   — buffer of unassigned slots for spawning new agents

Freeze schedule (guarantees zero forgetting):
    Task 1: train agents + keys → freeze → train VAEs → freeze → init SLDA
    Task t: route → collect novel buffer → spawn new agents → train new agents
            → train new VAEs → freeze → update SLDA (old classes unmodified)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from hdbscan import HDBSCAN
except ImportError:
    # Fallback to sklearn DBSCAN if hdbscan not available
    from sklearn.cluster import DBSCAN as HDBSCAN
from typing import Iterator, Tuple, List, Dict, Optional

from src.compositional.models import AgentNet, BlockDiagonalAggregator
from src.compositional.vae_router import VAERouter
from src.compositional.losses import CompositionalLoss
from src.slot_multi_agent.aggregator import StreamingLDAAggregator


class CompositionalRoutingPipeline:
    """
    Full compositional routing pipeline for exemplar-free class-incremental learning.

    Args:
        slot_dim:     Slot dimension from AdaSlot backbone
        d_h:          Hidden-label dimension (constant output of aggregator)
        latent_dim:   VAE latent dimension for routing
        theta_match:  Mahalanobis score threshold — above ⇒ match existing agent
        theta_novel:  (unused, kept for API compat) — below ⇒ unassigned
        b_min:        Min novel-buffer size before spawning new agents
        rho_min:      Min intra-cluster cosine similarity for spawning
        n_min:        Min cluster size for spawning
        loss_weights: Dict with keys alpha, beta, gamma, delta
        device:       'cuda' or 'cpu'
    """

    def __init__(
        self,
        slot_dim: int = 64,
        d_h: int = 128,
        latent_dim: int = 32,
        theta_match: float = -50.0,
        theta_novel: float = -200.0,
        b_min: int = 50,
        rho_min: float = 0.7,
        n_min: int = 10,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.slot_dim   = slot_dim
        self.d_h        = d_h
        self.latent_dim = latent_dim
        self.theta_match = theta_match
        self.theta_novel = theta_novel
        self.b_min  = b_min
        self.rho_min = rho_min
        self.n_min  = n_min

        lw = loss_weights or {}
        alpha = lw.get("alpha", 1.0)
        beta  = lw.get("beta",  0.5)
        gamma = lw.get("gamma", 0.5)
        delta = lw.get("delta", 0.1)

        # ── modules ──────────────────────────────────────────────────────────
        self.agents:     nn.ModuleDict   = nn.ModuleDict()   # {id: AgentNet}
        self.aggregator: BlockDiagonalAggregator = (
            BlockDiagonalAggregator(d_h=d_h).to(self.device)
        )
        self.router:  VAERouter = VAERouter(
            slot_dim=slot_dim, latent_dim=latent_dim
        ).to(self.device)
        self.slda: StreamingLDAAggregator = StreamingLDAAggregator(
            input_dim=d_h
        )
        self.loss_fn: CompositionalLoss = CompositionalLoss(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta
        ).to(self.device)

        # ── state ─────────────────────────────────────────────────────────────
        self.novel_buf:      List[torch.Tensor] = []   # unassigned slot tensors
        self.new_agent_ids:  List[str] = []            # spawned current task
        self.task1_cluster_centers: Optional[Dict[str, torch.Tensor]] = None  # Cluster centers for Task 1 routing

    # ── internal helpers ──────────────────────────────────────────────────────

    def _next_agent_id(self) -> str:
        return f"agent_{len(self.agents)}"

    def _spawn_agent(self, agent_id: str) -> None:
        """Create agent, key, and VAE for agent_id."""
        agent = AgentNet(slot_dim=self.slot_dim, d_h=self.d_h).to(self.device)
        self.agents[agent_id] = agent
        self.aggregator.add_agent_key(agent_id, device=self.device)
        self.router.spawn_vae(agent_id, self.device)

    def _route_slots(self, slots_flat: torch.Tensor) -> List[str]:
        """Route (N, slot_dim) slots → list of agent_id/'unassigned' strings."""
        # During Task 1 training, use cluster centers instead of VAEs (which aren't trained yet)
        if self.task1_cluster_centers is not None:
            return self._route_by_cluster_centers(slots_flat)
        return self.router.route(slots_flat, self.theta_match, self.theta_novel)
    
    def _route_by_cluster_centers(self, slots_flat: torch.Tensor) -> List[str]:
        """Route slots using nearest cluster center (Task 1 only)."""
        N = slots_flat.size(0)
        if not self.task1_cluster_centers:
            return ["unassigned"] * N
        
        agent_ids = list(self.task1_cluster_centers.keys())
        centers = torch.stack([self.task1_cluster_centers[aid] for aid in agent_ids])  # (M, slot_dim)
        
        # Compute distances: (N, slot_dim) vs (M, slot_dim) -> (N, M)
        slots_norm = F.normalize(slots_flat, p=2, dim=1)
        centers_norm = F.normalize(centers, p=2, dim=1)
        similarities = torch.matmul(slots_norm, centers_norm.t())  # (N, M) cosine similarity
        
        best_idx = similarities.argmax(dim=1)  # (N,)
        return [agent_ids[idx.item()] for idx in best_idx]

    @torch.no_grad()
    def _encode_h(
        self, slots: torch.Tensor, sigma_flat: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """
        Compute h_k for every slot, reshape to (B, K, d_h).
        Returns:
            h:          (B, K, d_h)
            slots_3d:   (B, K, slot_dim)   — original slots reshaped
            sigma_2d:   (B, K) list of lists
        """
        B, K, D = slots.shape
        h = torch.zeros(B, K, self.d_h, device=self.device)
        sigma_2d = [sigma_flat[b * K:(b + 1) * K] for b in range(B)]

        for b in range(B):
            for k in range(K):
                aid = sigma_2d[b][k]
                if aid != "unassigned" and aid in self.agents:
                    h[b, k] = self.agents[aid](slots[b, k].unsqueeze(0)).squeeze(0)

        return h, sigma_2d

    def _encode_h_with_grad(
        self, slots: torch.Tensor, sigma_flat: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """Same as _encode_h but WITH gradient (for training)."""
        B, K, D = slots.shape
        h = torch.zeros(B, K, self.d_h, device=self.device)
        sigma_2d = [sigma_flat[b * K:(b + 1) * K] for b in range(B)]

        for b in range(B):
            for k in range(K):
                aid = sigma_2d[b][k]
                if aid != "unassigned" and aid in self.agents:
                    h[b, k] = self.agents[aid](slots[b, k].unsqueeze(0)).squeeze(0)

        return h, sigma_2d

    # ── public API ────────────────────────────────────────────────────────────

    def init_agents_from_clustering(
        self,
        encoded_iter: Iterator,
        M0: int = 10,
        min_cluster_size: int = 10,
        min_samples: int = 5,
    ) -> None:
        """
        Task 1 initialization: collect all slots, HDBSCAN → spawn agents adaptively.

        Args:
            encoded_iter:      Generator yielding (X, y, slots_cpu) per batch.
            M0:                Max number of agents (for backwards compat; not enforced)
            min_cluster_size:  Min cluster size for HDBSCAN
            min_samples:       Min samples for HDBSCAN core points
        """
        all_slots = []
        for _, _, slots in encoded_iter:
            flat = slots.view(-1, self.slot_dim)
            all_slots.append(flat)
        all_slots = torch.cat(all_slots, dim=0)          # (N_total, slot_dim)

        slots_np = all_slots.cpu().numpy()
        
        # HDBSCAN/DBSCAN: auto-discovers number of clusters
        try:
            # Prefer HDBSCAN (better for varying densities)
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
            )
            labels = clusterer.fit_predict(slots_np)
            algo_name = "HDBSCAN"
        except TypeError:
            # Fallback to DBSCAN if HDBSCAN not available
            # eps = adaptive based on data scale
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(slots_np)
            distances, _ = nbrs.kneighbors(slots_np)
            eps = np.percentile(distances[:, -1], 50)  # median distance
            
            clusterer = HDBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(slots_np)
            algo_name = "DBSCAN"
        
        # Get unique clusters (excluding noise = -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        unique_labels = sorted(unique_labels)
        
        n_clusters = len(unique_labels)
        n_noise = (labels == -1).sum()
        print(f"  {algo_name} discovered {n_clusters} clusters (noise: {n_noise}/{len(labels)} slots)")

        # Store cluster centers for routing during Task 1 training
        self.task1_cluster_centers = {}
        
        # Group slots by cluster (for VAE training later)
        self._task1_clusters: Dict[str, List[torch.Tensor]] = {}
        
        for cluster_idx in unique_labels:
            aid = self._next_agent_id()
            self._spawn_agent(aid)
            self.new_agent_ids.append(aid)
            
            mask = labels == cluster_idx
            cluster_slots = all_slots[mask].to(self.device)
            self._task1_clusters[aid] = [cluster_slots]
            
            # Compute cluster center (mean of cluster members)
            cluster_center = cluster_slots.mean(dim=0)
            self.task1_cluster_centers[aid] = cluster_center

    def trainable_params(self) -> List[torch.nn.Parameter]:
        """Return all currently trainable parameters (unfrozen agents + keys)."""
        params = []
        for aid, agent in self.agents.items():
            params.extend(p for p in agent.parameters() if p.requires_grad)
        for aid, key in self.aggregator.keys.items():
            if key.requires_grad:
                params.append(key)
        return params

    def new_agent_params(self) -> List[torch.nn.Parameter]:
        """Return parameters of agents spawned in the current task."""
        params = []
        for aid in self.new_agent_ids:
            if aid in self.agents:
                params.extend(self.agents[aid].parameters())
            if aid in self.aggregator.keys:
                params.append(self.aggregator.keys[aid])
        return params

    def train_mode(self, new_only: bool = False) -> None:
        """Set training mode. If new_only, only newly spawned agents are set to train."""
        self.aggregator.train()
        if new_only:
            for aid, agent in self.agents.items():
                if aid in self.new_agent_ids:
                    agent.train()
                else:
                    agent.eval()
        else:
            for agent in self.agents.values():
                agent.train()

    def compute_losses(
        self,
        slots: torch.Tensor,           # (B, K, slot_dim) on device
        y: torch.Tensor,               # (B,) labels
        task_id: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Route slots, compute H, and compute all losses.

        Returns dict with keys: total, Lp, SupCon, agent, local.
        """
        B, K, _ = slots.shape
        slots_flat = slots.view(B * K, self.slot_dim)

        # Route (no grad needed for routing itself)
        with torch.no_grad():
            sigma_flat = self._route_slots(slots_flat)

        sigma_2d = [sigma_flat[b * K:(b + 1) * K] for b in range(B)]
        dec_list, slot_list = [], []   # for L_agent

        # Build h via stacking (NOT in-place) to keep autograd graph intact
        h_rows = []
        for b in range(B):
            h_cols = []
            for k in range(K):
                aid = sigma_2d[b][k]
                if aid != "unassigned" and aid in self.agents:
                    s_k  = slots[b, k].unsqueeze(0)           # (1, slot_dim)
                    h_k  = self.agents[aid](s_k)              # (1, d_h) - has gradient!
                    dec_k = self.agents[aid].decode(h_k)      # (1, slot_dim)
                    dec_list.append(dec_k)
                    slot_list.append(s_k)
                    h_cols.append(h_k.squeeze(0))             # (d_h,)
                else:
                    # Unassigned slot - create zero tensor connected to parameters
                    # Sum all keys with weight 0 to maintain gradient flow
                    h_zero = None
                    for key in self.aggregator.keys.values():
                        if h_zero is None:
                            h_zero = key * 0.0
                        else:
                            h_zero = h_zero + key * 0.0
                    # Fallback if no keys
                    if h_zero is None:
                        h_zero = torch.zeros(self.d_h, device=self.device)
                    h_cols.append(h_zero)
            h_rows.append(torch.stack(h_cols, dim=0))        # (K, d_h)
        h = torch.stack(h_rows, dim=0)                        # (B, K, d_h)

        # H = aggregated representation (still differentiable through h)
        H = self.aggregator(h, sigma_2d)                      # (B, d_h)

        # Stack decoder outputs (if any)
        if dec_list:
            dec_h  = torch.cat(dec_list, dim=0)               # (M, slot_dim)
            s_orig = torch.cat(slot_list, dim=0)              # (M, slot_dim)
        else:
            # No slots were assigned - create loss connected to parameters
            zero_loss = torch.zeros(1, device=self.device, requires_grad=True)
            for key in self.aggregator.keys.values():
                if key.requires_grad:
                    zero_loss = zero_loss + key.sum() * 0.0
            for agent in self.agents.values():
                for p in agent.parameters():
                    if p.requires_grad:
                        zero_loss = zero_loss + p.sum() * 0.0
                        break
                break
            return {k: zero_loss.squeeze() for k in ["total", "Lp", "SupCon", "agent", "local"]}

        result = self.loss_fn(H=H, y=y, dec_h=dec_h, s=s_orig)
        return {
            "total":  result["loss"],
            "Lp":     result["l_p"],
            "SupCon": result["l_supcon"],
            "agent":  result["l_agent"],
            "local":  result["l_local"],
        }


    def train_vaes(
        self,
        encoded_iter: Iterator,
        task_id: int,
        vae_epochs: int = 10,
        vae_lr: float = 1e-3,
    ) -> None:
        """
        For new agents: train their VAE then freeze it and init Welford stats.
        For old agents: update Welford stats only (network already frozen).
        """
        # Collect slot batches per agent
        agent_slot_batches: Dict[str, List[torch.Tensor]] = {
            aid: [] for aid in self.agents
        }

        for _, _, slots in encoded_iter:
            slots = slots.to(self.device)
            B, K, D = slots.shape
            with torch.no_grad():
                slots_flat = slots.view(B * K, D)
                sigma_flat = self._route_slots(slots_flat)

            for idx, aid in enumerate(sigma_flat):
                if aid != "unassigned" and aid in self.agents:
                    agent_slot_batches[aid].append(
                        slots_flat[idx].unsqueeze(0)
                    )

        # If task 1, also use cluster-assigned slots from init clustering
        if task_id == 1 and hasattr(self, "_task1_clusters"):
            for aid, batches in self._task1_clusters.items():
                agent_slot_batches[aid].extend(batches)

        for aid in self.agents:
            batches = agent_slot_batches[aid]
            if not batches:
                continue

            if aid in self.new_agent_ids:
                # Train VAE, then freeze, then init stats
                self.router.train_vae(aid, batches, epochs=vae_epochs, lr=vae_lr)

            # Update Welford stats (works for both new and old agents)
            for batch in batches:
                batch = batch.to(self.device)
                mu_q, _ = self.router.vaes[aid].encode(batch)
                self.router.update_stats(aid, mu_q.detach())

    def freeze_task(self, task_id: int) -> None:
        """Freeze all agents (and their keys) spawned during this task."""
        for aid in self.new_agent_ids:
            if aid in self.agents:
                for p in self.agents[aid].parameters():
                    p.requires_grad_(False)
                self.agents[aid].eval()
            self.aggregator.freeze_key(aid)
        self.new_agent_ids = []   # reset for next task
        
        # After Task 1, switch from cluster-based routing to VAE-based routing
        if task_id == 1 and self.task1_cluster_centers is not None:
            self.task1_cluster_centers = None

    @torch.no_grad()
    def update_slda(
        self,
        encoded_iter: Iterator,
        class_offset: int = 0,
    ) -> None:
        """
        Route all slots, compute H, feed (H, label) into SLDA.
        Old class statistics are never modified (StreamingLDAAggregator
        only appends when called with a new sample).
        """
        for _, y, slots in encoded_iter:
            slots = slots.to(self.device)
            B, K, _ = slots.shape
            slots_flat = slots.view(B * K, self.slot_dim)
            sigma_flat = self._route_slots(slots_flat)
            sigma_2d = [sigma_flat[b * K:(b + 1) * K] for b in range(B)]

            h = torch.zeros(B, K, self.d_h, device=self.device)
            for b in range(B):
                for k in range(K):
                    aid = sigma_2d[b][k]
                    if aid != "unassigned" and aid in self.agents:
                        h[b, k] = self.agents[aid](
                            slots[b, k].unsqueeze(0)
                        ).squeeze(0)

            H = self.aggregator(h, sigma_2d)   # (B, d_h)
            H_np = H.cpu().numpy()
            y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)

            for i in range(B):
                self.slda.learn_one(H_np[i], int(y_np[i]) + class_offset)

    @torch.no_grad()
    def collect_novel_buffer(self, encoded_iter: Iterator) -> None:
        """
        Routing pass: collect unassigned slots into self.novel_buf.
        Also updates Welford stats for matched agents.
        """
        for _, _, slots in encoded_iter:
            slots = slots.to(self.device)
            B, K, _ = slots.shape
            slots_flat = slots.view(B * K, self.slot_dim)
            sigma_flat = self._route_slots(slots_flat)

            for idx, aid in enumerate(sigma_flat):
                s = slots_flat[idx].unsqueeze(0)          # (1, slot_dim)
                if aid == "unassigned":
                    self.novel_buf.append(s.cpu())
                else:
                    # Update Welford stats for matched agent (VAE frozen)
                    mu_q, _ = self.router.vaes[aid].encode(s)
                    self.router.update_stats(aid, mu_q.detach())

    def spawn_new_agents(
        self,
        n_clusters_hint: Optional[int] = None,
        vae_epochs: int = 10,
        min_cluster_size: int = 10,
        min_samples: int = 5,
    ) -> int:
        """
        Cluster novel_buf and spawn new agents for qualifying clusters.

        A cluster qualifies if:
            - size ≥ self.n_min
            - mean intra-cluster cosine similarity ≥ self.rho_min

        Returns number of new agents spawned.
        """
        if len(self.novel_buf) < self.b_min:
            return 0

        buf = torch.cat(self.novel_buf, dim=0).to(self.device)  # (N, slot_dim)
        self.novel_buf = []

        slots_np = buf.cpu().numpy()
        
        # Use HDBSCAN/DBSCAN for adaptive clustering
        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
            )
            labels = clusterer.fit_predict(slots_np)
            algo_name = "HDBSCAN"
        except (TypeError, AttributeError):
            # Fallback to DBSCAN
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(slots_np)
            distances, _ = nbrs.kneighbors(slots_np)
            eps = np.percentile(distances[:, -1], 50)
            
            clusterer = HDBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(slots_np)
            algo_name = "DBSCAN"
        
        # Get unique clusters (excluding noise = -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        unique_labels = sorted(unique_labels)
        
        n_clusters_found = len(unique_labels)
        n_noise = (labels == -1).sum()
        print(f"  {algo_name} on novel buffer: {n_clusters_found} clusters, {n_noise} noise slots")

        n_spawned = 0
        for c in unique_labels:
            mask = labels == c
            cluster_slots = buf[mask]       # (m, slot_dim)
            m = cluster_slots.size(0)

            if m < self.n_min:
                continue

            # Intra-cluster cosine similarity check
            normed = torch.nn.functional.normalize(cluster_slots, dim=1)
            sim_matrix = normed @ normed.T            # (m, m)
            # Mean off-diagonal
            off_diag = sim_matrix.fill_diagonal_(0)
            mean_sim = off_diag.sum() / max(m * (m - 1), 1)

            if mean_sim.item() < self.rho_min:
                continue

            aid = self._next_agent_id()
            self._spawn_agent(aid)
            self.new_agent_ids.append(aid)
            n_spawned += 1

        return n_spawned

    @torch.no_grad()
    def predict(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for a batch.

        Args:
            slots: (B, K, slot_dim) on device
        Returns:
            preds: (B,) int64 tensor (predicted class ids)
        """
        B, K, _ = slots.shape
        slots_flat = slots.view(B * K, self.slot_dim)
        sigma_flat = self._route_slots(slots_flat)
        sigma_2d = [sigma_flat[b * K:(b + 1) * K] for b in range(B)]

        h = torch.zeros(B, K, self.d_h, device=self.device)
        for b in range(B):
            for k in range(K):
                aid = sigma_2d[b][k]
                if aid != "unassigned" and aid in self.agents:
                    h[b, k] = self.agents[aid](
                        slots[b, k].unsqueeze(0)
                    ).squeeze(0)

        H = self.aggregator(h, sigma_2d)   # (B, d_h)
        H_np = H.cpu().numpy()

        preds = []
        for i in range(B):
            pred = self.slda.predict_one(H_np[i])
            preds.append(pred if pred is not None else -1)

        return torch.tensor(preds, dtype=torch.long)
