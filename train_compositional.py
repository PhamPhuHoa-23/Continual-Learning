import argparse
import os
import json
import random
import logging
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm

from src.models.adaslot.model import AdaSlotModel
from src.compositional.pipeline import CompositionalRoutingPipeline
from src.data.continual_cifar100 import get_continual_cifar100_loaders
from src.losses.contrastive import SlotClusteringLoss
from src.losses.primitive import PrimitiveSelector, ConceptLearningLoss


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logger(run_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    # Avoid adding duplicate handlers on re-import
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─── Args ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Compositional Routing Pipeline with AdaSlot"
    )

    # Phases
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["adaslot", "task1", "task_t", "all"],
        help="Which phase to run"
    )

    # Model
    parser.add_argument("--slot_dim",    type=int,   default=64)
    parser.add_argument("--num_slots",   type=int,   default=7,
                        help="Max number of slots for AdaSlot")
    parser.add_argument("--d_h",         type=int,   default=128,
                        help="Agent output (hidden label) dimension")
    parser.add_argument("--latent_dim",  type=int,   default=32,
                        help="VAE latent dimension")

    # Continual Learning
    parser.add_argument("--n_tasks",            type=int,   default=5)
    parser.add_argument("--n_classes_per_task", type=int,   default=20,
                        help="Number of new classes per task (CIFAR-100 / n_tasks)")
    parser.add_argument("--M0",                 type=int,   default=10,
                        help="Max agents (backwards compat; HDBSCAN auto-discovers actual count)")
    parser.add_argument("--min_cluster_size",   type=int,   default=10,
                        help="HDBSCAN: minimum cluster size")
    parser.add_argument("--min_samples_cluster", type=int,  default=5,
                        help="HDBSCAN: minimum samples for core points")
    parser.add_argument("--theta_match",        type=float, default=-50.0,
                        help="Mahalanobis score threshold: above => match existing agent")
    parser.add_argument("--theta_novel",        type=float, default=-200.0,
                        help="Kept for API compat; slots below theta_match are unassigned")
    parser.add_argument("--b_min",              type=int,   default=50,
                        help="Min buffer size before clustering for new agent spawn")
    parser.add_argument("--rho_min",            type=float, default=0.7,
                        help="Min intra-cluster cosine similarity for agent spawn")
    parser.add_argument("--n_min",              type=int,   default=10,
                        help="Min cluster size for agent spawn")

    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0,  help="Weight for L_p")
    parser.add_argument("--beta",  type=float, default=0.5,  help="Weight for L_SupCon")
    parser.add_argument("--gamma", type=float, default=0.5,  help="Weight for L_agent (recon)")
    parser.add_argument("--delta", type=float, default=0.1,  help="Weight for L_local (neighbor)")
    
    # Primitive loss for AdaSlot (CompSLOT Paper Approach)
    parser.add_argument("--use_primitive_loss", action="store_true",
                        help="Use primitive loss during AdaSlot training (CompSLOT paper)")
    parser.add_argument("--primitive_alpha", type=float, default=10.0,
                        help="Weight for primitive loss (α in L_slot = L_re + α*L_p)")
    parser.add_argument("--primitive_temp", type=float, default=10.0,
                        help="Temperature for primitive similarity (τ_p)")
    
    # Contrastive/Clustering loss for AdaSlot (Legacy)
    parser.add_argument("--use_clustering_loss", action="store_true",
                        help="Add contrastive/clustering loss during AdaSlot training")
    parser.add_argument("--clustering_loss_type", type=str, default="contrastive",
                        choices=["contrastive", "prototype"],
                        help="Type of clustering loss: contrastive (SupCon) or prototype")
    parser.add_argument("--clustering_weight", type=float, default=0.5,
                        help="Weight for clustering loss")
    parser.add_argument("--clustering_temp", type=float, default=0.07,
                        help="Temperature for contrastive/prototype loss")
    parser.add_argument("--slot_aggregation", type=str, default="mean",
                        choices=["mean", "max", "attention"],
                        help="How to aggregate multiple slots per image")

    # Training
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--adaslot_epochs", type=int,   default=50,
                        help="Epochs for AdaSlot warm-up phase")
    parser.add_argument("--vae_epochs",     type=int,   default=10,
                        help="Epochs for per-agent VAE training")
    parser.add_argument("--sparse_weight",  type=float, default=1.0,
                        help="Sparse penalty weight for AdaSlot training")

    # Paths
    parser.add_argument("--adaslot_ckpt",  type=str, default=None,
                        help="Path to pre-trained AdaSlot checkpoint (skips adaslot phase)")
    parser.add_argument("--pipeline_ckpt", type=str, default=None,
                        help="Path to pipeline checkpoint to resume from")
    parser.add_argument("--output_dir",    type=str, default="checkpoints/compositional_runs")

    parser.add_argument("--adaslot_resolution", type=int, default=128,
                        help="Resolution expected by AdaSlot (images resized to this)")

    # System
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--workers",    type=int, default=4)

    # ── Test / Debug mode ─────────────────────────────────────────────────────
    parser.add_argument("--test_mode",   action="store_true",
                        help="Limit dataset size for quick smoke-testing")
    parser.add_argument("--max_samples", type=int, default=320,
                        help="Max samples per task in test_mode")

    return parser.parse_args()


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Subset loader (test mode) ───────────────────────────────────────────────
def _make_cached_subset_loader(loader, max_samples: int, batch_size: int, workers: int):
    """Return a DataLoader with cached subset for fast iteration in test mode."""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Load and cache the first max_samples
    X_list, y_list = [], []
    total = 0
    for X, y in loader:
        X_list.append(X)
        y_list.append(y)
        total += X.size(0)
        if total >= max_samples:
            break
    
    if not X_list:
        # Empty loader - return dummy
        X_cached = torch.zeros((0, 3, 32, 32))
        y_cached = torch.zeros((0,), dtype=torch.long)
    else:
        X_cached = torch.cat(X_list, dim=0)[:max_samples]
        y_cached = torch.cat(y_list, dim=0)[:max_samples]
    
    # Create in-memory dataset
    cached_dataset = TensorDataset(X_cached, y_cached)
    return DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Already in memory, no need for workers
        pin_memory=False,
        drop_last=False,
    )
def _make_subset_loader(loader, max_samples: int, batch_size: int, workers: int):
    """Return a new DataLoader limited to max_samples items from loader.dataset."""
    from torch.utils.data import DataLoader, SubsetRandomSampler
    n = min(max_samples, len(loader.dataset))
    indices = list(range(n))
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices),
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
    )

# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def save_pipeline(pipeline: CompositionalRoutingPipeline, path: str) -> None:
    """Serialize pipeline state to disk."""
    state = {
        "agents":     {k: v.state_dict() for k, v in pipeline.agents.items()},
        "agent_keys": {k: v.detach().cpu() for k, v in pipeline.aggregator.keys.items()},
        "vaes":       {k: v.state_dict() for k, v in pipeline.router.vaes.items()},
        "vae_stats":  {k: {
                          "count":     st["count"],
                          "mu":        st["mu"].cpu(),
                          "M2":        st["M2"].cpu(),
                          "Sigma_inv": st["Sigma_inv"].cpu(),
                      } for k, st in pipeline.router.stats.items()},
        "slda_means":  {k: torch.tensor(v) for k, v in pipeline.slda.means.items()},
        "slda_S":      torch.tensor(pipeline.slda.S),
        "slda_counts": dict(pipeline.slda.counts),
        "slda_total":  pipeline.slda.total_count,
        "new_agent_ids": pipeline.new_agent_ids,
    }
    torch.save(state, path)


def load_pipeline(pipeline: CompositionalRoutingPipeline, path: str) -> None:
    """Load pipeline state from disk."""
    state = torch.load(path, map_location=pipeline.device)
    dev = pipeline.device

    # Agents
    for k, sd in state["agents"].items():
        if k not in pipeline.agents:
            pipeline._spawn_agent(k)
        pipeline.agents[k].load_state_dict(sd)

    # Aggregator keys
    for k, w in state["agent_keys"].items():
        if k not in pipeline.aggregator.keys:
            pipeline.aggregator.add_agent_key(k)
        pipeline.aggregator.keys[k].data.copy_(w.to(dev))

    # VAEs + stats
    for k, sd in state["vaes"].items():
        if k not in pipeline.router.vaes:
            pipeline.router.spawn_vae(k, dev)
        pipeline.router.vaes[k].load_state_dict(sd)
    for k, st in state["vae_stats"].items():
        pipeline.router.stats[k] = {
            "count":     st["count"],
            "mu":        st["mu"].to(dev),
            "M2":        st["M2"].to(dev),
            "Sigma_inv": st["Sigma_inv"].to(dev),
        }

    # SLDA
    pipeline.slda.means = {k: v.cpu().numpy() for k, v in state["slda_means"].items()}
    pipeline.slda.S           = state["slda_S"].cpu().numpy()
    pipeline.slda.counts      = state["slda_counts"]
    pipeline.slda.total_count = state["slda_total"]
    pipeline.slda.is_dirty    = True

    pipeline.new_agent_ids = state.get("new_agent_ids", [])


# ─── Slot encoding ───────────────────────────────────────────────────────────

def encode_slots(
    loader,
    adaslot: AdaSlotModel,
    device: str,
    fallback_num_slots: int,
    fallback_slot_dim: int,
    adaslot_resolution: int = 128,
):
    """
    Generator: yields (X_cpu, y, slots_cpu) one batch at a time.
    Images are resized to adaslot_resolution x adaslot_resolution before encoding.
    Uses AdaSlot if available, otherwise random fallback for debugging.
    """
    if adaslot is not None:
        adaslot.eval()
        # DON'T use torch.no_grad() - we need gradient flow through agents!
        # AdaSlot is frozen (requires_grad=False) so it won't compute gradients anyway
        for X, y in loader:
            X_dev = X.to(device)
            # Resize to AdaSlot's expected resolution if needed
            if X_dev.shape[-1] != adaslot_resolution:
                X_dev = F.interpolate(
                    X_dev, size=(adaslot_resolution, adaslot_resolution),
                    mode="bilinear", align_corners=False,
                )
            slots = adaslot.encode(X_dev)   # (B, K, slot_dim)
            # Detach from AdaSlot graph but keep tensor "gradient-ready"
            yield X.cpu(), y, slots.detach().cpu()
    else:
        for X, y in loader:
            B = X.size(0)
            slots = torch.randn(B, fallback_num_slots, fallback_slot_dim)
            yield X, y, slots


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    pipeline: CompositionalRoutingPipeline,
    test_loaders: list,
    adaslot: AdaSlotModel,
    seen_tasks: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> float:
    """Evaluate accuracy on all seen tasks."""
    all_correct = 0
    all_total   = 0

    for t in range(seen_tasks):
        correct = 0
        total   = 0
        for X, y, slots in encode_slots(
            test_loaders[t], adaslot, args.device, args.num_slots, args.slot_dim,
            args.adaslot_resolution,
        ):
            preds = pipeline.predict(slots.to(args.device))   # (B,)
            correct += (preds.cpu() == y).sum().item()
            total   += y.size(0)

        task_acc = 100.0 * correct / max(total, 1)
        logger.info(f"  Task {t+1} accuracy: {task_acc:.2f}%")
        all_correct += correct
        all_total   += total

    avg_acc = 100.0 * all_correct / max(all_total, 1)
    logger.info(f"  Average accuracy over {seen_tasks} tasks: {avg_acc:.2f}%")
    return avg_acc


# ─── Phase: AdaSlot ──────────────────────────────────────────────────────────

def phase_adaslot(
    args: argparse.Namespace,
    train_loaders: list,
    run_dir: str,
    logger: logging.Logger,
) -> AdaSlotModel:
    logger.info("=== Phase: AdaSlot Warm-up ===")

    model = AdaSlotModel(num_slots=args.num_slots, slot_dim=args.slot_dim)
    model.to(args.device)

    # ─── Primitive Selector (CompSLOT Paper Approach) ───
    primitive_selector = None
    concept_loss_fn = None
    
    if args.use_primitive_loss:
        # Create primitive selector (Equation 2 from paper)
        primitive_selector = PrimitiveSelector(
            slot_dim=args.slot_dim,
            temperature=None  # Uses paper default: 100/√D_s
        ).to(args.device)
        
        # Create concept learning loss (L_slot = L_re + α*L_p)
        concept_loss_fn = ConceptLearningLoss(
            alpha=args.primitive_alpha,  # Weight for primitive loss
            temperature_p=args.primitive_temp  # Temperature for primitive similarity
        ).to(args.device)
        
        # Optimize both model and primitive selector
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(primitive_selector.parameters()), 
            lr=args.lr
        )
        
        logger.info(f"  Using primitive loss (α={args.primitive_alpha}, "
                   f"τ_p={args.primitive_temp})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # ─── Optional: Legacy Clustering Loss ───
    clustering_loss = None
    if args.use_clustering_loss:
        clustering_loss = SlotClusteringLoss(
            loss_type=args.clustering_loss_type,
            temperature=args.clustering_temp,
            num_classes=args.n_classes_per_task,  # Task 1 classes
            embedding_dim=args.slot_dim,
            aggregation=args.slot_aggregation
        ).to(args.device)
        logger.info(f"  Using clustering loss: {args.clustering_loss_type} "
                   f"(weight={args.clustering_weight}, temp={args.clustering_temp}, "
                   f"aggregation={args.slot_aggregation})")

    # Train on Task 1 data only (Slot Attention must be stable before CIL)
    epoch_bar = tqdm(range(args.adaslot_epochs), desc="[AdaSlot] Epochs", unit="ep")
    for epoch in epoch_bar:
        model.train()
        if primitive_selector is not None:
            primitive_selector.train()
            
        total_loss = 0.0
        n_batches  = 0

        batch_bar = tqdm(train_loaders[0], desc=f"  Ep {epoch+1}/{args.adaslot_epochs}",
                         leave=False, unit="batch")
        for X, y in batch_bar:
            X = X.to(args.device)
            y = y.to(args.device)  # Need labels for primitive loss

            # Resize to AdaSlot's expected resolution
            res = args.adaslot_resolution
            if X.shape[-1] != res:
                X = F.interpolate(
                    X, size=(res, res),
                    mode="bilinear", align_corners=False,
                )

            # Forward through AdaSlot
            out = model(X, global_step=epoch)

            # ─── OPTION 1: Primitive Loss (Paper's Approach) ───
            if args.use_primitive_loss:
                # Extract primitives from slots
                primitives, weights = primitive_selector(out["slots"])
                
                # Compute concept learning loss: L_slot = L_re + α*L_p
                losses = concept_loss_fn(
                    reconstructed=out["reconstruction"],
                    target=X,
                    primitives=primitives,
                    labels=y
                )
                
                loss = losses['total']
                loss_recon = losses['recon']
                loss_prim = losses['primitive']
                
                # Sparse penalty: encourage dropping unnecessary slots
                loss_sparse = args.sparse_weight * out["hard_keep_decision"].float().mean()
                loss = loss + loss_sparse
                
                batch_bar.set_postfix(
                    recon=f"{loss_recon.item():.2f}",
                    prim=f"{loss_prim.item():.4f}",
                    sparse=f"{loss_sparse.item():.4f}"
                )
            
            # ─── OPTION 2: Legacy Reconstruction + Clustering ───
            else:
                # L_recon = MSE(reconstruction, image)
                loss_recon  = F.mse_loss(out["reconstruction"], X, reduction="sum") / X.size(0)
                # Sparse penalty: encourage dropping unnecessary slots
                loss_sparse = args.sparse_weight * out["hard_keep_decision"].float().mean()
                loss = loss_recon + loss_sparse
                
                # Optional: Clustering loss to make slots more discriminative
                if clustering_loss is not None:
                    loss_cluster = clustering_loss(
                        slots=out["slots"],
                        labels=y,
                        masks=out.get("masks", None)
                    )
                    loss = loss + args.clustering_weight * loss_cluster
                    batch_bar.set_postfix(
                        recon=f"{loss_recon.item():.2f}",
                        sparse=f"{loss_sparse.item():.4f}",
                        cluster=f"{loss_cluster.item():.4f}"
                    )
                else:
                    batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)  # Increased from 1.0
            if primitive_selector is not None:
                torch.nn.utils.clip_grad_norm_(primitive_selector.parameters(), 100.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        epoch_bar.set_postfix(avg_loss=f"{avg:.4f}")
        if (epoch + 1) % max(1, args.adaslot_epochs // 5) == 0 or epoch == 0:
            logger.info(
                f"  [AdaSlot] Epoch {epoch+1}/{args.adaslot_epochs} "
                f"| L_total: {avg:.4f}"
            )

    ckpt_path = os.path.join(run_dir, "adaslot_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"AdaSlot checkpoint saved to {ckpt_path}")

    # Freeze Slot Attention after training
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─── Phase: Task 1 ───────────────────────────────────────────────────────────

def phase_task1(
    args: argparse.Namespace,
    adaslot: AdaSlotModel,
    train_loaders: list,
    run_dir: str,
    logger: logging.Logger,
) -> CompositionalRoutingPipeline:
    logger.info("=== Phase: Task 1 ===")

    pipeline = CompositionalRoutingPipeline(
        slot_dim=args.slot_dim,
        d_h=args.d_h,
        latent_dim=args.latent_dim,
        theta_match=args.theta_match,
        theta_novel=args.theta_novel,
        b_min=args.b_min,
        rho_min=args.rho_min,
        n_min=args.n_min,
        loss_weights=dict(
            alpha=args.alpha, beta=args.beta,
            gamma=args.gamma, delta=args.delta,
        ),
        device=args.device,
    )

    # --- 1a: Cluster slots → spawn agents (HDBSCAN auto-discovers count) ---
    logger.info("  Clustering slots with HDBSCAN...")
    pipeline.init_agents_from_clustering(
        encode_slots(train_loaders[0], adaslot, args.device,
                     args.num_slots, args.slot_dim, args.adaslot_resolution),
        M0=args.M0,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples_cluster,
    )
    logger.info(f"  Spawned {len(pipeline.agents)} agents from HDBSCAN clustering.")

    # --- 1b: Train agents + aggregator keys ---
    optimizer = torch.optim.Adam(pipeline.trainable_params(), lr=args.lr)

    epoch_bar = tqdm(range(args.epochs), desc="[Task 1] Epochs", unit="ep")
    for epoch in epoch_bar:
        acc_losses = {"Lp": 0., "SupCon": 0., "agent": 0., "total": 0.}
        n_batches = 0

        pipeline.train_mode()
        batch_bar = tqdm(
            encode_slots(train_loaders[0], adaslot, args.device,
                         args.num_slots, args.slot_dim, args.adaslot_resolution),
            desc=f"  Ep {epoch+1}/{args.epochs}", leave=False, unit="batch",
        )
        for X, y, slots in batch_bar:
            slots = slots.to(args.device)
            y     = y.to(args.device)

            losses = pipeline.compute_losses(slots, y, task_id=1)
            loss   = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in acc_losses:
                if k in losses:
                    acc_losses[k] += losses[k].item()
            n_batches += 1
            batch_bar.set_postfix(total=f"{losses['total'].item():.4f}")

        avg_str = " | ".join(
            f"L_{k}: {v/max(n_batches,1):.4f}" for k, v in acc_losses.items()
        )
        epoch_bar.set_postfix(total=f"{acc_losses['total']/max(n_batches,1):.4f}")
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            logger.info(f"  [Task 1] Epoch {epoch+1}/{args.epochs} | {avg_str}")

    # --- 1c: Train VAEs (one per agent) → freeze → init Welford stats ---
    logger.info("  Training VAEs for Task 1 agents...")
    pipeline.train_vaes(
        encode_slots(train_loaders[0], adaslot, args.device,
                     args.num_slots, args.slot_dim, args.adaslot_resolution),
        task_id=1,
        vae_epochs=args.vae_epochs,
    )

    # --- 1d: Freeze everything, init SLDA ---
    pipeline.freeze_task(task_id=1)
    pipeline.update_slda(
        encode_slots(train_loaders[0], adaslot, args.device,
                     args.num_slots, args.slot_dim, args.adaslot_resolution),
        class_offset=0,
    )
    logger.info(f"  Task 1 done. Agents: {len(pipeline.agents)}")

    ckpt_path = os.path.join(run_dir, "pipeline_task1.pt")
    save_pipeline(pipeline, ckpt_path)
    logger.info(f"  Pipeline checkpoint saved to {ckpt_path}")
    return pipeline


# ─── Phase: Task t > 1 ───────────────────────────────────────────────────────

def phase_task_t(
    args: argparse.Namespace,
    pipeline: CompositionalRoutingPipeline,
    adaslot: AdaSlotModel,
    train_loaders: list,
    test_loaders: list,
    start_task: int,
    run_dir: str,
    logger: logging.Logger,
) -> CompositionalRoutingPipeline:

    for task_idx in range(start_task, args.n_tasks):
        logger.info(f"=== Phase: Task {task_idx + 1} ===")
        class_offset = task_idx * args.n_classes_per_task

        # --- Routing pass: detect novel sub-concepts ---
        logger.info("  Routing slots, collecting unassigned buffer...")
        pipeline.collect_novel_buffer(
            encode_slots(
                train_loaders[task_idx], adaslot,
                args.device, args.num_slots, args.slot_dim,
                args.adaslot_resolution,
            )
        )

        n_spawned = pipeline.spawn_new_agents(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples_cluster,
        )
        logger.info(
            f"  Spawned {n_spawned} new agents. "
            f"Total agents: {len(pipeline.agents)}"
        )

        # --- Train new agents + their keys ---
        new_params = pipeline.new_agent_params()
        if new_params:
            optimizer = torch.optim.Adam(new_params, lr=args.lr)

            epoch_bar = tqdm(
                range(args.epochs),
                desc=f"[Task {task_idx+1}] Epochs", unit="ep",
            )
            for epoch in epoch_bar:
                acc_losses = {"Lp": 0., "SupCon": 0., "agent": 0.,
                              "local": 0., "total": 0.}
                n_batches = 0

                pipeline.train_mode(new_only=True)
                batch_bar = tqdm(
                    encode_slots(
                        train_loaders[task_idx], adaslot,
                        args.device, args.num_slots, args.slot_dim,
                        args.adaslot_resolution,
                    ),
                    desc=f"  Ep {epoch+1}/{args.epochs}", leave=False, unit="batch",
                )
                for X, y, slots in batch_bar:
                    slots = slots.to(args.device)
                    y     = y.to(args.device)

                    losses = pipeline.compute_losses(slots, y, task_idx + 1)
                    loss   = losses["total"]

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    for k in acc_losses:
                        if k in losses:
                            acc_losses[k] += losses[k].item()
                    n_batches += 1
                    batch_bar.set_postfix(total=f"{losses['total'].item():.4f}")

                avg_str = " | ".join(
                    f"L_{k}: {v/max(n_batches,1):.4f}"
                    for k, v in acc_losses.items()
                )
                epoch_bar.set_postfix(
                    total=f"{acc_losses['total']/max(n_batches,1):.4f}"
                )
                if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
                    logger.info(
                        f"  [Task {task_idx+1}] Epoch {epoch+1}/{args.epochs} "
                        f"| {avg_str}"
                    )
        else:
            logger.info("  No new agents — all sub-concepts reused from previous tasks.")

        # --- Train VAEs for new agents; update latent stats for old agents ---
        logger.info("  Training VAEs for new agents & updating old agent stats...")
        pipeline.train_vaes(
            encode_slots(
                train_loaders[task_idx], adaslot,
                args.device, args.num_slots, args.slot_dim,
                args.adaslot_resolution,
            ),
            task_id=task_idx + 1,
            vae_epochs=args.vae_epochs,
        )

        # --- Freeze new agents, update SLDA (old class stats unchanged) ---
        pipeline.freeze_task(task_id=task_idx + 1)
        pipeline.update_slda(
            encode_slots(
                train_loaders[task_idx], adaslot,
                args.device, args.num_slots, args.slot_dim,
                args.adaslot_resolution,
            ),
            class_offset=class_offset,
        )

        # --- Evaluate ---
        logger.info(f"  Evaluating after Task {task_idx + 1}...")
        evaluate(pipeline, test_loaders, adaslot, task_idx + 1, args, logger)

        # --- Checkpoint ---
        ckpt_path = os.path.join(run_dir, f"pipeline_task{task_idx + 1}.pt")
        save_pipeline(pipeline, ckpt_path)
        logger.info(f"  Checkpoint saved to {ckpt_path}")

    return pipeline


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {args.device} | Seed: {args.seed}")

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Config saved to {config_path}")

    # Data
    logger.info("Loading CIFAR-100 loaders...")
    train_loaders, test_loaders, class_order = get_continual_cifar100_loaders(
        n_tasks=args.n_tasks,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
        max_samples_per_task=args.max_samples if args.test_mode else None,
    )
    logger.info(f"Class order: {class_order}")

    # ── Test mode: cache data in memory for fast iteration ─────────
    if args.test_mode:
        logger.info(
            f"[TEST MODE] Data already limited to {args.max_samples} samples. Caching in memory..."
        )
        import time
        start = time.time()
        train_loaders = [
            _make_cached_subset_loader(l, args.max_samples, args.batch_size, args.workers)
            for l in train_loaders
        ]
        test_loaders = [
            _make_cached_subset_loader(l, args.max_samples, args.batch_size, args.workers)
            for l in test_loaders
        ]
        elapsed = time.time() - start
        logger.info(f"[TEST MODE] Cached {args.n_tasks} tasks in {elapsed:.1f}s!")

    # ── AdaSlot phase ──────────────────────────────────────────────────────
    adaslot = None

    if args.adaslot_ckpt is not None:
        logger.info(f"Loading AdaSlot from {args.adaslot_ckpt}")
        adaslot = AdaSlotModel(num_slots=args.num_slots, slot_dim=args.slot_dim)
        ckpt = torch.load(args.adaslot_ckpt, map_location=args.device)
        # Support both raw state_dict and wrapped {'model': ...} checkpoints
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        adaslot.load_state_dict(state, strict=False)
        adaslot.to(args.device)
        for p in adaslot.parameters():
            p.requires_grad_(False)

    elif args.phase in ["adaslot", "all"]:
        adaslot = phase_adaslot(args, train_loaders, run_dir, logger)

    # ── Task 1 phase ───────────────────────────────────────────────────────
    pipeline = None

    if args.pipeline_ckpt is not None:
        logger.info(f"Loading pipeline from {args.pipeline_ckpt}")
        pipeline = CompositionalRoutingPipeline(
            slot_dim=args.slot_dim, d_h=args.d_h, latent_dim=args.latent_dim,
            theta_match=args.theta_match, theta_novel=args.theta_novel,
            b_min=args.b_min, rho_min=args.rho_min, n_min=args.n_min,
            loss_weights=dict(
                alpha=args.alpha, beta=args.beta,
                gamma=args.gamma, delta=args.delta,
            ),
            device=args.device,
        )
        load_pipeline(pipeline, args.pipeline_ckpt)
        logger.info(
            f"Resumed pipeline: {len(pipeline.agents)} agents, "
            f"{len(pipeline.slda.means)} classes in SLDA."
        )

    elif args.phase in ["task1", "all"]:
        pipeline = phase_task1(args, adaslot, train_loaders, run_dir, logger)

    # ── Task t > 1 phase ──────────────────────────────────────────────────
    if args.phase in ["task_t", "all"]:
        if pipeline is None:
            raise ValueError(
                "Pipeline is not initialized. "
                "Run with --phase all, or provide --pipeline_ckpt."
            )
        # Determine which task to start from (based on SLDA classes seen)
        n_classes_seen = len(pipeline.slda.means)
        start_task = n_classes_seen // args.n_classes_per_task
        if start_task == 0:
            start_task = 1   # Task 1 just finished, start from task 2
        if start_task >= args.n_tasks:
            logger.info("All tasks already trained. Nothing to do.")
        else:
            logger.info(f"Resuming from task {start_task + 1}.")
            pipeline = phase_task_t(
                args, pipeline, adaslot,
                train_loaders, test_loaders,
                start_task, run_dir, logger,
            )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()