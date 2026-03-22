"""SimCLR training loop with PRCL defense integration hooks."""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from prcl.ssl.backbones.resnet import get_backbone
from prcl.ssl.heads.projection import ProjectionHead
from prcl.ssl.losses.infonce import InfoNCELoss

logger = logging.getLogger(__name__)


class SimCLRModel(nn.Module):
    """SimCLR encoder + projection head."""

    def __init__(self, backbone_name: str = "resnet18", projection_dim: int = 128,
                 projection_hidden_dim: int = 2048, pretrained: bool = False):
        super().__init__()
        self.backbone, self.feat_dim = get_backbone(backbone_name, pretrained=pretrained)
        self.projection_head = ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (representations h, projections z)."""
        h = self.backbone(x)
        z = self.projection_head(h)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only backbone representations (for linear probe)."""
        return self.backbone(x)


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    if cfg.ssl.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.ssl.lr, weight_decay=cfg.ssl.weight_decay
        )
    elif cfg.ssl.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.ssl.lr, weight_decay=cfg.ssl.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.ssl.optimizer}")


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    warmup_steps = cfg.ssl.warmup_epochs * steps_per_epoch
    total_steps = cfg.ssl.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_simclr(cfg, train_loader, run_path: Path, device: torch.device,
                 prcl_defense=None) -> SimCLRModel:
    """Main SimCLR training loop.

    Args:
        cfg: Resolved Hydra config (OmegaConf DictConfig).
        train_loader: DataLoader yielding ((view1, view2), indices).
        run_path: Path to the run directory for saving artifacts.
        device: Torch device.
        prcl_defense: Optional PRCL defense module (Phase 3).

    Returns:
        Trained SimCLRModel.
    """
    model = SimCLRModel(
        backbone_name=cfg.model.backbone,
        projection_dim=cfg.model.projection_dim,
        projection_hidden_dim=cfg.model.projection_hidden_dim,
        pretrained=cfg.model.pretrained,
    ).to(device)

    criterion = InfoNCELoss(temperature=cfg.ssl.temperature)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    use_amp = cfg.ssl.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch_metrics = []
    best_loss = float("inf")

    logger.info(
        f"Starting SimCLR training: {cfg.ssl.epochs} epochs, "
        f"batch_size={cfg.ssl.batch_size}, device={device}"
    )

    global_step = 0
    for epoch in range(cfg.ssl.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        # --- PRCL HOOK: epoch start ---
        prcl_epoch_stats = {}
        if prcl_defense is not None:
            prcl_defense.on_epoch_start(epoch)

        for _batch_idx, ((view1, view2), indices) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg.ssl.epochs}", leave=False
        )):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                # Forward pass: encode both views
                x = torch.cat([view1, view2], dim=0)
                h, z = model(x)

                # --- PRCL HOOK: compute suspicion scores & modified loss ---
                if prcl_defense is not None and cfg.defense.enabled:
                    loss, prcl_batch_stats = prcl_defense.compute_defended_loss(
                        h=h, z=z, indices=indices, view1=view1, view2=view2,
                        model=model, criterion=criterion,
                    )
                    for k, v in prcl_batch_stats.items():
                        prcl_epoch_stats.setdefault(k, []).append(v)
                else:
                    loss = criterion(z)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)

        # --- PRCL HOOK: epoch end ---
        if prcl_defense is not None:
            prcl_defense.on_epoch_end(epoch)

        # Build epoch metric entry
        metric_entry = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "epoch_time": round(epoch_time, 2),
        }

        # Add averaged PRCL stats for the epoch
        if prcl_epoch_stats:
            for k, vals in prcl_epoch_stats.items():
                metric_entry[k] = round(sum(vals) / len(vals), 6)

        epoch_metrics.append(metric_entry)

        # Logging
        if (epoch + 1) % max(1, cfg.logging.log_every_n_steps // len(train_loader)) == 1 or \
                epoch == 0 or epoch == cfg.ssl.epochs - 1:
            logger.info(
                f"  Epoch {epoch+1}/{cfg.ssl.epochs}  loss={avg_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}  time={epoch_time:.1f}s"
            )

        # Checkpoint saving
        if (epoch + 1) % cfg.logging.save_checkpoint_every == 0 or epoch == cfg.ssl.epochs - 1:
            ckpt_path = run_path / "checkpoints" / f"epoch_{epoch+1}.ckpt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": dict(cfg) if hasattr(cfg, "__iter__") else str(cfg),
            }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = run_path / "checkpoints" / "best.ckpt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
            }, best_path)

    # Save final metrics
    all_metrics = {
        "train": epoch_metrics,
        "eval": {},
        "summary": {
            "final_loss": round(avg_loss, 6),
            "best_loss": round(best_loss, 6),
            "epochs_completed": cfg.ssl.epochs,
            "total_train_time": round(sum(m["epoch_time"] for m in epoch_metrics), 2),
            "avg_epoch_time": round(
                sum(m["epoch_time"] for m in epoch_metrics) / len(epoch_metrics), 2
            ),
        },
    }
    with open(run_path / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save final checkpoint
    torch.save({
        "epoch": cfg.ssl.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, run_path / "checkpoints" / "last.ckpt")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    return model
