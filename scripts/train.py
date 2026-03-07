"""Main training entry point — Hydra-driven SimCLR / PRCL training."""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prcl.datasets.cifar import build_eval_dataloader, build_ssl_dataloader
from prcl.eval.linear_probe import train_linear_probe
from prcl.integritysuite.run_manager import create_run_dir, save_metrics
from prcl.ssl.methods.simclr_transforms import get_simclr_transform
from prcl.ssl.train_loop import train_simclr

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset_name: str) -> int:
    return {"cifar10": 10, "cifar100": 100, "stl10": 10}.get(dataset_name, 10)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create run directory
    defense_tag = cfg.defense.name if cfg.defense.enabled else "none"
    attack_tag = cfg.attack.name if cfg.attack.enabled else "clean"
    run_name = f"{cfg.ssl.method}_{cfg.dataset.name}_{cfg.model.backbone}_{defense_tag}_{attack_tag}"
    run_path = create_run_dir(
        base_dir=cfg.run_dir, run_name=run_name, config=cfg, seed=cfg.seed
    )
    logger.info(f"Run directory: {run_path}")

    # --- Attack setup (gated, disabled by default) ---
    from prcl.attacks.builder import build_attack

    attack_adapter = build_attack(cfg)
    poison_indices = None
    poison_fn = None
    if attack_adapter is not None:
        # Determine dataset size for index selection
        from prcl.datasets.cifar import get_cifar10, get_cifar100

        if cfg.dataset.name == "cifar10":
            _base = get_cifar10(cfg.dataset.data_dir, train=True)
        elif cfg.dataset.name == "cifar100":
            _base = get_cifar100(cfg.dataset.data_dir, train=True)
        elif cfg.dataset.name == "stl10":
            from prcl.datasets.stl10 import get_stl10
            _base = get_stl10(cfg.dataset.data_dir, split="train+unlabeled")
        else:
            raise ValueError(f"Unsupported dataset for attack: {cfg.dataset.name}")
        ds_size = cfg.dataset.subset_size or len(_base)
        poison_indices = attack_adapter.select_poison_indices(
            ds_size, cfg.attack.poison_ratio, seed=cfg.seed
        )
        poison_fn = attack_adapter.apply_trigger
        logger.info(
            f"Attack '{cfg.attack.name}' active: {len(poison_indices)} poisoned samples "
            f"({cfg.attack.poison_ratio*100:.1f}%)"
        )
        # Save attack metadata
        import json as _json

        with open(run_path / "attack_metadata.json", "w") as f:
            _json.dump(attack_adapter.get_metadata(), f, indent=2)

    # Build SSL transforms and dataloader
    ssl_transform = get_simclr_transform(cfg.dataset.name, train=True)
    train_loader, train_dataset = build_ssl_dataloader(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        transform=ssl_transform,
        batch_size=cfg.ssl.batch_size,
        num_workers=cfg.dataset.num_workers,
        subset_size=cfg.dataset.subset_size,
        poison_indices=poison_indices,
        poison_fn=poison_fn,
    )
    logger.info(f"Training data: {len(train_dataset)} samples, {len(train_loader)} batches")

    # --- PRCL defense setup (Phase 3 integration point) ---
    prcl_defense = None
    if cfg.defense.enabled and cfg.defense.name == "prcl":
        from prcl.defenses.prcl.defense import build_prcl_defense
        prcl_defense = build_prcl_defense(cfg, device, dataset_size=len(train_dataset))
        logger.info("PRCL defense enabled")

    # Train
    model = train_simclr(cfg, train_loader, run_path, device, prcl_defense=prcl_defense)

    # Linear probe evaluation
    logger.info("Running linear probe evaluation...")
    eval_transform = get_simclr_transform(cfg.dataset.name, train=False)
    num_classes = get_num_classes(cfg.dataset.name)

    eval_train_loader = build_eval_dataloader(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        transform=eval_transform,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.dataset.num_workers,
        train=True,
    )
    eval_test_loader = build_eval_dataloader(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        transform=eval_transform,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.dataset.num_workers,
        train=False,
    )

    probe_results = train_linear_probe(
        encoder=model.backbone,
        feat_dim=model.feat_dim,
        train_loader=eval_train_loader,
        test_loader=eval_test_loader,
        num_classes=num_classes,
        epochs=cfg.eval.epochs,
        lr=cfg.eval.lr,
        weight_decay=cfg.eval.weight_decay,
        device=device,
    )

    # Save final metrics
    import json
    metrics_path = run_path / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["eval"] = probe_results
    save_metrics(run_path, metrics)

    logger.info(f"Done! Linear probe accuracy: {probe_results['linear_probe_acc']:.4f}")
    logger.info(f"All artifacts saved to: {run_path}")


if __name__ == "__main__":
    main()
