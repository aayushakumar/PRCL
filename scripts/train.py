"""Main training entry point — Hydra-driven SimCLR / PRCL training."""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prcl.datasets.cifar import build_eval_dataloader, build_ssl_dataloader, get_cifar10, get_cifar100
from prcl.eval.linear_probe import train_linear_probe
from prcl.integritysuite.run_manager import create_run_dir, save_metrics
from prcl.integritysuite.schemas import ForensicMetrics, RunMetrics
from prcl.integritysuite.cards import generate_run_card
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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

    # Save eval metrics
    import json
    metrics_path = run_path / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["eval"] = probe_results
    save_metrics(run_path, metrics)
    logger.info(f"Linear probe accuracy: {probe_results['linear_probe_acc']:.4f}")

    # --- ASR evaluation (when attack is active) ---
    asr_result = None
    if attack_adapter is not None:
        from prcl.eval.asr_eval import TriggeredDataset, evaluate_asr
        from prcl.eval.linear_probe import LinearProbe
        from torch.utils.data import DataLoader

        # Train a fresh linear probe for ASR measurement
        asr_probe = LinearProbe(model.backbone, model.feat_dim, num_classes).to(device)
        asr_probe_optim = torch.optim.SGD(asr_probe.classifier.parameters(), lr=cfg.eval.lr, momentum=0.9)
        asr_probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(asr_probe_optim, T_max=cfg.eval.epochs)
        criterion_ce = torch.nn.CrossEntropyLoss()
        asr_probe.train()
        for _ep in range(cfg.eval.epochs):
            for imgs, labels in eval_train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = asr_probe(imgs)
                loss = criterion_ce(logits, labels)
                asr_probe_optim.zero_grad()
                loss.backward()
                asr_probe_optim.step()
            asr_probe_sched.step()

        if cfg.dataset.name == "cifar10":
            base_test = get_cifar10(cfg.dataset.data_dir, train=False)
        elif cfg.dataset.name == "cifar100":
            base_test = get_cifar100(cfg.dataset.data_dir, train=False)
        else:
            from prcl.datasets.stl10 import get_stl10
            base_test = get_stl10(cfg.dataset.data_dir, split="test")

        triggered_ds = TriggeredDataset(
            base_dataset=base_test,
            trigger_fn=attack_adapter.apply_trigger,
            target_class=cfg.attack.target_class,
            transform=eval_transform,
        )
        triggered_loader = DataLoader(triggered_ds, batch_size=cfg.eval.batch_size, num_workers=cfg.dataset.num_workers)
        asr_result = evaluate_asr(model.backbone, asr_probe.classifier, triggered_loader, cfg.attack.target_class, device)
        metrics["asr"] = asr_result
        save_metrics(run_path, metrics)
        logger.info(f"ASR: {asr_result['asr']:.4f}")

    # --- Forensic evaluation (when defense + attack active) ---
    forensic_result = None
    if prcl_defense is not None and attack_adapter is not None:
        import numpy as np
        from prcl.eval.forensics import evaluate_forensics

        all_scores = []
        model.eval()
        with torch.no_grad():
            for (view1, _view2), indices in train_loader:
                view1 = view1.to(device)
                h = model.backbone(view1)
                scores = prcl_defense.pcf_scorer.compute_scores(view1, model.backbone, h)
                all_scores.append((indices.numpy(), scores.cpu().numpy()))
        idx_all = np.concatenate([s[0] for s in all_scores])
        score_all = np.concatenate([s[1] for s in all_scores])
        ordered = np.argsort(idx_all)
        score_all = score_all[ordered]
        poison_mask = train_dataset.poison_mask[: len(score_all)]
        forensic_result = evaluate_forensics(score_all, poison_mask)
        metrics["forensics"] = forensic_result.model_dump()
        save_metrics(run_path, metrics)
        logger.info(f"Forensic ROC-AUC: {forensic_result.roc_auc:.4f}")

    # --- Generate run card ---
    run_metrics = RunMetrics(
        dataset=cfg.dataset.name,
        backbone=cfg.model.backbone,
        ssl_method=cfg.ssl.method,
        defense_mode=cfg.defense.name if cfg.defense.enabled else "none",
        attack_family=cfg.attack.name if cfg.attack.enabled else "none",
        poison_ratio=cfg.attack.poison_ratio if cfg.attack.enabled else 0.0,
        seed=cfg.seed,
        linear_probe_acc=probe_results.get("linear_probe_acc"),
        asr=asr_result["asr"] if asr_result else None,
        final_train_loss=metrics.get("summary", {}).get("final_loss"),
        total_train_time=metrics.get("summary", {}).get("total_train_time"),
        train_time_per_epoch=metrics.get("summary", {}).get("avg_epoch_time"),
        epochs_completed=cfg.ssl.epochs,
    )
    generate_run_card(
        run_path,
        run_metrics,
        forensic_metrics=forensic_result,
        attack_metadata=attack_adapter.get_metadata() if attack_adapter else None,
    )

    logger.info(f"Done! All artifacts saved to: {run_path}")


if __name__ == "__main__":
    main()
