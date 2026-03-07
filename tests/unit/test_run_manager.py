"""Unit tests for run manager and schemas."""

import json
import tempfile

from prcl.integritysuite.run_manager import create_run_dir, save_metrics
from prcl.integritysuite.schemas import EpochMetrics, ForensicMetrics, RunMetrics


class TestRunManager:
    def test_creates_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = create_run_dir(tmp, "test_run", seed=42)
            assert run_path.exists()
            assert (run_path / "hardware.json").exists()
            assert (run_path / "metadata.json").exists()
            assert (run_path / "metrics.json").exists()
            assert (run_path / "env.txt").exists()
            assert (run_path / "checkpoints").is_dir()
            assert (run_path / "cards").is_dir()
            assert (run_path / "logs").is_dir()

    def test_metadata_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = create_run_dir(tmp, "test_meta", seed=123)
            with open(run_path / "metadata.json") as f:
                meta = json.load(f)
            assert meta["seed"] == 123
            assert "git_hash" in meta
            assert "timestamp" in meta

    def test_save_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_path = create_run_dir(tmp, "test_metrics")
            metrics = {"train": [{"epoch": 1, "loss": 5.0}], "eval": {"acc": 0.9}}
            save_metrics(run_path, metrics)
            with open(run_path / "metrics.json") as f:
                loaded = json.load(f)
            assert loaded["eval"]["acc"] == 0.9


class TestSchemas:
    def test_run_metrics_defaults(self):
        rm = RunMetrics(dataset="cifar10", backbone="resnet18")
        assert rm.ssl_method == "simclr"
        assert rm.defense_mode == "none"
        assert rm.poison_ratio == 0.0
        assert rm.linear_probe_acc is None

    def test_run_metrics_full(self):
        rm = RunMetrics(
            dataset="cifar10",
            backbone="resnet18",
            defense_mode="prcl",
            attack_family="patch_backdoor",
            poison_ratio=0.01,
            linear_probe_acc=0.89,
            asr=0.15,
        )
        assert rm.asr == 0.15
        data = rm.model_dump()
        assert data["defense_mode"] == "prcl"

    def test_epoch_metrics(self):
        em = EpochMetrics(
            epoch=1, train_loss=5.2, lr=0.001, epoch_time=12.5,
            mean_suspicion=0.3, quarantine_count=5,
        )
        assert em.mean_suspicion == 0.3

    def test_forensic_metrics(self):
        fm = ForensicMetrics(roc_auc=0.85, pr_auc=0.72, top_k_recall={"10": 0.9, "50": 0.95})
        assert fm.roc_auc == 0.85
        assert fm.top_k_recall["10"] == 0.9
