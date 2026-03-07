"""Tests for IntegritySuite reporting, forensics, and cards."""

import json
from pathlib import Path

import numpy as np
import pytest

from prcl.eval.forensics import evaluate_forensics
from prcl.integritysuite.aggregate import collect_run_cards, make_main_table
from prcl.integritysuite.cards import generate_run_card
from prcl.integritysuite.schemas import ForensicMetrics, RunMetrics

# ---------- Forensic evaluation ----------


class TestForensicEvaluation:
    def test_perfect_separation(self):
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        mask = np.array([False, False, False, True, True])
        result = evaluate_forensics(scores, mask)
        assert result.roc_auc == 1.0
        assert result.pr_auc == 1.0
        assert result.score_separation > 0

    def test_random_scores(self):
        rng = np.random.RandomState(42)
        scores = rng.rand(100)
        mask = np.zeros(100, dtype=bool)
        mask[:10] = True
        result = evaluate_forensics(scores, mask)
        assert 0 <= result.roc_auc <= 1
        assert 0 <= result.pr_auc <= 1

    def test_top_k_recall_keys(self):
        scores = np.linspace(0, 1, 100)
        mask = np.zeros(100, dtype=bool)
        mask[90:] = True  # top 10 are actually poisoned
        result = evaluate_forensics(scores, mask, top_k_fractions=[0.1, 0.2])
        assert "top_0.10" in result.top_k_recall
        assert "top_0.20" in result.top_k_recall

    def test_no_poison_returns_empty(self):
        scores = np.random.rand(50)
        mask = np.zeros(50, dtype=bool)
        result = evaluate_forensics(scores, mask)
        assert result.roc_auc is None

    def test_all_poison_returns_empty(self):
        scores = np.random.rand(50)
        mask = np.ones(50, dtype=bool)
        result = evaluate_forensics(scores, mask)
        assert result.roc_auc is None

    def test_mean_scores(self):
        scores = np.array([0.2, 0.3, 0.8, 0.9])
        mask = np.array([False, False, True, True])
        result = evaluate_forensics(scores, mask)
        assert result.mean_clean_score == pytest.approx(0.25, abs=0.01)
        assert result.mean_poison_score == pytest.approx(0.85, abs=0.01)


# ---------- Run cards ----------


class TestRunCards:
    def test_generate_card_creates_files(self, tmp_path):
        run_path = tmp_path / "test_run"
        run_path.mkdir()

        metrics = RunMetrics(
            dataset="cifar10",
            backbone="resnet18",
            defense_mode="prcl",
            linear_probe_acc=0.85,
        )
        card_path = generate_run_card(run_path, metrics)
        assert card_path.exists()
        content = card_path.read_text()
        assert "cifar10" in content
        assert "resnet18" in content
        assert "0.8500" in content

    def test_card_with_forensics(self, tmp_path):
        run_path = tmp_path / "test_run"
        run_path.mkdir()

        metrics = RunMetrics(dataset="cifar10", backbone="resnet18")
        forensics = ForensicMetrics(roc_auc=0.95, pr_auc=0.88)
        card_path = generate_run_card(run_path, metrics, forensic_metrics=forensics)
        content = card_path.read_text()
        assert "0.9500" in content
        assert "ROC-AUC" in content

    def test_card_json_output(self, tmp_path):
        run_path = tmp_path / "test_run"
        run_path.mkdir()

        metrics = RunMetrics(dataset="cifar10", backbone="resnet18", asr=0.05)
        card_path = generate_run_card(run_path, metrics)
        json_path = card_path.parent / "run_card.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["run_metrics"]["asr"] == 0.05


# ---------- Aggregation ----------


class TestAggregation:
    def _create_run(self, base_dir: Path, name: str, metrics: RunMetrics):
        run_dir = base_dir / name
        cards_dir = run_dir / "cards"
        cards_dir.mkdir(parents=True)
        card_data = {
            "run_metrics": metrics.model_dump(),
            "forensic_metrics": None,
            "attack_metadata": None,
        }
        (cards_dir / "run_card.json").write_text(json.dumps(card_data))
        return run_dir

    def test_collect_finds_runs(self, tmp_path):
        self._create_run(
            tmp_path, "run1",
            RunMetrics(dataset="cifar10", backbone="resnet18", linear_probe_acc=0.8),
        )
        self._create_run(
            tmp_path, "run2",
            RunMetrics(dataset="cifar10", backbone="resnet18", defense_mode="prcl", linear_probe_acc=0.82),
        )
        df = collect_run_cards(tmp_path)
        assert len(df) == 2

    def test_main_table_format(self, tmp_path):
        self._create_run(
            tmp_path, "run1",
            RunMetrics(dataset="cifar10", backbone="resnet18", linear_probe_acc=0.85, asr=0.3),
        )
        df = collect_run_cards(tmp_path)
        table = make_main_table(df)
        assert "Dataset" in table.columns
        assert table["Asr"].iloc[0] == "30.0"

    def test_empty_dir(self, tmp_path):
        df = collect_run_cards(tmp_path)
        assert df.empty
