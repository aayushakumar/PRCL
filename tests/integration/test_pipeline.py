"""Integration test: end-to-end pipeline with tiny synthetic data.

Validates the full flow: data → poisoning → SimCLR + PRCL → eval → reporting.
Uses tiny random data (no real dataset download needed).
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prcl.attacks.adapters.patch_backdoor import PatchBackdoorAdapter
from prcl.datasets.cifar import PoisonAwareDataset
from prcl.defenses.prcl.pcf import ProbeAlignmentScorer
from prcl.defenses.prcl.quarantine import QuarantineManager
from prcl.defenses.prcl.robust_loss import WeightedInfoNCELoss
from prcl.eval.forensics import evaluate_forensics
from prcl.integritysuite.cards import generate_run_card
from prcl.integritysuite.schemas import RunMetrics
from prcl.ssl.backbones.resnet import get_backbone
from prcl.ssl.heads.projection import ProjectionHead
from prcl.ssl.losses.infonce import InfoNCELoss


class FakePILDataset(Dataset):
    """Tiny fake dataset returning PIL images for poisoning tests."""

    def __init__(self, size=64):
        from PIL import Image

        self.size = size
        self.targets = [i % 10 for i in range(size)]
        self._images = []
        rng = np.random.RandomState(42)
        for _ in range(size):
            arr = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            self._images.append(Image.fromarray(arr))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._images[idx], self.targets[idx]


class TwoViewTransform:
    """Simple two-view transform for integration test."""

    def __call__(self, img):
        from torchvision import transforms

        t = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        return t(img), t(img)


class TestEndToEnd:
    """Test the entire PRCL pipeline end-to-end with synthetic data."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_full_pipeline(self, device, tmp_path):
        """Run complete pipeline: poison → train → score → evaluate → report."""
        # 1. Setup attack adapter
        adapter = PatchBackdoorAdapter(patch_size=4, patch_position="bottom_right")
        poison_indices = adapter.select_poison_indices(64, poison_ratio=0.1, seed=42)

        # 2. Build poisoned dataset
        base = FakePILDataset(64)
        transform = TwoViewTransform()
        dataset = PoisonAwareDataset(
            base_dataset=base,
            transform=transform,
            poison_indices=poison_indices,
            poison_fn=adapter.apply_trigger,
        )

        assert dataset.num_poisoned == len(poison_indices)
        assert len(dataset) == 64

        loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)

        # 3. Build model
        backbone, feat_dim = get_backbone("resnet18")
        proj_head = ProjectionHead(feat_dim, 256, 64)
        model = nn.Sequential()
        model.backbone = backbone
        model.proj = proj_head
        model.to(device)

        # 4. PRCL components
        scorer = ProbeAlignmentScorer(probe_types=["blur"], normalize="batch_minmax")
        loss_fn = WeightedInfoNCELoss(temperature=0.5, mode="soft_weight")
        quarantine = QuarantineManager(dataset_size=64, threshold=0.9)

        # 5. Training loop (2 micro-epochs)
        optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(proj_head.parameters()), lr=1e-3
        )

        for epoch in range(2):
            for (view1, view2), indices in loader:
                view1, view2 = view1.to(device), view2.to(device)

                x = torch.cat([view1, view2], dim=0)
                h = backbone(x)
                z = proj_head(h)

                # PCF scoring on view1
                with torch.no_grad():
                    scores = scorer.compute_scores(view1, backbone)

                # Robust loss
                loss, stats = loss_fn(z, suspicion_scores=scores)

                # Quarantine tracking
                quarantine.update_scores(indices, scores)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            quarantine.on_epoch_end(epoch)

        # 6. Forensic evaluation
        # Gather all suspicion scores
        all_scores = quarantine.ema_scores.numpy()
        forensic = evaluate_forensics(
            all_scores, dataset.poison_mask,
            top_k_fractions=[0.1, 0.2],
        )
        assert forensic.roc_auc is not None
        assert 0 <= forensic.roc_auc <= 1

        # 7. Generate run card
        run_metrics = RunMetrics(
            dataset="synthetic",
            backbone="resnet18",
            ssl_method="simclr",
            defense_mode="prcl",
            attack_family="patch_backdoor",
            poison_ratio=0.1,
            linear_probe_acc=0.70,
            asr=0.05,
            epochs_completed=2,
        )
        card_path = generate_run_card(
            tmp_path, run_metrics,
            forensic_metrics=forensic,
            attack_metadata=adapter.get_metadata(),
        )
        assert card_path.exists()
        content = card_path.read_text()
        assert "patch_backdoor" in content
        assert "ROC-AUC" in content

        # Verify JSON card is valid
        json_card = tmp_path / "cards" / "run_card.json"
        assert json_card.exists()
        data = json.loads(json_card.read_text())
        assert data["run_metrics"]["defense_mode"] == "prcl"
        assert data["forensic_metrics"]["roc_auc"] is not None

    def test_clean_pipeline_no_attack(self, device, tmp_path):
        """Verify pipeline works without any attack (clean baseline)."""
        base = FakePILDataset(32)
        transform = TwoViewTransform()
        dataset = PoisonAwareDataset(base_dataset=base, transform=transform)
        assert dataset.num_poisoned == 0

        loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)

        backbone, feat_dim = get_backbone("resnet18")
        proj_head = ProjectionHead(feat_dim, 256, 64)
        loss_fn = InfoNCELoss(temperature=0.5)

        for (view1, view2), _indices in loader:
            x = torch.cat([view1, view2], dim=0)
            h = backbone(x)
            z = proj_head(h)
            loss = loss_fn(z)
            assert loss.item() > 0
            break  # One batch is enough

    def test_blend_attack_integration(self, device):
        """Verify blend attack integrates with the pipeline."""
        from prcl.attacks.adapters.blend_backdoor import BlendBackdoorAdapter

        adapter = BlendBackdoorAdapter(blend_alpha=0.15)
        poison_idx = adapter.select_poison_indices(64, 0.1, seed=0)

        base = FakePILDataset(64)
        transform = TwoViewTransform()
        dataset = PoisonAwareDataset(
            base_dataset=base,
            transform=transform,
            poison_indices=poison_idx,
            poison_fn=adapter.apply_trigger,
        )

        loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
        for (view1, _view2), _indices in loader:
            assert view1.shape == (8, 3, 32, 32)
            break
