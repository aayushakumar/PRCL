"""Quick sanity check — verifies the pipeline works on a tiny subset."""

import logging
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== PRCL Sanity Check ===")

    # 1. Test imports
    logger.info("Testing imports...")
    from prcl.integritysuite.run_manager import create_run_dir
    from prcl.integritysuite.schemas import EpochMetrics, RunMetrics
    from prcl.ssl.backbones.resnet import get_backbone
    from prcl.ssl.heads.projection import ProjectionHead
    from prcl.ssl.losses.infonce import InfoNCELoss
    from prcl.ssl.train_loop import SimCLRModel
    logger.info("  All imports OK")

    # 2. Test backbone
    logger.info("Testing backbone...")
    backbone, feat_dim = get_backbone("resnet18")
    assert feat_dim == 512
    x = torch.randn(2, 3, 32, 32)
    h = backbone(x)
    assert h.shape == (2, 512), f"Expected (2, 512), got {h.shape}"
    logger.info(f"  ResNet-18: input (2,3,32,32) -> features {h.shape}")

    # 3. Test projection head
    logger.info("Testing projection head...")
    proj = ProjectionHead(512, 2048, 128)
    z = proj(h)
    assert z.shape == (2, 128)
    logger.info(f"  Projection: (2,512) -> {z.shape}")

    # 4. Test InfoNCE loss
    logger.info("Testing InfoNCE loss...")
    criterion = InfoNCELoss(temperature=0.5)
    z_batch = torch.randn(8, 128)  # 4 pairs
    loss = criterion(z_batch)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
    logger.info(f"  InfoNCE loss: {loss.item():.4f}")

    # 5. Test SimCLR model
    logger.info("Testing SimCLR model...")
    model = SimCLRModel("resnet18", projection_dim=128, projection_hidden_dim=2048)
    x = torch.randn(4, 3, 32, 32)
    h, z = model(x)
    assert h.shape == (4, 512)
    assert z.shape == (4, 128)
    logger.info(f"  SimCLR: input (4,3,32,32) -> h={h.shape}, z={z.shape}")

    # 6. Test run manager
    logger.info("Testing run manager...")
    with tempfile.TemporaryDirectory() as tmp:
        run_path = create_run_dir(tmp, "sanity_test", seed=42)
        assert (run_path / "hardware.json").exists()
        assert (run_path / "metadata.json").exists()
        assert (run_path / "metrics.json").exists()
        assert (run_path / "checkpoints").is_dir()
        logger.info(f"  Run dir created: {run_path}")

    # 7. Test schemas
    logger.info("Testing schemas...")
    rm = RunMetrics(dataset="cifar10", backbone="resnet18", linear_probe_acc=0.85)
    assert rm.dataset == "cifar10"
    em = EpochMetrics(epoch=1, train_loss=5.2, lr=0.001, epoch_time=12.5)
    assert em.epoch == 1
    logger.info("  Schemas OK")

    logger.info("=== All sanity checks passed! ===")


if __name__ == "__main__":
    main()
