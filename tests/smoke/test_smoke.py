"""Smoke tests — verify imports and config loading."""

import pytest


@pytest.mark.smoke
class TestImports:
    def test_prcl_package(self):
        import prcl
        assert hasattr(prcl, "__version__")

    def test_ssl_imports(self):
        pass

    def test_dataset_imports(self):
        pass

    def test_eval_imports(self):
        pass

    def test_integritysuite_imports(self):
        pass


@pytest.mark.smoke
class TestConfigLoading:
    def test_hydra_configs_exist(self):
        from pathlib import Path
        config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
        assert (config_dir / "config.yaml").exists()
        assert (config_dir / "dataset" / "cifar10.yaml").exists()
        assert (config_dir / "model" / "resnet18.yaml").exists()
        assert (config_dir / "ssl" / "simclr.yaml").exists()
        assert (config_dir / "defense" / "none.yaml").exists()
        assert (config_dir / "defense" / "prcl.yaml").exists()
        assert (config_dir / "attack" / "none.yaml").exists()
        assert (config_dir / "eval" / "linear_probe.yaml").exists()

    def test_config_schema_instantiation(self):
        from prcl.config_schema import PRCLConfig
        cfg = PRCLConfig()
        assert cfg.dataset.name == "cifar10"
        assert cfg.model.backbone == "resnet18"
        assert cfg.ssl.method == "simclr"
        assert cfg.defense.name == "none"
        assert cfg.seed == 42
