"""Tests for attack adapters, safety gating, and builder."""


import numpy as np
import pytest
from PIL import Image

from prcl.attacks.adapters.blend_backdoor import BlendBackdoorAdapter
from prcl.attacks.adapters.patch_backdoor import PatchBackdoorAdapter
from prcl.attacks.safety import is_attack_allowed, require_attack_gate

# ---------- Safety gate ----------


class TestSafetyGate:
    def test_disabled_config_blocks(self, monkeypatch):
        monkeypatch.setenv("PRCL_ALLOW_ATTACKS", "1")
        assert not is_attack_allowed(cfg_enabled=False)

    def test_missing_env_blocks(self, monkeypatch):
        monkeypatch.delenv("PRCL_ALLOW_ATTACKS", raising=False)
        assert not is_attack_allowed(cfg_enabled=True)

    def test_both_required(self, monkeypatch):
        monkeypatch.setenv("PRCL_ALLOW_ATTACKS", "1")
        assert is_attack_allowed(cfg_enabled=True)

    def test_require_gate_raises_no_env(self, monkeypatch):
        monkeypatch.delenv("PRCL_ALLOW_ATTACKS", raising=False)
        with pytest.raises(RuntimeError, match="gated"):
            require_attack_gate(cfg_enabled=True)

    def test_require_gate_raises_disabled(self, monkeypatch):
        monkeypatch.setenv("PRCL_ALLOW_ATTACKS", "1")
        with pytest.raises(RuntimeError, match="not enabled"):
            require_attack_gate(cfg_enabled=False)


# ---------- Patch backdoor ----------


class TestPatchBackdoor:
    @pytest.fixture
    def adapter(self):
        return PatchBackdoorAdapter(patch_size=4, patch_position="bottom_right")

    def test_select_indices_count(self, adapter):
        indices = adapter.select_poison_indices(1000, poison_ratio=0.05)
        assert len(indices) == 50
        assert indices.dtype == np.intp or np.issubdtype(indices.dtype, np.integer)

    def test_select_indices_sorted(self, adapter):
        indices = adapter.select_poison_indices(1000, poison_ratio=0.01)
        assert np.all(np.diff(indices) > 0)

    def test_select_indices_deterministic(self, adapter):
        a = adapter.select_poison_indices(1000, 0.01, seed=42)
        b = adapter.select_poison_indices(1000, 0.01, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_apply_trigger_modifies_corner(self, adapter):
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        triggered = adapter.apply_trigger(img)
        pixels = np.array(triggered)
        # Bottom-right 4x4 should be white
        patch = pixels[28:32, 28:32]
        assert np.all(patch == 255)

    def test_apply_trigger_preserves_other_pixels(self, adapter):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        triggered = adapter.apply_trigger(img)
        pixels = np.array(triggered)
        # Top-left corner should be unchanged
        assert np.all(pixels[0:4, 0:4] == 128)

    def test_different_positions(self):
        for pos in ["top_left", "top_right", "bottom_left", "center"]:
            adapter = PatchBackdoorAdapter(patch_size=4, patch_position=pos)
            img = Image.new("RGB", (32, 32), (0, 0, 0))
            triggered = adapter.apply_trigger(img)
            assert triggered.size == (32, 32)

    def test_metadata(self, adapter):
        meta = adapter.get_metadata()
        assert meta["attack_type"] == "patch_backdoor"
        assert meta["patch_size"] == 4


# ---------- Blend backdoor ----------


class TestBlendBackdoor:
    @pytest.fixture
    def adapter(self):
        return BlendBackdoorAdapter(blend_alpha=0.1, target_class=0)

    def test_apply_trigger_preserves_shape(self, adapter):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        triggered = adapter.apply_trigger(img)
        assert triggered.size == (32, 32)

    def test_apply_trigger_modifies_image(self, adapter):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        triggered = adapter.apply_trigger(img)
        original = np.array(img)
        modified = np.array(triggered)
        # Should be different due to blend
        assert not np.array_equal(original, modified)

    def test_blend_deterministic(self, adapter):
        img = Image.new("RGB", (32, 32), (100, 100, 100))
        a = np.array(adapter.apply_trigger(img))
        b = np.array(adapter.apply_trigger(img))
        np.testing.assert_array_equal(a, b)

    def test_metadata(self, adapter):
        meta = adapter.get_metadata()
        assert meta["attack_type"] == "blend_backdoor"
        assert meta["blend_alpha"] == 0.1

    def test_select_indices(self, adapter):
        indices = adapter.select_poison_indices(500, 0.02)
        assert len(indices) == 10
