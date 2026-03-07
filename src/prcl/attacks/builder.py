"""Build attack adapters from config with safety gating."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .safety import require_attack_gate

if TYPE_CHECKING:
    from .adapters.base import AttackAdapter


def build_attack(cfg) -> AttackAdapter | None:
    """Build an attack adapter from Hydra config.

    Returns None if attacks are disabled. Raises RuntimeError if
    attack is enabled but safety gate is not satisfied.
    """
    if cfg.attack.name == "none" or not cfg.attack.enabled:
        return None

    # Enforce dual gate
    require_attack_gate(cfg.attack.enabled)

    if cfg.attack.name == "patch_backdoor":
        from .adapters.patch_backdoor import PatchBackdoorAdapter

        return PatchBackdoorAdapter(
            patch_size=cfg.attack.patch_size,
            patch_position=cfg.attack.patch_position,
            target_class=cfg.attack.target_class,
        )
    elif cfg.attack.name == "blend_backdoor":
        from .adapters.blend_backdoor import BlendBackdoorAdapter

        return BlendBackdoorAdapter(
            blend_alpha=cfg.attack.blend_alpha,
            target_class=cfg.attack.target_class,
        )
    else:
        raise ValueError(f"Unknown attack: {cfg.attack.name}")
