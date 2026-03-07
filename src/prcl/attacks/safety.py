"""Safety gate for attack code paths.

Attack adapters are gated behind TWO requirements:
1. The attack config must have `enabled: true`
2. The environment variable PRCL_ALLOW_ATTACKS must be set to "1"

This ensures no accidental execution of poisoning code.
"""

import os

_GATE_ENV_VAR = "PRCL_ALLOW_ATTACKS"


def is_attack_allowed(cfg_enabled: bool) -> bool:
    """Check if attack execution is permitted."""
    env_allowed = os.environ.get(_GATE_ENV_VAR, "0") == "1"
    return cfg_enabled and env_allowed


def require_attack_gate(cfg_enabled: bool) -> None:
    """Raise if attack execution is not permitted."""
    if not cfg_enabled:
        raise RuntimeError(
            "Attack is not enabled in config. Set attack.enabled=true to proceed."
        )
    if os.environ.get(_GATE_ENV_VAR, "0") != "1":
        raise RuntimeError(
            f"Attack code is gated. Set environment variable {_GATE_ENV_VAR}=1 "
            "to confirm intentional use of poisoning adapters. "
            "This repository is defense-first; attack code is for controlled evaluation only."
        )
