"""Run directory manager — creates structured run folders with full reproducibility metadata."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_pip_freeze() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    return info


def create_run_dir(
    base_dir: str = "./runs",
    run_name: str | None = None,
    config: dict | None = None,
    seed: int = 42,
) -> Path:
    """Create a timestamped run directory with full reproducibility metadata.

    Returns the Path to the created run directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{run_name}" if run_name else timestamp

    run_path = Path(base_dir) / dir_name
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "checkpoints").mkdir(exist_ok=True)
    (run_path / "cards").mkdir(exist_ok=True)
    (run_path / "logs").mkdir(exist_ok=True)

    # Save hardware info
    hw_info = _get_hardware_info()
    with open(run_path / "hardware.json", "w") as f:
        json.dump(hw_info, f, indent=2)

    # Save pip freeze
    with open(run_path / "env.txt", "w") as f:
        f.write(_get_pip_freeze())

    # Save git commit
    git_hash = _get_git_hash()

    # Save reproducibility metadata
    meta = {
        "timestamp": timestamp,
        "git_hash": git_hash,
        "seed": seed,
        "command": " ".join(sys.argv),
    }
    with open(run_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save resolved config if provided
    if config is not None:
        from omegaconf import OmegaConf

        with open(run_path / "config_resolved.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(config))

    # Initialize empty metrics file
    with open(run_path / "metrics.json", "w") as f:
        json.dump({"train": [], "eval": {}}, f, indent=2)

    return run_path


def save_metrics(run_path: Path, metrics: dict) -> None:
    """Overwrite the metrics.json file with updated metrics."""
    with open(run_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
