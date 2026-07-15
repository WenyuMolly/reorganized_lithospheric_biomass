"""Helpers for reproducible, timestamped run directories."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

from .paths import RUNS_DIR


def _validate_run_id(run_id: str) -> str:
    """Reject path-like run IDs before they are used as directory names."""
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty directory name, not a path")
    return run_id


def create_run_id(now: datetime | None = None) -> str:
    """Create a local-time run identifier such as ``20260714_153012``."""
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def get_run_id(run_id: str | None = None) -> str:
    """Use an explicit ID, ``BIOMASS_RUN_ID``, or a newly generated timestamp."""
    selected = run_id or os.environ.get("BIOMASS_RUN_ID") or create_run_id()
    return _validate_run_id(selected)


def run_directory(
    workflow: str,
    run_id: str | None = None,
    *,
    create: bool = True,
) -> Path:
    """Return ``runs/<workflow>/<run_id>`` and create it by default."""
    if not workflow or Path(workflow).name != workflow:
        raise ValueError("workflow must be a single directory name")
    directory = RUNS_DIR / workflow / get_run_id(run_id)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory
