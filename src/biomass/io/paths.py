"""Project-relative filesystem locations used by biomass workflows."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
FIGURES_DIR = PROJECT_ROOT / "figures"


def _project_relative(base: Path, *parts: str | Path, create_parent: bool = False) -> Path:
    """Build a project-relative path and optionally create its parent directory."""
    path = base.joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts: str | Path, create_parent: bool = False) -> Path:
    """Return a path below ``data/``."""
    return _project_relative(DATA_DIR, *parts, create_parent=create_parent)


def run_path(*parts: str | Path, create_parent: bool = False) -> Path:
    """Return a path below ``runs/``."""
    return _project_relative(RUNS_DIR, *parts, create_parent=create_parent)


def figure_path(*parts: str | Path, create_parent: bool = False) -> Path:
    """Return a path below ``figures/``."""
    return _project_relative(FIGURES_DIR, *parts, create_parent=create_parent)
