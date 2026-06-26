from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_dir(attempt: str, run: str) -> Path:
    path = Path(f"{attempt}Attempt") / str(run) / "Plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_corr_matrix(corr_matrix: pd.DataFrame, attempt: str, run: str) -> None:
    """Save a correlation-matrix diagnostic plot used by baseline_xgboost.py."""
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(corr_matrix.columns)), 7))
    image = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr_matrix.index, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman r")
    fig.tight_layout()
    fig.savefig(_plot_dir(attempt, run) / "correlation_matrix.png", dpi=300)
    plt.close(fig)


def plotPredictedTest(y_true, y_pred, attempt: str, run: str) -> None:
    """Save observed-versus-predicted gradient diagnostics."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.7)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.any():
        lower = min(float(y_true[finite].min()), float(y_pred[finite].min()))
        upper = max(float(y_true[finite].max()), float(y_pred[finite].max()))
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed geothermal gradient")
    ax.set_ylabel("Predicted geothermal gradient")
    ax.set_title(f"Observed vs. predicted ({run})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_plot_dir(attempt, run) / "predicted_vs_observed.png", dpi=300)
    plt.close(fig)


def plot_feature(importances, features, attempt: str, run: str) -> None:
    """Save feature-importance bar plot."""
    values = np.asarray(importances, dtype=float)
    order = np.argsort(values)

    fig_h = max(5, 0.35 * len(features))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(np.asarray(features)[order], values[order])
    ax.set_xlabel("Feature importance")
    ax.set_title(f"XGBoost feature importance ({run})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(_plot_dir(attempt, run) / "feature_importance.png", dpi=300)
    plt.close(fig)

