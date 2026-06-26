#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature importance combo plot for XGBoost (oceanic / continental):
  - Baseline importance (gain, normalized to sum=1) from the *trained* baseline model
  - LOFO ΔR² (test): R²_full - R²_without  (can be negative)
  - LOFO Δ(RMSE/Mean(y)) on test: (RMSE/Mean)_without - (RMSE/Mean)_full (positive = worse)

Baseline (full model) is evaluated by LOADING your saved model object (Scheme A):
  - If the saved object is GridSearchCV/RandomizedSearchCV: use best_estimator_
  - Else: use it directly as an XGBRegressor-like estimator

LOFO refits use identical hyperparameters to the loaded baseline estimator.

Outputs:
  - feature_combo_results.csv
  - feature_combo_lofo_deltas.png
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor


# ---------- Utilities ----------
def read_y(path: str, y_col: str | None):
    """Read target y from CSV/TXT; if y_col missing, use first column."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None)

    if df.shape[1] == 1 and y_col is None:
        return df.iloc[:, 0].astype(float).to_numpy()

    if y_col is not None and y_col in df.columns:
        return df[y_col].astype(float).to_numpy()

    return df.iloc[:, 0].astype(float).to_numpy()


def load_baseline_estimator(model_path: str):
    """
    Load a pickled model.
    - If it's a SearchCV object, use best_estimator_.
    - Otherwise, return the loaded object itself.
    """
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    if hasattr(obj, "best_estimator_") and obj.best_estimator_ is not None:
        est = obj.best_estimator_
        kind = obj.__class__.__name__ + ".best_estimator_"
    else:
        est = obj
        kind = obj.__class__.__name__

    # Minimal sanity check
    if not hasattr(est, "predict"):
        raise TypeError(f"Loaded baseline object ({kind}) does not implement predict().")

    return est, kind


def get_xgb_params_from_estimator(est) -> dict:
    """
    Extract XGBRegressor params from an estimator to refit LOFO models.
    Handles:
      - XGBRegressor
      - sklearn Pipeline ending in XGBRegressor (best-effort)
    """
    # Direct XGBRegressor
    if isinstance(est, XGBRegressor):
        params = est.get_params()
    else:
        # Best-effort: pipeline-like (has named_steps)
        if hasattr(est, "named_steps"):
            # try to find an XGBRegressor in steps
            xgb_step = None
            for _, step in est.named_steps.items():
                if isinstance(step, XGBRegressor):
                    xgb_step = step
                    break
            if xgb_step is None:
                raise TypeError("Baseline estimator is a pipeline, but no XGBRegressor step was found.")
            params = xgb_step.get_params()
        else:
            # If it's not XGBRegressor/pipeline, still try get_params
            if not hasattr(est, "get_params"):
                raise TypeError("Baseline estimator has no get_params(); cannot refit LOFO with identical hyperparameters.")
            params = est.get_params()

    # Ensure objective and n_jobs are set reasonably
    params = dict(params)
    params.update(dict(objective="reg:squarederror", n_jobs=params.get("n_jobs", 4)))

    # Remove keys that are not accepted / can cause warnings
    for k in ["importance_type", "device", "verbosity"]:
        params.pop(k, None)

    return params


def get_gain_normalized(model: XGBRegressor, columns):
    """Return normalized 'gain' importance as a dict {feature: share} (sum = 1)."""
    bst = model.get_booster()
    raw = bst.get_score(importance_type="gain") or {}
    mapped = {f"f{i}": col for i, col in enumerate(columns)}
    gain = {mapped.get(k, k): v for k, v in raw.items()}

    for c in columns:
        gain.setdefault(c, 0.0)

    total = float(sum(gain.values()))
    return {k: (v / total if total > 0 else 0.0) for k, v in gain.items()}


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Gain + LOFO ΔR² + Δ(RMSE/Mean) combo plot (Scheme A baseline load).")
    ap.add_argument("--domain", default="oceanic", choices=["oceanic", "continental"],
                    help="Choose which set of paths to use.")
    ap.add_argument("--y-col", default=None, help="Optional target column name.")
    default_out_dir = Path(__file__).resolve().parents[3] / "figures" / "generated"
    ap.add_argument("--out-csv", default=str(default_out_dir / "feature_combo_results_oceanic.csv"), help="Output CSV.")
    ap.add_argument("--out-fig", default=str(default_out_dir / "feature_combo_lofo_deltas_oceanic.png"), help="Output figure.")
    ap.add_argument("--sort-by", default="delta_r2",
                    choices=["delta_r2", "delta_rmse_over_mean", "gain_norm"],
                    help="Sort bars: ΔR² (desc), Δ(RMSE/Mean) (desc), or gain (desc).")
    ap.add_argument("--train-x", default=None, help="Override training feature CSV.")
    ap.add_argument("--test-x", default=None, help="Override test feature CSV.")
    ap.add_argument("--train-y", default=None, help="Override training target CSV/TXT.")
    ap.add_argument("--test-y", default=None, help="Override test target CSV/TXT.")
    ap.add_argument("--model", default=None, help="Override pickled model path.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    geothermal_root = project_root / "runs" / "geothermal"
    presets = {
        "oceanic": {
            "train_x": geothermal_root / "1stAttempt/oceanic_final/x_train_data.csv",
            "test_x":  geothermal_root / "1stAttempt/oceanic_final/x_test_data.csv",
            "train_y": geothermal_root / "1stAttempt/oceanic_final/y_train_data.csv",
            "test_y":  geothermal_root / "1stAttempt/oceanic_final/y_test_data.csv",
            "model":   geothermal_root / "1stAttempt/oceanic_final/myModel1st.model",
        },
        "continental": {
            "train_x": geothermal_root / "1stAttempt/continental_final/x_train_data.csv",
            "test_x":  geothermal_root / "1stAttempt/continental_final/x_test_data.csv",
            "train_y": geothermal_root / "1stAttempt/continental_final/y_train_data.csv",
            "test_y":  geothermal_root / "1stAttempt/continental_final/y_test_data.csv",
            "model":   geothermal_root / "1stAttempt/continental_final/myModel1st.model",
        },
    }
    paths = presets[args.domain]
    overrides = {
        "train_x": args.train_x,
        "test_x": args.test_x,
        "train_y": args.train_y,
        "test_y": args.test_y,
        "model": args.model,
    }
    paths = {key: Path(overrides[key]) if overrides[key] else Path(value) for key, value in paths.items()}

    # Load data
    Xtr = pd.read_csv(paths["train_x"])
    Xte = pd.read_csv(paths["test_x"])
    ytr = read_y(paths["train_y"], args.y_col)
    yte = read_y(paths["test_y"], args.y_col)

    # Align columns & fill NA with train medians (match your training-time CSV usage)
    Xte = Xte[Xtr.columns]
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    for c in Xtr.columns:
        med = float(np.nanmedian(Xtr[c].to_numpy()))
        Xtr[c] = Xtr[c].fillna(med)
        Xte[c] = Xte[c].fillna(med)

    # ---- Scheme A baseline: load saved model and evaluate directly ----
    baseline_est, baseline_kind = load_baseline_estimator(paths["model"])

    # Baseline predictions (use the loaded trained estimator)
    yhat_full = baseline_est.predict(Xte.to_numpy())
    r2_full = float(r2_score(yte, yhat_full))
    rmse_full = float(root_mean_squared_error(yte, yhat_full))

    mean_y = float(np.mean(yte))
    if not np.isfinite(mean_y) or mean_y == 0:
        mean_y = 1.0
    rmse_full_over_mean = rmse_full / mean_y

    # For gain, we need an XGBRegressor instance. If baseline is SearchCV, best_estimator_ should be XGBRegressor.
    if isinstance(baseline_est, XGBRegressor):
        gain_norm = get_gain_normalized(baseline_est, Xtr.columns)
        params = get_xgb_params_from_estimator(baseline_est)
    else:
        # If baseline is not XGBRegressor, try to extract params; gain may be unavailable
        params = get_xgb_params_from_estimator(baseline_est)
        gain_norm = {c: 0.0 for c in Xtr.columns}  # fallback; avoids crash

    print(f"[INFO] Baseline loaded as: {baseline_kind}")
    print(f"[INFO] Full-model test R² = {r2_full:.4f}, RMSE/Mean(y) = {rmse_full_over_mean:.4f}")

    # ---- LOFO refits (identical hyperparameters as baseline estimator) ----
    rows = []
    for fdrop in Xtr.columns:
        keep = [c for c in Xtr.columns if c != fdrop]
        mdl = XGBRegressor(**params)
        mdl.fit(Xtr[keep].to_numpy(), ytr)

        yhat_wo = mdl.predict(Xte[keep].to_numpy())
        r2_wo = float(r2_score(yte, yhat_wo))
        rmse_wo = float(root_mean_squared_error(yte, yhat_wo))
        rmse_over_mean_wo = rmse_wo / mean_y

        delta_r2 = r2_full - r2_wo
        delta_rmse_over_mean = rmse_over_mean_wo - rmse_full_over_mean  # positive = worse

        rows.append(dict(
            feature=fdrop,
            gain_norm=float(gain_norm.get(fdrop, 0.0)),

            # Full model baselines (same for all rows; kept for convenience)
            r2_full=r2_full,
            rmse_full_over_mean=rmse_full_over_mean,

            # Without-feature model metrics
            r2_without=r2_wo,
            rmse_over_mean_without=rmse_over_mean_wo,

            # LOFO deltas
            delta_r2=delta_r2,
            delta_rmse_over_mean=delta_rmse_over_mean,
        ))

    df = pd.DataFrame(rows)

    # Sort for plotting
    if args.sort_by == "delta_r2":
        df = df.sort_values("delta_r2", ascending=False)
    elif args.sort_by == "delta_rmse_over_mean":
        df = df.sort_values("delta_rmse_over_mean", ascending=False)
    else:
        df = df.sort_values("gain_norm", ascending=False)

    # Save CSV
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved CSV: {args.out_csv}")

    # --- Plot: Gain + LOFO deltas ---
    fig_h = max(5, 0.48 * len(df))
    fig, ax = plt.subplots(figsize=(10.8, fig_h))
    y = np.arange(len(df))
    h = 0.24

    ax.barh(y + h, df["gain_norm"],            height=h, label="Gain (normalized)")
    ax.barh(y,     df["delta_r2"],             height=h, label="LOFO ΔR² (test)")
    ax.barh(y - h, df["delta_rmse_over_mean"], height=h, label="LOFO Δ(RMSE/Mean(y)) (test)")

    ax.axvline(0.0, color="k", linewidth=0.8, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance / Δ performance (relative to full model)")
    ax.set_title(f"Feature importance (Gain + LOFO deltas) — {args.domain}")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=300)
    print(f"[OK] Saved figure: {args.out_fig}")


if __name__ == "__main__":
    main()
