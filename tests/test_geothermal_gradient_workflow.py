from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from biomass.geothermal.baseline_xgboost import xgboostPro


@dataclass
class Args:
    params_algorithm: str = "random"
    device: str = "cpu"
    tree_method: str = "hist"
    Attempt: str = "test"
    Run: str = "oceanic_smoke"


def _synthetic_training_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    gradient = 35.0 + 2.0 * f1 - 1.5 * f2 + 0.5 * f3 + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "is_land": False,
            "lat": rng.uniform(-60, 60, n),
            "lon": rng.uniform(-180, 180, n),
            "feature_a": f1,
            "feature_b": f2,
            "feature_c": f3,
            "gradient": gradient,
        }
    )


def test_geothermal_gradient_model_trains_and_predicts_on_small_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import biomass.geothermal.baseline_xgboost as baseline

    monkeypatch.setattr(baseline, "plot_corr_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(baseline, "plotPredictedTest", lambda *args, **kwargs: None)
    monkeypatch.setattr(baseline, "plot_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(baseline.plt.Figure, "savefig", lambda *args, **kwargs: None)

    data_path = tmp_path / "geothermal_training.csv"
    _synthetic_training_frame().to_csv(data_path, index=False)

    args = Args()
    output_dir = Path(f"{args.Attempt}Attempt") / args.Run
    (output_dir / "Plots").mkdir(parents=True)
    (Path(f"{args.Attempt}Attempt") / f"{args.Run}error" / "Plots").mkdir(parents=True)

    x_train, x_test, y_train, y_test, features = xgboostPro.load_data(
        data_path,
        args.Attempt,
        args.Run,
        is_land=False,
    )

    model_runner = xgboostPro(args)
    model_runner.model_params = {
        "max_depth": [2],
        "n_estimators": [5],
        "gamma": [0],
        "reg_lambda": [1],
        "min_child_weight": [1],
        "colsample_bytree": [1.0],
        "subsample": [1.0],
        "eta": [0.2],
    }
    model_runner.init_model()

    trained = model_runner.train(x_train, y_train, args.Attempt, args.Run, args.params_algorithm)
    preds = trained.predict(x_test)

    assert len(features) >= 1
    assert preds.shape[0] == y_test.shape[0]
    assert np.isfinite(preds).all()

    model_runner.save_model()
    loaded = xgboostPro.load_model(output_dir / "myModeltest.model")
    loaded_preds = loaded.predict(x_test)

    np.testing.assert_allclose(preds, loaded_preds)


def test_geothermal_inference_splits_missing_and_observed_gradients(tmp_path):
    frame = _synthetic_training_frame(8)
    frame.loc[[1, 3], "gradient"] = np.nan
    frame.loc[[0, 2, 4], "is_land"] = True
    data_path = tmp_path / "geothermal_inference.csv"
    frame.to_csv(data_path, index=False)

    ocean_missing, ocean_observed, continental_missing, continental_observed = xgboostPro.data_inference(data_path)

    assert len(ocean_missing) == 2
    assert len(ocean_observed) == 3
    assert len(continental_missing) == 0
    assert len(continental_observed) == 3
