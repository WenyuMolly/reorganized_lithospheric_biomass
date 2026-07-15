from datetime import datetime

import pandas as pd
import pytest

from biomass.io import create_run_id, read_csv_checked, run_directory, write_csv


def test_create_run_id_is_stable_for_given_time():
    assert create_run_id(datetime(2026, 7, 14, 15, 30, 12)) == "20260714_153012"


def test_run_directory_uses_explicit_run_id(monkeypatch, tmp_path):
    monkeypatch.setattr("biomass.io.runs.RUNS_DIR", tmp_path)

    directory = run_directory("oceanic", "test_run")

    assert directory == tmp_path / "oceanic" / "test_run"
    assert directory.is_dir()


def test_run_directory_rejects_path_like_identifiers():
    with pytest.raises(ValueError, match="directory name"):
        run_directory("oceanic", "../outside")


def test_checked_csv_round_trip(tmp_path):
    target = write_csv(pd.DataFrame({"lat": [1.0], "gradient": [20.0]}), tmp_path / "table.csv")

    result = read_csv_checked(target, required_columns=("lat", "gradient"))

    pd.testing.assert_frame_equal(result, pd.DataFrame({"lat": [1.0], "gradient": [20.0]}))


def test_checked_csv_reports_missing_columns(tmp_path):
    target = write_csv(pd.DataFrame({"lat": [1.0]}), tmp_path / "table.csv")

    with pytest.raises(ValueError, match="gradient"):
        read_csv_checked(target, required_columns=("lat", "gradient"))
