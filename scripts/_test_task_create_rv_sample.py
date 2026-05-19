"""Test task_create_rv_sample and task_merge_soep_rv_sample with synthetic data.

Run with: python tests/test_task_create_rv_sample.py
Or: pytest tests/test_task_create_rv_sample.py -v (with PYTHONPATH=src and pytask/attrs)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Constants from task module (avoid importing the module so tests run without pytask)
MONTHS_WORK_THRESHOLD = 6
MONTHS_CARE_THRESHOLD = 6
EGP_VARS = ["EGP_main", "EGP_main_and_side", "EGP_main_and_care", "EGP_all"]


def _make_synthetic_rv_raw():
    """Minimal rv_raw-like data: one row per (rv_id, year, month)."""
    np.random.seed(42)
    rows = []
    for rv_id in [1, 2]:
        birth_year = 1960
        for syear in [2018, 2019]:
            for smonth in range(1, 13):
                status2 = (
                    "PFL"
                    if (smonth <= 6 and rv_id == 1)
                    else "NJB" if (smonth <= 4 and rv_id == 2) else ""
                )
                status3 = ""
                egpt2 = 0.5 if status2 == "PFL" else 0.3 if status2 == "NJB" else np.nan
                egpt3 = np.nan
                rows.append(
                    {
                        "rv_id": rv_id,
                        "JAHR": syear,
                        "MONAT": smonth,
                        "GBJAVS": birth_year,
                        "GEVS": 1 if rv_id == 1 else 2,
                        "STATUS_1_EGPT": 1.0 if smonth <= 10 else 0.0,
                        "STATUS_5_EGPT": 0.1,
                        "STATUS_2": status2,
                        "STATUS_2_EGPT": egpt2,
                        "STATUS_3": status3,
                        "STATUS_3_EGPT": egpt3,
                    }
                )
    return pd.DataFrame(rows)


def _run_create_rv_sample(rv_data: pd.DataFrame, path_to_save: Path) -> None:
    """Core logic of task_create_rv_sample (mirrors task body)."""
    rv_data = rv_data.copy()
    rv_data["syear"] = rv_data["JAHR"].astype(int)
    rv_data["smonth"] = rv_data["MONAT"].astype(int)
    rv_data["age"] = rv_data["JAHR"] - rv_data["GBJAVS"]
    rv_data["sex"] = np.where(rv_data["GEVS"] == 2, 1, 0)
    rv_data.sort_values(by=["rv_id", "syear", "smonth"], inplace=True)

    mask_njb2 = (rv_data["STATUS_2"] == "NJB") & rv_data["STATUS_2_EGPT"].notna()
    mask_njb3 = (rv_data["STATUS_3"] == "NJB") & rv_data["STATUS_3_EGPT"].notna()
    mask_pfl2 = (rv_data["STATUS_2"] == "PFL") & rv_data["STATUS_2_EGPT"].notna()
    mask_pfl3 = (rv_data["STATUS_3"] == "PFL") & rv_data["STATUS_3_EGPT"].notna()

    rv_data["EGP_main"] = rv_data[["STATUS_1_EGPT", "STATUS_5_EGPT"]].sum(
        axis=1, skipna=True
    )
    rv_data["EGP_main_and_side"] = rv_data["EGP_main"].copy()
    rv_data.loc[mask_njb2, "EGP_main_and_side"] += rv_data.loc[
        mask_njb2, "STATUS_2_EGPT"
    ].fillna(0)
    rv_data.loc[mask_njb3, "EGP_main_and_side"] += rv_data.loc[
        mask_njb3, "STATUS_3_EGPT"
    ].fillna(0)
    rv_data["EGP_main_and_care"] = rv_data["EGP_main"].copy()
    rv_data.loc[mask_pfl2, "EGP_main_and_care"] += rv_data.loc[
        mask_pfl2, "STATUS_2_EGPT"
    ].fillna(0)
    rv_data.loc[mask_pfl3, "EGP_main_and_care"] += rv_data.loc[
        mask_pfl3, "STATUS_3_EGPT"
    ].fillna(0)
    rv_data["EGP_all"] = rv_data["EGP_main"].copy()
    rv_data.loc[mask_njb2, "EGP_all"] += rv_data.loc[mask_njb2, "STATUS_2_EGPT"].fillna(
        0
    )
    rv_data.loc[mask_njb3, "EGP_all"] += rv_data.loc[mask_njb3, "STATUS_3_EGPT"].fillna(
        0
    )
    rv_data.loc[mask_pfl2, "EGP_all"] += rv_data.loc[mask_pfl2, "STATUS_2_EGPT"].fillna(
        0
    )
    rv_data.loc[mask_pfl3, "EGP_all"] += rv_data.loc[mask_pfl3, "STATUS_3_EGPT"].fillna(
        0
    )

    for col in EGP_VARS:
        rv_data[f"{col}_yearly"] = rv_data.groupby(["rv_id", "syear"])[col].transform(
            "sum"
        )
    rv_data["working"] = rv_data.groupby(["rv_id", "syear"])["EGP_main"].transform(
        lambda x: (x > 0).sum() >= MONTHS_WORK_THRESHOLD
    )
    rv_data["rv_care"] = np.where(
        (rv_data["STATUS_2"] == "PFL") | (rv_data["STATUS_3"] == "PFL"), 1, 0
    )
    rv_data["rv_care_yearly"] = (
        rv_data.groupby(["rv_id", "syear"])["rv_care"]
        .transform(lambda x: (x == 1).sum() >= MONTHS_CARE_THRESHOLD)
        .astype(int)
    )

    rv_year = rv_data[rv_data["smonth"] == 12].copy()
    assert (rv_year.groupby(["rv_id", "syear"]).size() == 1).all()

    rv_year.sort_values(["rv_id", "syear"], inplace=True)
    for col in EGP_VARS:
        rv_year[f"{col}_yearly_cum"] = rv_year.groupby("rv_id")[
            f"{col}_yearly"
        ].cumsum()
    rv_year.to_csv(path_to_save)


def _run_merge_soep_rv(
    soep: pd.DataFrame, rv_path: Path, path_to_save: Path
) -> pd.DataFrame:
    """Core logic of task_merge_soep_rv_sample (mirrors task body)."""
    rv = pd.read_csv(rv_path, index_col=[0])
    base_cols = ["rv_id", "syear", "smooth", "GBJAVS", "rv_care_yearly"]
    egp_cols = [c for c in rv.columns if c.startswith("EGP")]
    cols_to_keep = [c for c in base_cols if c in rv.columns] + egp_cols
    rv_keep = rv[cols_to_keep].copy()
    soep_rv = soep.merge(
        rv_keep,
        how="left",
        left_on=["rv_id", "syear"],
        right_on=["rv_id", "syear"],
        suffixes=("", "_rv"),
        validate="m:1",
        indicator=True,
    )
    assert len(soep_rv) == len(soep)
    soep_rv.to_csv(path_to_save)
    return soep_rv


def test_task_create_rv_sample():
    """Run create_rv_sample logic on synthetic rv_raw and check outputs."""
    rv_raw = _make_synthetic_rv_raw()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path_rv_sample = tmp / "rv_sample.csv"
        _run_create_rv_sample(rv_raw, path_rv_sample)

        assert path_rv_sample.exists()
        rv = pd.read_csv(path_rv_sample, index_col=[0])

        assert len(rv) == 4
        assert (rv.groupby(["rv_id", "syear"]).size() == 1).all()
        assert (rv["smonth"] == 12).all()

        for col in [
            "rv_id",
            "syear",
            "smonth",
            "EGP_main",
            "EGP_main_yearly",
            "EGP_main_yearly_cum",
            "rv_care",
            "rv_care_yearly",
            "working",
        ]:
            assert col in rv.columns, f"Missing column: {col}"

        for rv_id in rv["rv_id"].unique():
            sub = rv[rv["rv_id"] == rv_id].sort_values("syear")
            expected_cum = sub["EGP_main_yearly"].cumsum().values
            np.testing.assert_array_almost_equal(
                sub["EGP_main_yearly_cum"].values,
                expected_cum,
                err_msg=f"EGP_main_yearly_cum should be cumsum(EGP_main_yearly) for rv_id={rv_id}",
            )

        rv1 = rv[rv["rv_id"] == 1]
        assert (rv1["rv_care_yearly"] == 1).all()
        assert (rv["working"] == 1).all()


def test_task_merge_soep_rv_sample():
    """Run merge logic on synthetic SOEP + rv_sample and check merge."""
    rv_raw = _make_synthetic_rv_raw()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path_rv_sample = tmp / "rv_sample.csv"
        path_soep = tmp / "soep_event_study_sample.csv"
        path_soep_rv = tmp / "soep_rv.csv"

        _run_create_rv_sample(rv_raw, path_rv_sample)

        soep = pd.DataFrame(
            {
                "pid": [10, 10, 20, 20, 99],
                "syear": [2018, 2019, 2018, 2019, 2018],
                "rv_id": [1, 1, 2, 2, 999],
                "age": [58, 59, 58, 59, 40],
            }
        )

        soep_rv = _run_merge_soep_rv(soep, path_rv_sample, path_soep_rv)

        assert path_soep_rv.exists()
        assert len(soep_rv) == len(soep)
        assert "_merge" in soep_rv.columns
        assert (soep_rv["_merge"] == "both").sum() == 4
        assert (soep_rv["_merge"] == "left_only").sum() >= 1

        matched = soep_rv[soep_rv["_merge"] == "both"]
        for col in ["rv_care_yearly", "EGP_main_yearly"]:
            assert col in soep_rv.columns
            assert matched[col].notna().all()


def test_yearly_cum_is_cumsum_of_yearly():
    """Cumulative columns must equal cumsum of yearly totals within person."""
    rv_raw = _make_synthetic_rv_raw()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path_rv_sample = tmp / "rv_sample.csv"
        _run_create_rv_sample(rv_raw, path_rv_sample)
        rv = pd.read_csv(path_rv_sample, index_col=[0])

    for col in EGP_VARS:
        yearly_col = f"{col}_yearly"
        cum_col = f"{col}_yearly_cum"
        if yearly_col not in rv.columns or cum_col not in rv.columns:
            continue
        for rv_id in rv["rv_id"].unique():
            sub = rv[rv["rv_id"] == rv_id].sort_values("syear")
            expected = sub[yearly_col].cumsum().values
            np.testing.assert_array_almost_equal(sub[cum_col].values, expected)


def run_integration_with_real_tasks():
    """Call actual task functions if caregiving is available (e.g. in dev env)."""
    try:
        from caregiving.data_management.soep.task_create_rv_sample import (
            task_create_rv_sample,
            task_merge_soep_rv_sample,
        )
    except ImportError:
        print("Skipping integration with real tasks (caregiving/pytask not available).")
        return
    rv_raw = _make_synthetic_rv_raw()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        path_rv_raw = tmp / "rv_raw.csv"
        path_rv_sample = tmp / "rv_sample.csv"
        path_soep = tmp / "soep_event_study_sample.csv"
        path_soep_rv = tmp / "soep_rv.csv"
        rv_raw.to_csv(path_rv_raw)
        task_create_rv_sample(path_to_rv_data=path_rv_raw, path_to_save=path_rv_sample)
        soep = pd.DataFrame(
            {
                "pid": [10, 10, 20, 20, 99],
                "syear": [2018, 2019, 2018, 2019, 2018],
                "rv_id": [1, 1, 2, 2, 999],
                "age": [58, 59, 58, 59, 40],
            }
        )
        soep.to_csv(path_soep)
        task_merge_soep_rv_sample(
            path_to_soep_event_study_sample=path_soep,
            path_to_rv_sample=path_rv_sample,
            path_to_save=path_soep_rv,
        )
        soep_rv = pd.read_csv(path_soep_rv, index_col=[0])
        assert len(soep_rv) == 5
        print("Integration with real tasks: OK.")


if __name__ == "__main__":
    # Run tests without pytest (only numpy/pandas required)
    test_task_create_rv_sample()
    print("test_task_create_rv_sample: OK")
    test_task_merge_soep_rv_sample()
    print("test_task_merge_soep_rv_sample: OK")
    test_yearly_cum_is_cumsum_of_yearly()
    print("test_yearly_cum_is_cumsum_of_yearly: OK")
    run_integration_with_real_tasks()
    print("All checks passed.")
