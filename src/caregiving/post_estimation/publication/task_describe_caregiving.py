"""Describe care demand and informal caregiving from simulated data (LaTeX table).

This module produces a single LaTeX table summarizing:
- Panel A: Care demand (ages, first spell start)
- Panel B: Informal caregiving (ages, first spell start)
- Panel C: Distribution of caregiving years
- Panel D: Care demand spell length distribution (all, light, intensive)
- Panel E/G: Average care demand spell length by agent's age at mother's death
  (all care / light only / intensive only)

Three pytask tasks run the same logic on:
1) simulated_data_estimated_params.pkl
2) simulated_data_estimated_params_back_to_Jan7.pkl
3) simulated_data_estimated_params_Jan7.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_utils import ensure_agent_period
from caregiving.model.shared import (
    CARE_DEMAND_INTENSIVE,
    CARE_DEMAND_LIGHT,
    DEAD,
    INFORMAL_CARE,
    PARENT_RECENTLY_DEAD,
    RETIREMENT,
)

# -----------------------------------------------------------------------------
# Pytask tasks (same structure, different data path and output path)
# -----------------------------------------------------------------------------


@pytask.mark.post_estimation
@pytask.mark.describe_caregiving
def task_describe_caregiving_estimated_params(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "describe_caregiving_estimated_params.tex",
):
    """Describe care demand and caregiving (simulated_data_estimated_params.pkl)."""
    run_describe_caregiving(
        path_to_simulated_data=path_to_simulated_data,
        path_to_specs=path_to_specs,
        path_to_output_table=path_to_table,
        table_label="tab:describe_caregiving_estimated_params",
    )


@pytask.mark.post_estimation
# @pytask.mark.describe_caregiving
def task_describe_caregiving_back_to_jan7(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_back_to_Jan7.pkl",
    path_to_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "describe_caregiving_back_to_Jan7.tex",
):
    """Describe care demand and caregiving (simulated_data_estimated_params_back_to_Jan7.pkl)."""
    run_describe_caregiving(
        path_to_simulated_data=path_to_simulated_data,
        path_to_specs=path_to_specs,
        path_to_output_table=path_to_table,
        table_label="tab:describe_caregiving_back_to_jan7",
    )


@pytask.mark.post_estimation
# @pytask.mark.describe_caregiving
def task_describe_caregiving_jan7(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params_Jan7.pkl",
    path_to_table: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "describe_caregiving_Jan7.tex",
):
    """Describe care demand and caregiving (simulated_data_estimated_params_Jan7.pkl)."""
    run_describe_caregiving(
        path_to_simulated_data=path_to_simulated_data,
        path_to_specs=path_to_specs,
        path_to_output_table=path_to_table,
        table_label="tab:describe_caregiving_jan7",
    )


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def prepare_data(
    df_raw: pd.DataFrame,
    start_age: int,
) -> pd.DataFrame:
    """Ensure agent/period/age and restrict to alive person-periods."""
    df = df_raw.reset_index(drop=True)
    df = ensure_agent_period(df)
    if "age" not in df.columns:
        df["age"] = df["period"] + start_age
    df = df.loc[df["health"] != DEAD].copy()
    return df


def add_first_care_demand_and_caregiving(df: pd.DataFrame) -> pd.DataFrame:
    """Add first_care_demand_period, first_caregiving_period, and current_caregiving."""
    df = df.copy()
    care_demand_mask = df["care_demand"] > 0
    first_care_demand = (
        df.loc[care_demand_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_demand_period"})
    )
    informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df["choice"].isin(informal_care_codes)
    first_caregiving = (
        df.loc[caregiving_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_caregiving_period"})
    )
    df = df.merge(first_care_demand, on="agent", how="left")
    df = df.merge(first_caregiving, on="agent", how="left")
    df["current_caregiving"] = caregiving_mask.astype(int)
    return df


def add_age_at_mother_death(df: pd.DataFrame) -> pd.DataFrame:
    """Add age_at_mother_death per agent (agent's age when mother dies)."""
    death_mask = df["mother_dead"] == PARENT_RECENTLY_DEAD
    first_death = (
        df.loc[death_mask, ["agent", "period", "age"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"age": "age_at_mother_death"})[
            ["agent", "age_at_mother_death"]
        ]
    )
    return df.merge(first_death, on="agent", how="left")


def _consecutive_spells(
    df: pd.DataFrame,
    condition: pd.Series,
) -> pd.DataFrame:
    """Return one row per spell: agent, spell_length, start_period. Consecutive periods with condition True."""
    sub = df.loc[condition].copy()
    if sub.empty:
        return pd.DataFrame(columns=["agent", "spell_length", "start_period"])
    sub = sub.sort_values(["agent", "period"])
    sub["_prev_period"] = sub.groupby("agent")["period"].shift(1)
    sub["_gap"] = (sub["period"] - sub["_prev_period"]) != 1
    sub["_spell_start"] = sub["_gap"] | sub["_prev_period"].isna()
    sub["_spell_id"] = sub.groupby("agent")["_spell_start"].cumsum()
    spell_len = (
        sub.groupby(["agent", "_spell_id"], as_index=False)
        .agg(
            spell_length=("period", "count"),
            start_period=("period", "min"),
        )
        .drop(columns=["_spell_id"])
    )
    return spell_len


def care_demand_spells(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (all care demand spells, light-only spells, intensive-only spells)."""
    any_care = df["care_demand"] > 0
    light = df["care_demand"] == CARE_DEMAND_LIGHT
    intensive = df["care_demand"] == CARE_DEMAND_INTENSIVE
    spells_all = _consecutive_spells(df, any_care)
    spells_light = _consecutive_spells(df, light)
    spells_intensive = _consecutive_spells(df, intensive)
    return spells_all, spells_light, spells_intensive


def caregiving_spells(df: pd.DataFrame) -> pd.DataFrame:
    """Return consecutive caregiving spells (one row per spell: agent, spell_length, start_period)."""
    return _consecutive_spells(df, df["current_caregiving"] == 1)


# Age bins [40,44], [45,49], ..., [65,69] for spell-start and agent-age grouping
CARE_AGE_BINS = [(40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]
CARE_AGE_BIN_LABELS = ["40--44", "45--49", "50--54", "55--59", "60--64", "65--69"]


def _add_start_age_to_spells(spells: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add start_age (agent's age at spell start) to spells. Requires spells to have start_period."""
    if spells.empty or "start_period" not in spells.columns:
        return spells.copy()
    age_at_period = (
        df[["agent", "period", "age"]]
        .drop_duplicates(["agent", "period"])
        .rename(columns={"period": "start_period", "age": "start_age"})
    )
    out = spells.merge(
        age_at_period[["agent", "start_period", "start_age"]],
        on=["agent", "start_period"],
        how="left",
    )
    return out


# -----------------------------------------------------------------------------
# Panel statistics (each returns a dict of values for the table)
# -----------------------------------------------------------------------------


def compute_panel_a(df: pd.DataFrame) -> dict[str, Any]:
    """Panel A: Care demand – mean/median age when in care demand, age at first care demand."""
    cd = df.loc[df["care_demand"] > 0]
    if cd.empty:
        return {
            "mean_age_care_demand": (np.nan, np.nan),
            "median_age_care_demand": (np.nan, np.nan),
            "mean_age_first_care_demand": (np.nan, np.nan),
            "median_age_first_care_demand": (np.nan, np.nan),
        }
    ages = cd["age"]
    first_rows = (
        df.loc[df["care_demand"] > 0]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")[["agent", "age"]]
        .rename(columns={"age": "age_first"})
    )
    first_ages_series = first_rows.set_index("agent")["age_first"]
    return {
        "mean_age_care_demand": (float(ages.mean()), float(ages.std())),
        "median_age_care_demand": (float(ages.median()), float(ages.std())),
        "mean_age_first_care_demand": (
            float(first_ages_series.mean()),
            float(first_ages_series.std()),
        ),
        "median_age_first_care_demand": (
            float(first_ages_series.median()),
            float(first_ages_series.std()),
        ),
    }


def compute_panel_b(df: pd.DataFrame) -> dict[str, Any]:
    """Panel B: Informal caregiving – mean/median age when caregiving, age at first spell."""
    caregiving = df.loc[df["current_caregiving"] == 1]
    if caregiving.empty:
        return {
            "mean_age_caregiver": (np.nan, np.nan),
            "median_age_caregiver": (np.nan, np.nan),
            "mean_age_first_caregiving": (np.nan, np.nan),
            "median_age_first_caregiving": (np.nan, np.nan),
        }
    ages = caregiving["age"]
    first_rows = (
        df.loc[df["current_caregiving"] == 1]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")[["agent", "age"]]
        .rename(columns={"age": "age_first"})
    )
    first_ages_series = first_rows.set_index("agent")["age_first"]
    return {
        "mean_age_caregiver": (float(ages.mean()), float(ages.std())),
        "median_age_caregiver": (float(ages.median()), float(ages.std())),
        "mean_age_first_caregiving": (
            float(first_ages_series.mean()),
            float(first_ages_series.std()),
        ),
        "median_age_first_caregiving": (
            float(first_ages_series.median()),
            float(first_ages_series.std()),
        ),
    }


def compute_panel_c(df: pd.DataFrame) -> dict[str, Any]:
    """Panel C (total): Distribution of total caregiving years per agent (shares 1/2/3/4/5+, <=2/<=3/<=4/<=5)."""
    years = (
        df.groupby("agent")["current_caregiving"]
        .sum()
        .astype(int)
        .rename("caregiving_years")
    )
    with_care = years[years > 0]
    if with_care.empty:
        n = 0
        mean_y = np.nan
        se_mean = np.nan
        median_y = np.nan
        se_median = np.nan
        p1 = p2 = p3 = p4 = p5plus = np.nan
        ple2 = ple3 = ple4 = ple5 = np.nan
    else:
        n = len(with_care)
        mean_y = float(with_care.mean())
        se_mean = float(with_care.std() / np.sqrt(n))
        median_y = float(with_care.median())
        # approximate SE of median
        se_median = float(1.253 * with_care.std() / np.sqrt(n))
        p1 = 100 * (with_care == 1).sum() / n
        p2 = 100 * (with_care == 2).sum() / n
        p3 = 100 * (with_care == 3).sum() / n
        p4 = 100 * (with_care == 4).sum() / n
        p5plus = 100 * (with_care >= 5).sum() / n
        ple2 = 100 * (with_care <= 2).sum() / n
        ple3 = 100 * (with_care <= 3).sum() / n
        ple4 = 100 * (with_care <= 4).sum() / n
        ple5 = 100 * (with_care <= 5).sum() / n
    return {
        "avg_caregiving_years": (mean_y, se_mean),
        "median_caregiving_years": (median_y, se_median),
        "pct_1_year": p1,
        "pct_2_years": p2,
        "pct_3_years": p3,
        "pct_4_years": p4,
        "pct_5_plus_years": p5plus,
        "pct_le_2_years": ple2,
        "pct_le_3_years": ple3,
        "pct_le_4_years": ple4,
        "pct_le_5_years": ple5,
    }


def compute_panel_c_consecutive(spells: pd.DataFrame) -> dict[str, Any]:
    """Panel C (consecutive): Distribution of consecutive caregiving spell lengths (% of spells, cond. on care > 0)."""
    if spells.empty:
        return {
            "avg_caregiving_years": (np.nan, np.nan),
            "median_caregiving_years": (np.nan, np.nan),
            "pct_1_year": np.nan,
            "pct_2_years": np.nan,
            "pct_3_years": np.nan,
            "pct_4_years": np.nan,
            "pct_5_plus_years": np.nan,
            "pct_le_2_years": np.nan,
            "pct_le_3_years": np.nan,
            "pct_le_4_years": np.nan,
            "pct_le_5_years": np.nan,
        }
    s = spells["spell_length"]
    n = len(s)
    mean_y = float(s.mean())
    se_mean = float(s.std() / np.sqrt(n))
    median_y = float(s.median())
    se_median = float(1.253 * s.std() / np.sqrt(n))
    return {
        "avg_caregiving_years": (mean_y, se_mean),
        "median_caregiving_years": (median_y, se_median),
        "pct_1_year": 100 * (s == 1).sum() / n,
        "pct_2_years": 100 * (s == 2).sum() / n,
        "pct_3_years": 100 * (s == 3).sum() / n,
        "pct_4_years": 100 * (s == 4).sum() / n,
        "pct_5_plus_years": 100 * (s >= 5).sum() / n,
        "pct_le_2_years": 100 * (s <= 2).sum() / n,
        "pct_le_3_years": 100 * (s <= 3).sum() / n,
        "pct_le_4_years": 100 * (s <= 4).sum() / n,
        "pct_le_5_years": 100 * (s <= 5).sum() / n,
    }


def _spell_length_distribution(spells: pd.DataFrame) -> dict[str, float]:
    """Share of spells of length 1, 2, 3, 4, 5+ (percent, sum to 100)."""
    if spells.empty or spells["spell_length"].sum() == 0:
        return {
            "share_1": np.nan,
            "share_2": np.nan,
            "share_3": np.nan,
            "share_4": np.nan,
            "share_5plus": np.nan,
        }
    n_spells = len(spells)
    s = spells["spell_length"]
    return {
        "share_1": 100 * (s == 1).sum() / n_spells,
        "share_2": 100 * (s == 2).sum() / n_spells,
        "share_3": 100 * (s == 3).sum() / n_spells,
        "share_4": 100 * (s == 4).sum() / n_spells,
        "share_5plus": 100 * (s >= 5).sum() / n_spells,
    }


def compute_panel_d(
    spells_all: pd.DataFrame,
    spells_light: pd.DataFrame,
    spells_intensive: pd.DataFrame,
) -> dict[str, Any]:
    """Panel D: Consecutive care demand spell length distribution (all, light, intensive)."""
    return {
        "all": _spell_length_distribution(spells_all),
        "light": _spell_length_distribution(spells_light),
        "intensive": _spell_length_distribution(spells_intensive),
    }


def _total_years_distribution(years: pd.Series) -> dict[str, float]:
    """Share of agents (with years > 0) who have exactly 1, 2, 3, 4, 5+ total years."""
    with_any = years[years > 0]
    if with_any.empty:
        return {
            "share_1": np.nan,
            "share_2": np.nan,
            "share_3": np.nan,
            "share_4": np.nan,
            "share_5plus": np.nan,
        }
    n = len(with_any)
    return {
        "share_1": 100 * (with_any == 1).sum() / n,
        "share_2": 100 * (with_any == 2).sum() / n,
        "share_3": 100 * (with_any == 3).sum() / n,
        "share_4": 100 * (with_any == 4).sum() / n,
        "share_5plus": 100 * (with_any >= 5).sum() / n,
    }


def compute_panel_d_total(df: pd.DataFrame) -> dict[str, Any]:
    """Panel D (total): Distribution of total care demand years per agent (% with 1, 2, ... 5+ years)."""
    years_all = (
        df.groupby("agent")
        .apply(lambda g: (g["care_demand"] > 0).sum())
        .rename("total_years")
    )
    years_light = (
        df.groupby("agent")
        .apply(lambda g: (g["care_demand"] == CARE_DEMAND_LIGHT).sum())
        .rename("total_years")
    )
    years_intensive = (
        df.groupby("agent")
        .apply(lambda g: (g["care_demand"] == CARE_DEMAND_INTENSIVE).sum())
        .rename("total_years")
    )
    return {
        "all": _total_years_distribution(years_all),
        "light": _total_years_distribution(years_light),
        "intensive": _total_years_distribution(years_intensive),
    }


def _avg_spell_length_by_mother_death_age(
    spells: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, float]:
    """Average spell length in buckets: mother death before agent age <50, 50-<60, 60-<70."""
    if spells.empty:
        return {"avg_lt_50": np.nan, "avg_50_60": np.nan, "avg_60_70": np.nan}
    age_death = df.dropna(subset=["age_at_mother_death"]).drop_duplicates("agent")[
        ["agent", "age_at_mother_death"]
    ]
    merged = spells.merge(age_death, on="agent", how="inner")
    if merged.empty:
        return {"avg_lt_50": np.nan, "avg_50_60": np.nan, "avg_60_70": np.nan}
    a = merged["age_at_mother_death"]
    return {
        "avg_lt_50": float(merged.loc[a < 50, "spell_length"].mean()),
        "avg_50_60": float(merged.loc[(a >= 50) & (a < 60), "spell_length"].mean()),
        "avg_60_70": float(merged.loc[(a >= 60) & (a < 70), "spell_length"].mean()),
    }


def compute_panel_e_f_g(
    spells_all: pd.DataFrame,
    spells_light: pd.DataFrame,
    spells_intensive: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Panel E (all), F (light), G (intensive): avg consecutive spell length by agent age at mother death."""
    return {
        "all": _avg_spell_length_by_mother_death_age(spells_all, df),
        "light": _avg_spell_length_by_mother_death_age(spells_light, df),
        "intensive": _avg_spell_length_by_mother_death_age(spells_intensive, df),
    }


def _avg_total_years_by_mother_death_age(
    agent_total_years: pd.Series,
    df: pd.DataFrame,
) -> dict[str, float]:
    """Average total care demand years per agent in buckets by agent's age at mother's death."""
    if agent_total_years.empty or agent_total_years.sum() == 0:
        return {"avg_lt_50": np.nan, "avg_50_60": np.nan, "avg_60_70": np.nan}
    age_death = df.dropna(subset=["age_at_mother_death"]).drop_duplicates("agent")[
        ["agent", "age_at_mother_death"]
    ]
    merged = (
        agent_total_years.rename("total_years")
        .reset_index()
        .merge(age_death, on="agent", how="inner")
    )
    if merged.empty:
        return {"avg_lt_50": np.nan, "avg_50_60": np.nan, "avg_60_70": np.nan}
    a = merged["age_at_mother_death"]
    return {
        "avg_lt_50": float(merged.loc[a < 50, "total_years"].mean()),
        "avg_50_60": float(merged.loc[(a >= 50) & (a < 60), "total_years"].mean()),
        "avg_60_70": float(merged.loc[(a >= 60) & (a < 70), "total_years"].mean()),
    }


def compute_panel_e_f_g_total(df: pd.DataFrame) -> dict[str, Any]:
    """Avg total care demand years (non-consecutive) by agent age at mother death (all, light, intensive)."""
    years_all = df.groupby("agent").apply(lambda g: (g["care_demand"] > 0).sum())
    years_light = df.groupby("agent").apply(
        lambda g: (g["care_demand"] == CARE_DEMAND_LIGHT).sum()
    )
    years_intensive = df.groupby("agent").apply(
        lambda g: (g["care_demand"] == CARE_DEMAND_INTENSIVE).sum()
    )
    return {
        "all": _avg_total_years_by_mother_death_age(years_all, df),
        "light": _avg_total_years_by_mother_death_age(years_light, df),
        "intensive": _avg_total_years_by_mother_death_age(years_intensive, df),
    }


def compute_panel_h(
    spells_caregiving: pd.DataFrame,
    spells_care_demand_all: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Panel H: Share of caregiving spells (left) and share of care demand spells (right) starting in each age bin.
    Shares are out of ALL such spells, so each column sums to 100%.
    """
    spells_cg = _add_start_age_to_spells(spells_caregiving.copy(), df)
    spells_cd = _add_start_age_to_spells(spells_care_demand_all.copy(), df)
    out_cg = {}
    out_cd = {}
    n_cg = len(spells_cg)
    n_cd = len(spells_cd)
    for (lo, hi), lab in zip(CARE_AGE_BINS, CARE_AGE_BIN_LABELS):
        mask_cg = spells_cg["start_age"].between(lo, hi - 1, inclusive="both")
        mask_cd = spells_cd["start_age"].between(lo, hi - 1, inclusive="both")
        if n_cg > 0:
            out_cg[lab] = 100 * mask_cg.sum() / n_cg
        else:
            out_cg[lab] = np.nan
        if n_cd > 0:
            out_cd[lab] = 100 * mask_cd.sum() / n_cd
        else:
            out_cd[lab] = np.nan
    return {"consec_caregiving": out_cg, "care_demand": out_cd}


def compute_panel_i(
    spells_caregiving: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Panel I: By age bin of spell start: left = avg consecutive caregiving spell length, right = avg total caregiving years (agents whose first spell starts in that bin)."""
    spells_cg = _add_start_age_to_spells(spells_caregiving.copy(), df)
    total_years = df.groupby("agent")["current_caregiving"].sum().rename("total_years")
    first_caregiving_age = (
        df.loc[df["current_caregiving"] == 1]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")[["agent", "age"]]
        .rename(columns={"age": "first_caregiving_age"})
    )
    agents_with_total = total_years.reset_index().merge(
        first_caregiving_age, on="agent", how="inner"
    )
    out_consec = {}
    out_total = {}
    for (lo, hi), lab in zip(CARE_AGE_BINS, CARE_AGE_BIN_LABELS):
        # Consecutive: spells that start in this age bin
        mask = spells_cg["start_age"].between(lo, hi - 1, inclusive="both")
        sub = spells_cg.loc[mask, "spell_length"]
        out_consec[lab] = float(sub.mean()) if len(sub) > 0 else np.nan
        # Total: agents whose first caregiving age is in this bin
        agent_mask = agents_with_total["first_caregiving_age"].between(
            lo, hi - 1, inclusive="both"
        )
        sub_total = agents_with_total.loc[agent_mask, "total_years"]
        out_total[lab] = float(sub_total.mean()) if len(sub_total) > 0 else np.nan
    return {"consec_avg_length": out_consec, "total_avg_years": out_total}


def compute_panel_j(
    df: pd.DataFrame,
    spells_caregiving: pd.DataFrame,
) -> dict[str, Any]:
    """Panel J: Same outcomes as Panel I (avg consec caregiving spell length / avg total caregiving years) but grouped by age at start of *first care demand* (not caregiving spell start)."""
    first_care_demand_age = (
        df.loc[df["care_demand"] > 0]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")[["agent", "age"]]
        .rename(columns={"age": "first_care_demand_age"})
    )
    total_years = df.groupby("agent")["current_caregiving"].sum().rename("total_years")
    agents_with_total = total_years.reset_index().merge(
        first_care_demand_age, on="agent", how="inner"
    )
    out_consec = {}
    out_total = {}
    for (lo, hi), lab in zip(CARE_AGE_BINS, CARE_AGE_BIN_LABELS):
        agent_mask = agents_with_total["first_care_demand_age"].between(
            lo, hi - 1, inclusive="both"
        )
        agents_in_bin = set(agents_with_total.loc[agent_mask, "agent"])
        # Left: avg consecutive caregiving spell length (over spells of agents whose first care demand is in this bin)
        if agents_in_bin:
            spell_mask = spells_caregiving["agent"].isin(agents_in_bin)
            sub = spells_caregiving.loc[spell_mask, "spell_length"]
            out_consec[lab] = float(sub.mean()) if len(sub) > 0 else np.nan
        else:
            out_consec[lab] = np.nan
        # Right: avg total caregiving years (over those agents)
        sub_total = agents_with_total.loc[agent_mask, "total_years"]
        out_total[lab] = float(sub_total.mean()) if len(sub_total) > 0 else np.nan
    return {"consec_avg_length": out_consec, "total_avg_years": out_total}


def compute_panel_k(
    df: pd.DataFrame,
    spells_caregiving: pd.DataFrame,
) -> dict[str, Any]:
    """Panel K: Same outcomes as Panel J (avg consec caregiving spell length / avg total caregiving years) but grouped by agent's age at mother's death."""
    age_death = df.dropna(subset=["age_at_mother_death"]).drop_duplicates("agent")[
        ["agent", "age_at_mother_death"]
    ]
    total_years = df.groupby("agent")["current_caregiving"].sum().rename("total_years")
    agents_with_total = total_years.reset_index().merge(
        age_death, on="agent", how="inner"
    )
    out_consec = {}
    out_total = {}
    for (lo, hi), lab in zip(CARE_AGE_BINS, CARE_AGE_BIN_LABELS):
        agent_mask = agents_with_total["age_at_mother_death"].between(
            lo, hi - 1, inclusive="both"
        )
        agents_in_bin = set(agents_with_total.loc[agent_mask, "agent"])
        if agents_in_bin:
            spell_mask = spells_caregiving["agent"].isin(agents_in_bin)
            sub = spells_caregiving.loc[spell_mask, "spell_length"]
            out_consec[lab] = float(sub.mean()) if len(sub) > 0 else np.nan
        else:
            out_consec[lab] = np.nan
        sub_total = agents_with_total.loc[agent_mask, "total_years"]
        out_total[lab] = float(sub_total.mean()) if len(sub_total) > 0 else np.nan
    return {"consec_avg_length": out_consec, "total_avg_years": out_total}


def get_experience_at_retirement_entry(df: pd.DataFrame) -> pd.Series:
    """Experience (in years) at last period before retirement, per agent.

    Uses exp_years from the period just before the first retirement period, so
    experience is measured before it is transformed into pension points. Agents
    who never retire or retire at period 0 are not included.
    """
    if "exp_years" not in df.columns:
        return pd.Series(dtype=float)
    retirement_values = np.asarray(RETIREMENT).ravel().tolist()
    first_ret = (
        df.loc[df["choice"].isin(retirement_values)]
        .groupby("agent")["period"]
        .min()
        .rename("retirement_period")
    )
    first_ret = first_ret[first_ret >= 1]
    if first_ret.empty:
        return pd.Series(dtype=float)
    first_ret = first_ret.reset_index()
    first_ret["last_working_period"] = first_ret["retirement_period"] - 1
    merged = df.merge(
        first_ret[["agent", "last_working_period"]],
        on="agent",
        how="right",
    )
    at_entry = merged.loc[
        merged["period"] == merged["last_working_period"],
        ["agent", "exp_years"],
    ].drop_duplicates("agent")
    return at_entry.set_index("agent")["exp_years"]


def compute_panel_l(df: pd.DataFrame) -> dict[str, Any]:
    """Panel L: By first care demand age bin. Left = avg experience (years) at retirement entry; right = avg total caregiving years. Caregivers only."""
    exp_at_ret = get_experience_at_retirement_entry(df)
    if exp_at_ret.empty:
        return {
            "exp_at_retirement": {lab: np.nan for lab in CARE_AGE_BIN_LABELS},
            "total_avg_years": {lab: np.nan for lab in CARE_AGE_BIN_LABELS},
        }
    first_care_demand_age = (
        df.loc[df["care_demand"] > 0]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent", keep="first")[["agent", "age"]]
        .rename(columns={"age": "first_care_demand_age"})
    )
    total_years = df.groupby("agent")["current_caregiving"].sum().rename("total_years")
    caregivers = total_years[total_years > 0]
    agents_df = (
        caregivers.reset_index()
        .merge(first_care_demand_age, on="agent", how="inner")
        .merge(exp_at_ret.rename("exp_at_ret"), on="agent", how="left")
    )
    out_exp = {}
    out_total = {}
    for (lo, hi), lab in zip(CARE_AGE_BINS, CARE_AGE_BIN_LABELS):
        mask = agents_df["first_care_demand_age"].between(lo, hi - 1, inclusive="both")
        sub = agents_df.loc[mask]
        sub_exp = sub["exp_at_ret"].dropna()
        out_exp[lab] = float(sub_exp.mean()) if len(sub_exp) > 0 else np.nan
        out_total[lab] = float(sub["total_years"].mean()) if len(sub) > 0 else np.nan
    return {"exp_at_retirement": out_exp, "total_avg_years": out_total}


def compute_panel_m(df: pd.DataFrame) -> dict[str, Any]:
    """Panel M: By mother death age bin. Left = avg experience (years) at retirement entry; right = avg total caregiving years. Caregivers only."""
    exp_at_ret = get_experience_at_retirement_entry(df)
    if exp_at_ret.empty:
        return {
            "exp_at_retirement": {lab: np.nan for lab in CARE_AGE_BIN_LABELS},
            "total_avg_years": {lab: np.nan for lab in CARE_AGE_BIN_LABELS},
        }
    age_death = df.dropna(subset=["age_at_mother_death"]).drop_duplicates("agent")[
        ["agent", "age_at_mother_death"]
    ]
    total_years = df.groupby("agent")["current_caregiving"].sum().rename("total_years")
    caregivers = total_years[total_years > 0]
    agents_df = (
        caregivers.reset_index()
        .merge(age_death, on="agent", how="inner")
        .merge(exp_at_ret.rename("exp_at_ret"), on="agent", how="left")
    )
    out_exp = {}
    out_total = {}
    for (lo, hi), lab in zip(CARE_AGE_BINS, CARE_AGE_BIN_LABELS):
        mask = agents_df["age_at_mother_death"].between(lo, hi - 1, inclusive="both")
        sub = agents_df.loc[mask]
        sub_exp = sub["exp_at_ret"].dropna()
        out_exp[lab] = float(sub_exp.mean()) if len(sub_exp) > 0 else np.nan
        out_total[lab] = float(sub["total_years"].mean()) if len(sub) > 0 else np.nan
    return {"exp_at_retirement": out_exp, "total_avg_years": out_total}


# -----------------------------------------------------------------------------
# Run all panels and build LaTeX
# -----------------------------------------------------------------------------


def run_all_panels(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all panel statistics. df must already be prepared with age, first_* and current_caregiving."""
    spells_all, spells_light, spells_intensive = care_demand_spells(df)
    spells_caregiving = caregiving_spells(df)
    return {
        "panel_a": compute_panel_a(df),
        "panel_b": compute_panel_b(df),
        "panel_c": compute_panel_c(df),
        "panel_c_consecutive": compute_panel_c_consecutive(spells_caregiving),
        "panel_d": compute_panel_d(spells_all, spells_light, spells_intensive),
        "panel_d_total": compute_panel_d_total(df),
        "panel_e_f_g": compute_panel_e_f_g(
            spells_all, spells_light, spells_intensive, df
        ),
        "panel_e_f_g_total": compute_panel_e_f_g_total(df),
        "panel_h": compute_panel_h(spells_caregiving, spells_all, df),
        "panel_i": compute_panel_i(spells_caregiving, df),
        "panel_j": compute_panel_j(df, spells_caregiving),
        "panel_k": compute_panel_k(df, spells_caregiving),
        "panel_l": compute_panel_l(df),
        "panel_m": compute_panel_m(df),
    }


def _fmt_val_sd(val: float, sd: float) -> str:
    if np.isnan(val):
        return "---"
    if np.isnan(sd):
        return f"{val:.2f}"
    return f"{val:.2f} ({sd:.2f})"


def _fmt_val_se(val: float, se: float) -> str:
    if np.isnan(val):
        return "---"
    if np.isnan(se):
        return f"{val:.2f}"
    return f"{val:.2f} ({se:.2f})"


def _fmt_pct(x: float) -> str:
    if np.isnan(x):
        return "---"
    return f"{x:.1f}"


def _fmt_avg(x: float) -> str:
    if np.isnan(x):
        return "---"
    return f"{x:.2f}"


def build_latex_table(
    stats: dict[str, Any], label: str = "tab:describe_caregiving"
) -> str:
    """Turn panel stats into a single LaTeX table (two data columns: consecutive spells, total care demand years)."""
    a = stats["panel_a"]
    b = stats["panel_b"]
    c = stats["panel_c"]
    c_consec = stats["panel_c_consecutive"]
    d = stats["panel_d"]
    d_tot = stats["panel_d_total"]
    efg = stats["panel_e_f_g"]
    efg_tot = stats["panel_e_f_g_total"]
    h = stats["panel_h"]
    panel_i = stats["panel_i"]
    panel_j = stats["panel_j"]
    panel_k = stats["panel_k"]
    panel_l = stats["panel_l"]
    panel_m = stats["panel_m"]

    # Empty cell for single-column panels (A, B)
    empty = ""

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Care demand and informal caregiving (simulated data).}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        " & \\multicolumn{1}{c}{\\textbf{Consecutive}} & \\multicolumn{1}{c}{\\textbf{Total}} \\\\",
        " & \\multicolumn{1}{c}{\\textbf{spells}} & \\multicolumn{1}{c}{\\textbf{care demand years}} \\\\",
        "\\midrule",
        "\\textbf{Panel A: Care demand} & & \\\\",
        "\\midrule",
        f"Mean age (care demand $>0$) & {_fmt_val_sd(a['mean_age_care_demand'][0], a['mean_age_care_demand'][1])} & {empty} \\\\",
        f"Median age (care demand $>0$) & {_fmt_val_sd(a['median_age_care_demand'][0], a['median_age_care_demand'][1])} & {empty} \\\\",
        f"Mean age at start of first care demand & {_fmt_val_sd(a['mean_age_first_care_demand'][0], a['mean_age_first_care_demand'][1])} & {empty} \\\\",
        f"Median age at start of first care demand & {_fmt_val_sd(a['median_age_first_care_demand'][0], a['median_age_first_care_demand'][1])} & {empty} \\\\",
        "\\midrule",
        "\\textbf{Panel B: Informal caregiving} & & \\\\",
        "\\midrule",
        f"Mean age (caregiver) & {_fmt_val_sd(b['mean_age_caregiver'][0], b['mean_age_caregiver'][1])} & {empty} \\\\",
        f"Median age (caregiver) & {_fmt_val_sd(b['median_age_caregiver'][0], b['median_age_caregiver'][1])} & {empty} \\\\",
        f"Mean age at start of first caregiving spell & {_fmt_val_sd(b['mean_age_first_caregiving'][0], b['mean_age_first_caregiving'][1])} & {empty} \\\\",
        f"Median age at start of first caregiving spell & {_fmt_val_sd(b['median_age_first_caregiving'][0], b['median_age_first_caregiving'][1])} & {empty} \\\\",
        "\\midrule",
        "\\textbf{Panel C: Consecutive caregiving years (\\% of spells) / Total caregiving years (\\% of agents, cond. on care $>0$)} & & \\\\",
        "\\midrule",
        f"Avg. number of informal caregiving years & {_fmt_val_se(c_consec['avg_caregiving_years'][0], c_consec['avg_caregiving_years'][1])} & {_fmt_val_se(c['avg_caregiving_years'][0], c['avg_caregiving_years'][1])} \\\\",
        f"Median number of informal caregiving years & {_fmt_val_se(c_consec['median_caregiving_years'][0], c_consec['median_caregiving_years'][1])} & {_fmt_val_se(c['median_caregiving_years'][0], c['median_caregiving_years'][1])} \\\\",
        f"\\% exactly 1 year (cond. on care $>0$) & {_fmt_pct(c_consec['pct_1_year'])} & {_fmt_pct(c['pct_1_year'])} \\\\",
        f"\\% exactly 2 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_2_years'])} & {_fmt_pct(c['pct_2_years'])} \\\\",
        f"\\% exactly 3 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_3_years'])} & {_fmt_pct(c['pct_3_years'])} \\\\",
        f"\\% exactly 4 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_4_years'])} & {_fmt_pct(c['pct_4_years'])} \\\\",
        f"\\% more than 4 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_5_plus_years'])} & {_fmt_pct(c['pct_5_plus_years'])} \\\\",
        f"\\% $\\leq$ 2 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_le_2_years'])} & {_fmt_pct(c['pct_le_2_years'])} \\\\",
        f"\\% $\\leq$ 3 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_le_3_years'])} & {_fmt_pct(c['pct_le_3_years'])} \\\\",
        f"\\% $\\leq$ 4 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_le_4_years'])} & {_fmt_pct(c['pct_le_4_years'])} \\\\",
        f"\\% $\\leq$ 5 years (cond. on care $>0$) & {_fmt_pct(c_consec['pct_le_5_years'])} & {_fmt_pct(c['pct_le_5_years'])} \\\\",
        "\\midrule",
        "\\textbf{Panel D: Consecutive spell length (\\% of spells) / Total years (\\% of agents with care demand $>0$)} & & \\\\",
        "\\midrule",
        "\\textit{All care demand} & & \\\\",
        f"\\quad 1 year & {_fmt_pct(d['all']['share_1'])} & {_fmt_pct(d_tot['all']['share_1'])} \\\\",
        f"\\quad 2 years & {_fmt_pct(d['all']['share_2'])} & {_fmt_pct(d_tot['all']['share_2'])} \\\\",
        f"\\quad 3 years & {_fmt_pct(d['all']['share_3'])} & {_fmt_pct(d_tot['all']['share_3'])} \\\\",
        f"\\quad 4 years & {_fmt_pct(d['all']['share_4'])} & {_fmt_pct(d_tot['all']['share_4'])} \\\\",
        f"\\quad 5 years and longer & {_fmt_pct(d['all']['share_5plus'])} & {_fmt_pct(d_tot['all']['share_5plus'])} \\\\",
        "\\textit{Light care demand only} & & \\\\",
        f"\\quad 1 year & {_fmt_pct(d['light']['share_1'])} & {_fmt_pct(d_tot['light']['share_1'])} \\\\",
        f"\\quad 2 years & {_fmt_pct(d['light']['share_2'])} & {_fmt_pct(d_tot['light']['share_2'])} \\\\",
        f"\\quad 3 years & {_fmt_pct(d['light']['share_3'])} & {_fmt_pct(d_tot['light']['share_3'])} \\\\",
        f"\\quad 4 years & {_fmt_pct(d['light']['share_4'])} & {_fmt_pct(d_tot['light']['share_4'])} \\\\",
        f"\\quad 5 years and longer & {_fmt_pct(d['light']['share_5plus'])} & {_fmt_pct(d_tot['light']['share_5plus'])} \\\\",
        "\\textit{Intensive care demand only} & & \\\\",
        f"\\quad 1 year & {_fmt_pct(d['intensive']['share_1'])} & {_fmt_pct(d_tot['intensive']['share_1'])} \\\\",
        f"\\quad 2 years & {_fmt_pct(d['intensive']['share_2'])} & {_fmt_pct(d_tot['intensive']['share_2'])} \\\\",
        f"\\quad 3 years & {_fmt_pct(d['intensive']['share_3'])} & {_fmt_pct(d_tot['intensive']['share_3'])} \\\\",
        f"\\quad 4 years & {_fmt_pct(d['intensive']['share_4'])} & {_fmt_pct(d_tot['intensive']['share_4'])} \\\\",
        f"\\quad 5 years and longer & {_fmt_pct(d['intensive']['share_5plus'])} & {_fmt_pct(d_tot['intensive']['share_5plus'])} \\\\",
        "\\midrule",
        "\\textbf{Panel E: Avg. consecutive spell length / Avg. total years by agent's age at mother's death} & & \\\\",
        "\\midrule",
        f"Mother death before agent age 50 & {_fmt_avg(efg['all']['avg_lt_50'])} & {_fmt_avg(efg_tot['all']['avg_lt_50'])} \\\\",
        f"Mother death when agent age 50--$<60$ & {_fmt_avg(efg['all']['avg_50_60'])} & {_fmt_avg(efg_tot['all']['avg_50_60'])} \\\\",
        f"Mother death when agent age 60--$<70$ & {_fmt_avg(efg['all']['avg_60_70'])} & {_fmt_avg(efg_tot['all']['avg_60_70'])} \\\\",
        "\\midrule",
        "\\textbf{Panel F: Same, light care demand only} & & \\\\",
        "\\midrule",
        f"Mother death before agent age 50 & {_fmt_avg(efg['light']['avg_lt_50'])} & {_fmt_avg(efg_tot['light']['avg_lt_50'])} \\\\",
        f"Mother death when agent age 50--$<60$ & {_fmt_avg(efg['light']['avg_50_60'])} & {_fmt_avg(efg_tot['light']['avg_50_60'])} \\\\",
        f"Mother death when agent age 60--$<70$ & {_fmt_avg(efg['light']['avg_60_70'])} & {_fmt_avg(efg_tot['light']['avg_60_70'])} \\\\",
        "\\midrule",
        "\\textbf{Panel G: Same, intensive care demand only} & & \\\\",
        "\\midrule",
        f"Mother death before agent age 50 & {_fmt_avg(efg['intensive']['avg_lt_50'])} & {_fmt_avg(efg_tot['intensive']['avg_lt_50'])} \\\\",
        f"Mother death when agent age 50--$<60$ & {_fmt_avg(efg['intensive']['avg_50_60'])} & {_fmt_avg(efg_tot['intensive']['avg_50_60'])} \\\\",
        f"Mother death when agent age 60--$<70$ & {_fmt_avg(efg['intensive']['avg_60_70'])} & {_fmt_avg(efg_tot['intensive']['avg_60_70'])} \\\\",
        "\\midrule",
        "\\textbf{Panel H: Share of spells starting in age bucket (\\%; each column sums to 100\\%)} & & \\\\",
        "\\midrule",
    ]
    for lab in CARE_AGE_BIN_LABELS:
        lines.append(
            f"Age {lab} & {_fmt_pct(h['consec_caregiving'][lab])} & {_fmt_pct(h['care_demand'][lab])} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "\\textbf{Panel I: Avg. consecutive spell length / Avg. total caregiving years by age at spell start} & & \\\\",
            "\\midrule",
        ]
    )
    for lab in CARE_AGE_BIN_LABELS:
        lines.append(
            f"Spell start age {lab} & {_fmt_avg(panel_i['consec_avg_length'][lab])} & {_fmt_avg(panel_i['total_avg_years'][lab])} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "\\textbf{Panel J: Avg. consecutive spell length / Avg. total caregiving years by age at \\textit{first care demand} start} & & \\\\",
            "\\midrule",
        ]
    )
    for lab in CARE_AGE_BIN_LABELS:
        lines.append(
            f"First care demand age {lab} & {_fmt_avg(panel_j['consec_avg_length'][lab])} & {_fmt_avg(panel_j['total_avg_years'][lab])} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "\\textbf{Panel K: Avg. consecutive spell length / Avg. total caregiving years by age at mother's death} & & \\\\",
            "\\midrule",
        ]
    )
    for lab in CARE_AGE_BIN_LABELS:
        lines.append(
            f"Mother death age {lab} & {_fmt_avg(panel_k['consec_avg_length'][lab])} & {_fmt_avg(panel_k['total_avg_years'][lab])} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "\\textbf{Panel L: Avg. experience (years) at retirement entry / Avg. total caregiving years by age at \\textit{first care demand} start} & & \\\\",
            "\\midrule",
        ]
    )
    for lab in CARE_AGE_BIN_LABELS:
        lines.append(
            f"First care demand age {lab} & {_fmt_avg(panel_l['exp_at_retirement'][lab])} & {_fmt_avg(panel_l['total_avg_years'][lab])} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "\\textbf{Panel M: Avg. experience (years) at retirement entry / Avg. total caregiving years by age at mother's death} & & \\\\",
            "\\midrule",
        ]
    )
    for lab in CARE_AGE_BIN_LABELS:
        lines.append(
            f"Mother death age {lab} & {_fmt_avg(panel_m['exp_at_retirement'][lab])} & {_fmt_avg(panel_m['total_avg_years'][lab])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def run_describe_caregiving(
    path_to_simulated_data: Path,
    path_to_specs: Path,
    path_to_output_table: Path,
    table_label: str = "tab:describe_caregiving",
) -> None:
    """Load data, compute all panels, write LaTeX table to path_to_output_table."""
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)
    start_age = int(specs["start_age"])
    df = pd.read_pickle(path_to_simulated_data)
    df = prepare_data(df, start_age)
    df = add_first_care_demand_and_caregiving(df)
    df = add_age_at_mother_death(df)
    stats = run_all_panels(df)
    latex = build_latex_table(stats, label=table_label)
    path_to_output_table.parent.mkdir(parents=True, exist_ok=True)
    path_to_output_table.write_text(latex, encoding="utf-8")
