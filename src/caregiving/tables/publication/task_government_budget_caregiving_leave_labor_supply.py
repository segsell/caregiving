"""Publication: labor supply by age (full vs normal caregiving leave scenarios)."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.counterfactual.plotting_utils import prepare_dfs_for_comparison
from caregiving.model.shared import FULL_TIME, PART_TIME, UNEMPLOYED, WORK

MAX_AGE_PLOT = 90


def _share_by_age(
    df: pd.DataFrame,
    choice_array: np.ndarray,
    start_age: int,
    name: str,
) -> pd.Series:
    """Compute share of choice in choice_array by age (mean of indicator)."""
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "choice" not in df.columns:
        return pd.Series(dtype=float)
    if "age" not in df.columns:
        df["age"] = start_age + df["period"]
    choices = np.asarray(choice_array)
    df["indicator"] = df["choice"].isin(choices).astype(np.float64)
    return (
        df.groupby("age", observed=False)["indicator"].mean().rename(name).sort_index()
    )


@pytask.mark.tables
@pytask.mark.publication
def task_labor_supply_by_age_caregiving_leave_full_vs_normal(  # noqa: PLR0915
    path_to_full_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_normal_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_ft_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_full_time_share_by_age_levels.pdf",
    path_to_save_ft_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_full_time_share_by_age_difference.pdf",
    path_to_save_pt_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_part_time_share_by_age_levels.pdf",
    path_to_save_pt_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_part_time_share_by_age_difference.pdf",
    path_to_save_unemp_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_unemployed_share_by_age_levels.pdf",
    path_to_save_unemp_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_unemployed_share_by_age_difference.pdf",
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare labor supply (FT, PT, unemployed share) by age: full vs normal leave.

    Produces 6 plots:
    - Full-time share by age (levels: two lines; difference: full minus normal).
    - Part-time share by age (levels: two lines; difference: full minus normal).
    - Unemployed share by age (levels: two lines; difference: full minus normal).
    """
    full_df = pd.read_pickle(path_to_full_policy_sim)
    normal_df = pd.read_pickle(path_to_normal_policy_sim)
    full_df, normal_df = prepare_dfs_for_comparison(
        full_df,
        normal_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    start_age = int(specs.get("start_age", 30))

    full_ft = _share_by_age(full_df, FULL_TIME, start_age, "full_leave")
    normal_ft = _share_by_age(normal_df, FULL_TIME, start_age, "normal_leave")
    full_pt = _share_by_age(full_df, PART_TIME, start_age, "full_leave")
    normal_pt = _share_by_age(normal_df, PART_TIME, start_age, "normal_leave")
    full_unemp = _share_by_age(full_df, UNEMPLOYED, start_age, "full_leave")
    normal_unemp = _share_by_age(normal_df, UNEMPLOYED, start_age, "normal_leave")

    age_index = np.arange(start_age, MAX_AGE_PLOT + 1, dtype=int)

    def _to_levels(s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
        out = pd.concat([s1, s2], axis=1)
        out = out.reindex(age_index).fillna(0)
        out.columns = ["full_leave", "normal_leave"]
        return out

    ft_levels = _to_levels(full_ft, normal_ft)
    pt_levels = _to_levels(full_pt, normal_pt)
    unemp_levels = _to_levels(full_unemp, normal_unemp)

    path_to_save_ft_levels.parent.mkdir(parents=True, exist_ok=True)

    # --- Full-time: levels and difference ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        ft_levels.index, ft_levels["full_leave"], label="Full leave", color="C0", lw=2
    )
    ax.plot(
        ft_levels.index,
        ft_levels["normal_leave"],
        label="Normal leave",
        color="C1",
        lw=2,
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Full-time share")
    ax.set_title("Full-time share by age (full vs normal caregiving leave)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path_to_save_ft_levels)
    plt.close(fig)

    diff_ft = ft_levels["full_leave"] - ft_levels["normal_leave"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diff_ft.index, diff_ft.values, color="C2", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Age")
    ax.set_ylabel("Difference in full-time share (full minus normal)")
    ax.set_title("Full-time share difference by age")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_to_save_ft_difference)
    plt.close(fig)

    # --- Part-time: levels and difference ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        pt_levels.index, pt_levels["full_leave"], label="Full leave", color="C0", lw=2
    )
    ax.plot(
        pt_levels.index,
        pt_levels["normal_leave"],
        label="Normal leave",
        color="C1",
        lw=2,
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Part-time share")
    ax.set_title("Part-time share by age (full vs normal caregiving leave)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path_to_save_pt_levels)
    plt.close(fig)

    diff_pt = pt_levels["full_leave"] - pt_levels["normal_leave"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diff_pt.index, diff_pt.values, color="C2", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Age")
    ax.set_ylabel("Difference in part-time share (full minus normal)")
    ax.set_title("Part-time share difference by age")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_to_save_pt_difference)
    plt.close(fig)

    # --- Unemployed: levels and difference ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        unemp_levels.index,
        unemp_levels["full_leave"],
        label="Full leave",
        color="C0",
        lw=2,
    )
    ax.plot(
        unemp_levels.index,
        unemp_levels["normal_leave"],
        label="Normal leave",
        color="C1",
        lw=2,
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Unemployed share")
    ax.set_title("Unemployed share by age (full vs normal caregiving leave)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path_to_save_unemp_levels)
    plt.close(fig)

    diff_unemp = unemp_levels["full_leave"] - unemp_levels["normal_leave"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diff_unemp.index, diff_unemp.values, color="C2", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Age")
    ax.set_ylabel("Difference in unemployed share (full minus normal)")
    ax.set_title("Unemployed share difference by age")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_to_save_unemp_difference)
    plt.close(fig)


@pytask.mark.tables
@pytask.mark.publication
def task_working_share_by_age_caregiving_leave_full_vs_normal(
    path_to_full_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_full_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_normal_policy_sim: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_levels: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_working_share_by_age_levels.pdf",
    path_to_save_difference: Annotated[Path, Product] = BLD
    / "tables"
    / "publication"
    / "caregiving_leave_working_share_by_age_difference.pdf",
    restrict_same_agents: bool = True,
    ever_care_demand: bool = True,
) -> None:
    """Compare share working (PT + FT) by age: full vs normal caregiving leave.

    Produces two plots:
    - Working share by age (levels: two lines; difference: full minus normal).
    """
    full_df = pd.read_pickle(path_to_full_policy_sim)
    normal_df = pd.read_pickle(path_to_normal_policy_sim)
    full_df, normal_df = prepare_dfs_for_comparison(
        full_df,
        normal_df,
        ever_care_demand=ever_care_demand,
        restrict_same_agents=restrict_same_agents,
    )
    specs = pickle.load(path_to_specs.open("rb"))
    start_age = int(specs.get("start_age", 30))

    full_working = _share_by_age(full_df, WORK, start_age, "full_leave")
    normal_working = _share_by_age(normal_df, WORK, start_age, "normal_leave")

    age_index = np.arange(start_age, MAX_AGE_PLOT + 1, dtype=int)
    levels = pd.concat([full_working, normal_working], axis=1)
    levels = levels.reindex(age_index).fillna(0)
    levels.columns = ["full_leave", "normal_leave"]

    path_to_save_levels.parent.mkdir(parents=True, exist_ok=True)

    # --- Levels: share working (PT + FT) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels.index, levels["full_leave"], label="Full leave", color="C0", lw=2)
    ax.plot(
        levels.index,
        levels["normal_leave"],
        label="Normal leave",
        color="C1",
        lw=2,
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Working share (PT + FT)")
    ax.set_title("Working share by age (full vs normal caregiving leave)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path_to_save_levels)
    plt.close(fig)

    # --- Difference: full minus normal ---
    diff = levels["full_leave"] - levels["normal_leave"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diff.index, diff.values, color="C2", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Age")
    ax.set_ylabel("Difference in working share (full minus normal)")
    ax.set_title("Working share difference by age")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_to_save_difference)
    plt.close(fig)
