"""Plot differences in labor supply by age between scenarios.

Original and no-care-demand scenario.

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from linearmodels.panel import PanelOLS
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    NOT_WORKING,
    PART_TIME,
    RETIREMENT,
    SEX,
    UNEMPLOYED,
    WORK,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    NOT_WORKING_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)


@pytask.mark.counterfactual_differences
def task_plot_labor_supply_differences(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_differences_by_age.png",
    path_to_plot_percentage_deviations: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_percentage_deviations_by_age.png",
    ever_caregivers: bool = True,
) -> None:
    """Plot differences in labor supply by age between scenarios."""

    # Load data
    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure an 'agent' column exists (source data often indexed by
    # MultiIndex agent, period)
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute labor supply shares by age for both scenarios
    original_shares = compute_labor_supply_shares_by_age(df_original, is_original=True)
    no_care_demand_shares = compute_labor_supply_shares_by_age(
        df_no_care_demand, is_original=False
    )

    # Compute differences
    differences = compute_labor_supply_differences(
        original_shares, no_care_demand_shares
    )
    # Create plot
    create_labor_supply_difference_plot(differences, path_to_plot)

    # # Compute percentage deviations
    # pct_devs = compute_labor_supply_percentage_deviations(
    #     original_shares, no_care_demand_shares
    # )

    # # Plot
    # create_labor_supply_percentage_plot(pct_devs, path_to_plot_percentage_deviations)


def compute_labor_supply_shares_by_age(
    df: pd.DataFrame, is_original: bool = True
) -> pd.DataFrame:
    """Compute labor supply shares by age using value_counts approach.

    Mirrors the approach in the model-fit plotting: map raw choices into
    4 aggregate groups (retired, unemployed, part-time, full-time), then
    compute age-specific normalized counts.
    """

    # Ensure required columns
    df_local = df[["choice", "age"]].copy()

    if is_original:
        choice_groups = {
            0: np.asarray(RETIREMENT).ravel().tolist(),
            1: np.asarray(UNEMPLOYED).ravel().tolist(),
            2: np.asarray(PART_TIME).ravel().tolist(),
            3: np.asarray(FULL_TIME).ravel().tolist(),
        }
    else:
        choice_groups = {
            0: np.asarray(RETIREMENT_NO_CARE_DEMAND).ravel().tolist(),
            1: np.asarray(UNEMPLOYED_NO_CARE_DEMAND).ravel().tolist(),
            2: np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist(),
            3: np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist(),
        }

    # Build a flat map from raw codes to aggregate group id
    code_to_group = {}
    for group_id, codes in choice_groups.items():
        for code in codes:
            code_to_group[code] = group_id

    # Map raw choice to group id
    df_local["choice_group"] = (
        df_local["choice"].map(code_to_group).fillna(0).astype(int)
    )

    # Compute normalized shares per age
    shares_by_age = (
        df_local.groupby("age", observed=False)["choice_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns={0: "retired", 1: "unemployed", 2: "part_time", 3: "full_time"})
        .reset_index()
    )

    # Construct output with desired columns
    out = pd.DataFrame(
        {
            "age": shares_by_age["age"],
            "is_full_time": shares_by_age.get("full_time", 0),
            "is_part_time": shares_by_age.get("part_time", 0),
        }
    )
    out["is_working"] = out["is_full_time"] + out["is_part_time"]
    out["is_not_working"] = 1 - out["is_working"]

    return out


def compute_labor_supply_differences(
    original_shares: pd.DataFrame, no_care_demand_shares: pd.DataFrame
) -> pd.DataFrame:
    """Compute differences in labor supply shares between scenarios."""

    # Merge on age only (assuming single sex model)
    merged = pd.merge(
        original_shares,
        no_care_demand_shares,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    # Compute differences (no_care_demand - original)
    merged["working_diff"] = (
        merged["is_working_no_care_demand"] - merged["is_working_original"]
    )
    merged["part_time_diff"] = (
        merged["is_part_time_no_care_demand"] - merged["is_part_time_original"]
    )
    merged["full_time_diff"] = (
        merged["is_full_time_no_care_demand"] - merged["is_full_time_original"]
    )

    return merged


def create_labor_supply_difference_plot(
    differences: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing labor supply differences by age."""

    # Create figure with subplots (single row since only women)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Labor Supply Differences by Age: No Care Demand vs Original (Women)",
        fontsize=16,
    )

    # Plot 1: Working (any employment)
    axes[0].plot(differences["age"], differences["working_diff"], "b-", linewidth=2)
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_title("Working Rate Difference")
    axes[0].set_ylabel("Difference (No Care - Original)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Part-time vs Full-time
    axes[1].plot(
        differences["age"],
        differences["part_time_diff"],
        "g-",
        linewidth=2,
        label="Part-time",
    )
    axes[1].plot(
        differences["age"],
        differences["full_time_diff"],
        "r-",
        linewidth=2,
        label="Full-time",
    )
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_title("Part-time vs Full-time Differences")
    axes[1].set_ylabel("Difference (No Care - Original)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Calculate common y-axis range for both plots
    working_diff = differences["working_diff"].values
    part_time_diff = differences["part_time_diff"].values
    full_time_diff = differences["full_time_diff"].values

    # Find the overall min and max across all difference series
    all_values = np.concatenate([working_diff, part_time_diff, full_time_diff])
    common_min = np.min(all_values)
    common_max = np.max(all_values)

    # Add padding
    common_range = common_max - common_min
    padding = common_range * 0.1
    common_y_min = common_min - padding
    common_y_max = common_max + padding

    # Set x-axis properties and common y-axis range
    axes[0].set_xlabel("Age")
    axes[0].set_xlim(30, 70)
    axes[0].set_ylim(common_y_min, common_y_max)

    axes[1].set_xlabel("Age")
    axes[1].set_xlim(30, 70)
    axes[1].set_ylim(common_y_min, common_y_max)

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Labor supply difference plot saved to: {path_to_plot}")


def compute_labor_supply_percentage_deviations(
    original_shares: pd.DataFrame, no_care_demand_shares: pd.DataFrame
) -> pd.DataFrame:
    """Compute percentage deviations: (no_care - original) / original.

    Safely handles division by zero by leaving results at 0 when the
    original share is 0.
    """

    merged = pd.merge(
        original_shares,
        no_care_demand_shares,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    def pct_diff(num, den):
        out = np.zeros_like(num)
        mask = den != 0
        out[mask] = (num[mask] - den[mask]) / den[mask]
        return out

    merged["working_pct_diff"] = pct_diff(
        merged["is_working_no_care_demand"].to_numpy(),
        merged["is_working_original"].to_numpy(),
    )
    merged["part_time_pct_diff"] = pct_diff(
        merged["is_part_time_no_care_demand"].to_numpy(),
        merged["is_part_time_original"].to_numpy(),
    )
    merged["full_time_pct_diff"] = pct_diff(
        merged["is_full_time_no_care_demand"].to_numpy(),
        merged["is_full_time_original"].to_numpy(),
    )

    return merged


def create_labor_supply_percentage_plot(
    pct_devs: pd.DataFrame, path_to_plot: Path
) -> None:
    """Create plot showing percentage deviations by age.

    Percentage deviations are shown for working, part-time and full-time.
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Labor Supply Percentage Deviations by Age: No Care vs Original (Women)",
        fontsize=16,
    )

    # Working percentage deviation
    axes[0].plot(pct_devs["age"], pct_devs["working_pct_diff"], "b-", linewidth=2)
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_title("Working Rate Percentage Deviation")
    axes[0].set_ylabel("(No Care - Original) / Original")
    axes[0].grid(True, alpha=0.3)

    # Part-time vs Full-time percentage deviations
    axes[1].plot(
        pct_devs["age"],
        pct_devs["part_time_pct_diff"],
        "g-",
        linewidth=2,
        label="Part-time",
    )
    axes[1].plot(
        pct_devs["age"],
        pct_devs["full_time_pct_diff"],
        "r-",
        linewidth=2,
        label="Full-time",
    )
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_title("Part-time vs Full-time Percentage Deviations")
    axes[1].set_ylabel("(No Care - Original) / Original")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Set x-range consistent with other plots
    axes[0].set_xlabel("Age")
    axes[0].set_xlim(30, 67)
    axes[1].set_xlabel("Age")
    axes[1].set_xlim(30, 67)

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


def create_labor_supply_age_profile_plot(
    df_original: pd.DataFrame,
    df_no_care_demand: pd.DataFrame,
    path_to_plot: Path,
) -> None:
    """Plot FT, PT and Not Working age profiles for both scenarios in one plot."""

    df_original["sex"] = SEX
    df_no_care_demand["sex"] = SEX

    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    shares_original = compute_labor_supply_shares_by_age(df_original, is_original=True)
    shares_ncd = compute_labor_supply_shares_by_age(
        df_no_care_demand, is_original=False
    )

    merged = pd.merge(
        shares_original,
        shares_ncd,
        on=["age"],
        suffixes=("_original", "_no_care_demand"),
    )

    plt.figure(figsize=(12, 7))
    plt.title("Labor Supply Age Profiles: Original vs No Care Demand")

    # Full-time
    plt.plot(
        merged["age"],
        merged["is_full_time_original"],
        label="Full-time (Original)",
        color="tab:red",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_full_time_no_care_demand"],
        label="Full-time (No Care)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )

    # Part-time
    plt.plot(
        merged["age"],
        merged["is_part_time_original"],
        label="Part-time (Original)",
        color="tab:green",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_part_time_no_care_demand"],
        label="Part-time (No Care)",
        color="tab:green",
        linestyle="--",
        linewidth=2,
    )

    # Not working (unemployed or retired)
    plt.plot(
        merged["age"],
        merged["is_not_working_original"],
        label="Not working (Original)",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        merged["age"],
        merged["is_not_working_no_care_demand"],
        label="Not working (No Care)",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Age")
    plt.ylabel("Share")
    plt.xlim(30, 70)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
def task_plot_labor_supply_age_profiles(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "labor_supply_age_profiles.png",
    ever_caregivers: bool = True,
) -> None:
    """Task: plot FT, PT, Not Working age profiles for both scenarios."""

    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Ensure 'agent' exists
    if "agent" not in df_original.columns:
        if isinstance(df_original.index, pd.MultiIndex) and (
            "agent" in df_original.index.names
        ):
            df_original = df_original.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_original = df_original.reset_index()

    if "agent" not in df_no_care_demand.columns:
        if isinstance(df_no_care_demand.index, pd.MultiIndex) and (
            "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index(
                level=["agent"]
            )  # keep period indexed
        else:
            df_no_care_demand = df_no_care_demand.reset_index()

    # Optional sample restriction to ever-caregivers in the original scenario
    if ever_caregivers:
        informal_care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(informal_care_codes), "agent"
        ].unique()

        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    create_labor_supply_age_profile_plot(
        df_original=df_original,
        df_no_care_demand=df_no_care_demand,
        path_to_plot=path_to_plot,
    )


# # ===================================================================================
# # Event study
# # ===================================================================================


# def _prepare_event_study_panel(
#     df_original: pd.DataFrame,
#     df_no_care_demand: pd.DataFrame,
#     window: int = 16,
#     ever_caregivers: bool = True,
# ) -> pd.DataFrame:
#     """Build stacked panel with event time around first caregiving start.

#     - Computes first caregiving start in original per agent.
#     - Attaches the same event time to the counterfactual rows.
#     - Creates binary outcome `work`.
#     - Trims to event time in [-window, window] and drops missing timing.
#     """

#     # Ensure 'agent' and 'period' columns exist
#     for df in (df_original, df_no_care_demand):
#         # Ensure agent
#         if "agent" not in df.columns:
#             if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
#                 df.reset_index(level=["agent"], inplace=True)
#             else:
#                 df.reset_index(inplace=True)
#         # Ensure period
#         if "period" not in df.columns:
#             if isinstance(df.index, pd.MultiIndex) and ("period" in df.index.names):
#                 df.reset_index(level=["period"], inplace=True)
#             else:
#                 if "age" in df.columns:
#                     # Derive a pseudo-period anchored at each agent's min age
#                     df["period"] = df.groupby("agent")["age"].transform(
#                         lambda s: s - s.min()
#                     )
#                 else:
#                     # Fallback: sequential period within agent
#                     df["period"] = df.groupby("agent").cumcount()

#     # After ensuring columns, fully reset any residual index to avoid
#     # label/level ambiguity
#     df_original = df_original.reset_index(drop=True)
#     df_no_care_demand = df_no_care_demand.reset_index(drop=True)

#     # Identify first caregiving start in original
#     informal = np.asarray(INFORMAL_CARE).ravel().tolist()
#     caregiving_mask = df_original["choice"].isin(informal)
#     first_care = (
#         df_original.loc[caregiving_mask, ["agent", "period"]]
#         .sort_values(["agent", "period"])
#         .drop_duplicates("agent")
#         .rename(columns={"period": "treat_start"})
#     )

#     # Optional: restrict to ever caregivers
#     if ever_caregivers:
#         caregiver_ids = first_care["agent"].unique()
#         df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
#         df_no_care_demand = df_no_care_demand[
#             df_no_care_demand["agent"].isin(caregiver_ids)
#         ].copy()

#     # Attach treat_start to both datasets
#     df_o = df_original.merge(first_care, on="agent", how="left")
#     df_c = df_no_care_demand.merge(first_care, on="agent", how="left")

#     # Compute event time
#     for d in (df_o, df_c):
#         d["event_time"] = d["period"] - d["treat_start"]

#     # Outcome: work
#     df_o["work"] = df_o["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(int)
#     df_c["work"] = (
#         df_c["choice"]
#         .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
#         .astype(int)
#     )

#     # Stack and keep within window, drop rows without timing
#     panel = pd.concat(
#         [
#             df_o.assign(scenario="original"),
#             df_c.assign(scenario="no_care"),
#         ],
#         ignore_index=True,
#     )

#     panel = panel[(panel["treat_start"].notna())].copy()
#     panel = panel[(panel["event_time"] >= -window) & (panel["event_time"] <= window)]

#     return panel


# def _event_study_twfe(panel: pd.DataFrame, window: int = 16) -> pd.DataFrame:
#     """Estimate TWFE event-study without baseline (-1) using OLS with dummies.

#     Returns a DataFrame with columns: event_time, beta.
#     """
#     # Build design: dummies for event_time (exclude -1), individual FE, period FE
#     evt_vals = list(range(-window, window + 1))
#     baseline = -1
#     evt_used = [k for k in evt_vals if k != baseline]

#     # Dummies
#     D_evt = pd.get_dummies(panel["event_time"]).reindex(
#         columns=evt_used, fill_value=0
#     )
#     D_ind = pd.get_dummies(panel["agent"], drop_first=True)
#     D_t = pd.get_dummies(panel["period"], drop_first=True)

#     X = pd.concat([D_evt, D_ind, D_t], axis=1).astype(float).values
#     y = panel["work"].astype(float).values

#     # OLS via least squares
#     beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

#     # First len(evt_used) coefficients correspond to event-time effects
#     beta_evt = beta_hat[: len(evt_used)]
#     out = pd.DataFrame({"event_time": evt_used, "beta": beta_evt})

#     # Add baseline at zero for plotting convenience
#     out = pd.concat(
#         [out, pd.DataFrame({"event_time": [baseline], "beta": [0.0]})],
#         ignore_index=True,
#     ).sort_values("event_time")
#     return out


# def create_event_study_plot(estimates: pd.DataFrame, path_to_plot: Path) -> None:
#     """Plot dynamic DiD event-study coefficients from -window to +window."""
#     plt.figure(figsize=(12, 6))
#     plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
#     plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
#     plt.plot(estimates["event_time"], estimates["beta"], marker="o")
#     plt.xlabel("Event time (years from first caregiving start)")
#     plt.ylabel("Effect on working (share)")
#     plt.xlim(-16, 16)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
#     plt.close()


# def _event_study_twfe_lm(panel: pd.DataFrame, window: int = 16) -> pd.DataFrame:
#     """Estimate TWFE event-study using linearmodels PanelOLS.

#     Uses a categorical for event_time with baseline -1, and includes
#     entity (agent) and time (period) fixed effects.
#     """

#     df = panel.copy()
#     # Ensure required index
#     if not ("agent" in df.columns and "period" in df.columns):
#         raise ValueError("Panel must contain 'agent' and 'period' columns.")

#     # Limit to window and drop missing timing
#     df = df[df["treat_start"].notna()].copy()
#     df = df[(df["event_time"] >= -window) & (df["event_time"] <= window)].copy()

#     # Categorical event-time with baseline -1
#     df["event_time_cat"] = df["event_time"].astype(int)

#     # Build a MultiIndex (entity=time index order is important: time x entity)
#     df = df.set_index(["period", "agent"]).sort_index()

#     # Fit PanelOLS with entity and time effects; use formula with treatment coding
#     # Patsy-compatible: C(event_time_cat, Treatment(reference=-1))
#     mod = PanelOLS.from_formula(
#         "work ~ 0 + C(event_time_cat, Treatment(reference=-1)) + "
#         "EntityEffects + TimeEffects",
#         data=df,
#         drop_absorbed=True,
#         check_rank=True,
#     )
#     res = mod.fit()

#     # Extract event-time coefficients
#     params = res.params
#     evt_coefs = []
#     for name, val in params.items():
#         if name.startswith("C(event_time_cat, Treatment(reference=-1))["):
#             # name like C(event_time_cat, Treatment(reference=-1))[T.-16]
#             try:
#                 key = name.split("[")[-1].strip("]")
#                 # keys look like 'T.-16' or 'T.0' -> take after 'T.'
#                 if key.startswith("T."):
#                     key = key[2:]
#                 evt = int(key)
#                 evt_coefs.append((evt, float(val)))
#             except Exception:
#                 continue

#     out = pd.DataFrame(evt_coefs, columns=["event_time", "beta"]).sort_values(
#         "event_time"
#     )
#     # Ensure baseline -1 present at 0
#     if (-1) not in set(out["event_time"].tolist()):
#         out = pd.concat(
#             [out, pd.DataFrame({"event_time": [-1], "beta": [0.0]})],
#             ignore_index=True,
#         ).sort_values("event_time")
#     return out


# # @pytask.mark.counterfactual_differences
# def event_study_work(
#     path_to_original_data: Path = BLD
#     / "solve_and_simulate"
#     / "simulated_data_estimated_params.pkl",
#     path_to_no_care_demand_data: Path = BLD
#     / "solve_and_simulate"
#     / "simulated_data_no_care_demand.pkl",
#     path_to_plot: Annotated[Path, Product] = BLD
#     / "plots"
#     / "counterfactual"
#     / "event_study_work.png",
#     ever_caregivers: bool = True,
#     window: int = 16,
# ) -> None:
#     """Run and plot event-study (TWFE) using counterfactual controls.

#     - Treatment timing: first caregiving start in the original scenario.
#     - Controls: same agents in the no-care-demand model (true counterfactual).
#     - FE: individual and period fixed effects.
#     - Outcome: working indicator.
#     """

#     df_original = pd.read_pickle(path_to_original_data)
#     df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

#     # Remove dead periods to focus on active horizon
#     df_original = df_original[df_original["health"] != DEAD].copy()
#     df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

#     panel = _prepare_event_study_panel(
#         df_original=df_original,
#         df_no_care_demand=df_no_care_demand,
#         window=window,
#         ever_caregivers=ever_caregivers,
#     )

#     est = _event_study_twfe(panel, window=window)
#     create_event_study_plot(est, path_to_plot)


# ===================================================================================
# Distance to first care spell profiles
# ===================================================================================


def _ensure_agent_period(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that 'agent' and 'period' are columns (not index levels)."""
    if "agent" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("agent" in df.index.names):
            df = df.reset_index(level=["agent"])  # keep period indexed if present
        else:
            df = df.reset_index()
    if "period" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and ("period" in df.index.names):
            df = df.reset_index(level=["period"])  # keep any other levels
        else:
            if "age" in df.columns:
                df["period"] = df.groupby("agent")["age"].transform(
                    lambda s: s - s.min()
                )
            else:
                df["period"] = df.groupby("agent").cumcount()
    return df


def _add_distance_to_first_care(df_original: pd.DataFrame) -> pd.DataFrame:
    """Add distance_to_first_care column to original data where 0 is first care."""
    # Flatten any existing index to avoid column/index name ambiguity
    df = df_original.reset_index(drop=True)
    df = _ensure_agent_period(df)
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    caregiving_mask = df["choice"].isin(care_codes)
    first_care = (
        df.loc[caregiving_mask, ["agent", "period"]]
        .sort_values(["agent", "period"])
        .drop_duplicates("agent")
        .rename(columns={"period": "first_care_period"})
    )
    out = df.merge(first_care, on="agent", how="left")
    out["distance_to_first_care"] = out["period"] - out["first_care_period"]
    return out


def create_outcome_profiles_by_distance_plot(
    df_original_with_dist: pd.DataFrame,
    df_no_care_demand_with_dist: pd.DataFrame,
    path_to_plot: Path,
    x_window: int = 16,
) -> None:
    """Plot mean work, ft, pt by distance_to_first_care for both scenarios (6 lines)."""

    # Build indicators
    work_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(WORK).ravel().tolist())
        .astype(float)
    )
    ft_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(FULL_TIME).ravel().tolist())
        .astype(float)
    )
    pt_o = (
        df_original_with_dist["choice"]
        .isin(np.asarray(PART_TIME).ravel().tolist())
        .astype(float)
    )

    work_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    ft_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    pt_c = (
        df_no_care_demand_with_dist["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    df_o = pd.DataFrame(
        {
            "distance": df_original_with_dist["distance_to_first_care"],
            "work": work_o,
            "ft": ft_o,
            "pt": pt_o,
        }
    )
    df_c = pd.DataFrame(
        {
            "distance": df_no_care_demand_with_dist["distance_to_first_care"],
            "work": work_c,
            "ft": ft_c,
            "pt": pt_c,
        }
    )

    # Restrict window if requested
    df_o = df_o[(df_o["distance"] >= -x_window) & (df_o["distance"] <= x_window)]
    df_c = df_c[(df_c["distance"] >= -x_window) & (df_c["distance"] <= x_window)]

    prof_o = df_o.groupby("distance", observed=False).mean().reset_index()
    prof_c = df_c.groupby("distance", observed=False).mean().reset_index()

    plt.figure(figsize=(12, 7))
    plt.title("Outcomes by Distance to First Care Spell (Original vs No Care)")

    # six lines
    plt.plot(
        prof_o["distance"],
        prof_o["work"],
        label="Work (Original)",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["work"],
        label="Work (No Care)",
        color="tab:blue",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        prof_o["distance"],
        prof_o["ft"],
        label="FT (Original)",
        color="tab:red",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["ft"],
        label="FT (No Care)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        prof_o["distance"],
        prof_o["pt"],
        label="PT (Original)",
        color="tab:green",
        linewidth=2,
    )
    plt.plot(
        prof_c["distance"],
        prof_c["pt"],
        label="PT (No Care)",
        color="tab:green",
        linestyle="--",
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Distance to first care spell (periods)")
    plt.ylabel("Mean outcome")
    plt.xlim(-x_window, x_window)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences
def task_plot_outcomes_by_distance_to_first_care(
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "outcomes_by_distance_to_first_care.png",
    ever_caregivers: bool = True,
    window: int = 16,
) -> None:
    """Create distance_to_first_care and plot mean outcomes by distance (6 lines)."""

    df_original = pd.read_pickle(path_to_original_data)
    df_no_care_demand = pd.read_pickle(path_to_no_care_demand_data)

    # Restrict to alive periods
    df_original = df_original[df_original["health"] != DEAD].copy()
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Ensure agent/period exist
    df_original = _ensure_agent_period(df_original)
    df_no_care_demand = _ensure_agent_period(df_no_care_demand)

    # Optional restriction to ever-caregivers
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_original.loc[
            df_original["choice"].isin(care_codes), "agent"
        ].unique()
        df_original = df_original[df_original["agent"].isin(caregiver_ids)].copy()
        df_no_care_demand = df_no_care_demand[
            df_no_care_demand["agent"].isin(caregiver_ids)
        ].copy()

    # Compute distance in original and copy to counterfactual
    df_original = _add_distance_to_first_care(df_original)
    # Merge the per-agent first_care_period and compute distance in counterfactual
    dist_map = (
        df_original.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    df_no_care_demand = df_no_care_demand.merge(dist_map, on="agent", how="left")
    df_no_care_demand["distance_to_first_care"] = (
        df_no_care_demand["period"] - df_no_care_demand["first_care_period"]
    )

    # Plot
    create_outcome_profiles_by_distance_plot(
        df_original_with_dist=df_original,
        df_no_care_demand_with_dist=df_no_care_demand,
        path_to_plot=path_to_plot,
        x_window=window,
    )


@pytask.mark.counterfactual_differences_matched
def task_plot_matched_differences_by_distance(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_by_distance.png",
    ever_caregivers: bool = True,
    window: int = 20,
) -> None:
    """Compute matched period differences (orig - no-care), then average by distance.

    Steps:
      1) Restrict to alive and (optionally) ever-caregivers.
      2) Ensure agent/period columns.
      3) Build per-period outcomes (work, ft, pt) for both scenarios.
      4) Merge on (agent, period) and compute differences.
      5) Compute distance_to_first_care from original, attach to merged.
      6) Average diffs by distance and plot three series.
    """

    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()

    # Ensure agent/period
    df_o = _ensure_agent_period(df_o)
    df_c = _ensure_agent_period(df_c)

    # Fully flatten any residual index levels named 'agent' or 'period'
    if isinstance(df_o.index, pd.MultiIndex):
        idx_names_o = {n for n in df_o.index.names if n is not None}
        if ("agent" in idx_names_o) or ("period" in idx_names_o):
            df_o = df_o.reset_index()
    if isinstance(df_c.index, pd.MultiIndex):
        idx_names_c = {n for n in df_c.index.names if n is not None}
        if ("agent" in idx_names_c) or ("period" in idx_names_c):
            df_c = df_c.reset_index()

    # Ensure no index name collisions remain (fully flatten)
    df_o = df_o.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)

    # Ever-caregiver restriction
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_o.loc[df_o["choice"].isin(care_codes), "agent"].unique()
        df_o = df_o[df_o["agent"].isin(caregiver_ids)].copy()
        df_c = df_c[df_c["agent"].isin(caregiver_ids)].copy()

    # Outcomes per period
    o_work = df_o["choice"].isin(np.asarray(WORK).ravel().tolist()).astype(float)
    o_ft = df_o["choice"].isin(np.asarray(FULL_TIME).ravel().tolist()).astype(float)
    o_pt = df_o["choice"].isin(np.asarray(PART_TIME).ravel().tolist()).astype(float)

    c_work = (
        df_c["choice"]
        .isin(np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_ft = (
        df_c["choice"]
        .isin(np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_pt = (
        df_c["choice"]
        .isin(np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    o_cols = df_o[["agent", "period"]].copy()
    o_cols["work_o"] = o_work
    o_cols["ft_o"] = o_ft
    o_cols["pt_o"] = o_pt

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["work_c"] = c_work
    c_cols["ft_c"] = c_ft
    c_cols["pt_c"] = c_pt

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_work"] = merged["work_o"] - merged["work_c"]
    merged["diff_ft"] = merged["ft_o"] - merged["ft_c"]
    merged["diff_pt"] = merged["pt_o"] - merged["pt_c"]

    # Compute distance in original and attach
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[
            ["diff_work", "diff_ft", "diff_pt"]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Plot
    plt.figure(figsize=(12, 7))
    plt.title("Matched Differences by Distance to First Care Spell")

    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_work"],
        label="Work (Orig - No Care)",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_ft"],
        label="FT (Orig - No Care)",
        color="tab:red",
        linewidth=2,
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_pt"],
        label="PT (Orig - No Care)",
        color="tab:green",
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Distance to first care spell (periods)")
    plt.ylabel("Mean difference (Original - No Care)")
    plt.xlim(-window, window)
    plt.ylim(-0.125, 0.025)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


@pytask.mark.counterfactual_differences_matched
def task_plot_matched_differences_by_distance_no_work(  # noqa: PLR0915
    path_to_original_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_no_care_demand_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "matched_differences_by_distance_no_work.png",
    ever_caregivers: bool = True,
    window: int = 16,
) -> None:
    """Matched differences for no-work outcomes: unemployed, retired, not working."""

    # Load
    df_o = pd.read_pickle(path_to_original_data)
    df_c = pd.read_pickle(path_to_no_care_demand_data)

    # Alive restriction
    df_o = df_o[df_o["health"] != DEAD].copy()
    df_c = df_c[df_c["health"] != DEAD].copy()

    # Ensure agent/period and flatten any index
    df_o = _ensure_agent_period(df_o)
    df_c = _ensure_agent_period(df_c)
    df_o = df_o.reset_index(drop=True)
    df_c = df_c.reset_index(drop=True)

    # Ever-caregiver restriction
    if ever_caregivers:
        care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
        caregiver_ids = df_o.loc[df_o["choice"].isin(care_codes), "agent"].unique()
        df_o = df_o[df_o["agent"].isin(caregiver_ids)].copy()
        df_c = df_c[df_c["agent"].isin(caregiver_ids)].copy()

    # No-work outcomes per period (0/1)
    o_unemployed = (
        df_o["choice"].isin(np.asarray(UNEMPLOYED).ravel().tolist()).astype(float)
    )
    o_retired = (
        df_o["choice"].isin(np.asarray(RETIREMENT).ravel().tolist()).astype(float)
    )
    o_not_working = (
        df_o["choice"].isin(np.asarray(NOT_WORKING).ravel().tolist()).astype(float)
    )

    c_unemployed = (
        df_c["choice"]
        .isin(np.asarray(UNEMPLOYED_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_retired = (
        df_c["choice"]
        .isin(np.asarray(RETIREMENT_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )
    c_not_working = (
        df_c["choice"]
        .isin(np.asarray(NOT_WORKING_NO_CARE_DEMAND).ravel().tolist())
        .astype(float)
    )

    # Build per-period frames
    o_cols = df_o[["agent", "period"]].copy()
    o_cols["unemployed_o"] = o_unemployed
    o_cols["retired_o"] = o_retired
    o_cols["not_working_o"] = o_not_working

    c_cols = df_c[["agent", "period"]].copy()
    c_cols["unemployed_c"] = c_unemployed
    c_cols["retired_c"] = c_retired
    c_cols["not_working_c"] = c_not_working

    # Merge on (agent, period) to get matched differences
    merged = o_cols.merge(c_cols, on=["agent", "period"], how="inner")
    merged["diff_unemployed"] = merged["unemployed_o"] - merged["unemployed_c"]
    merged["diff_retired"] = merged["retired_o"] - merged["retired_c"]
    merged["diff_not_working"] = merged["not_working_o"] - merged["not_working_c"]

    # Compute distance based on original
    df_o_dist = _add_distance_to_first_care(df_o)
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_period"]
        .first()
        .reset_index()
    )
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care"] = merged["period"] - merged["first_care_period"]

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care"] >= -window)
        & (merged["distance_to_first_care"] <= window)
    ]

    # Average differences by distance
    prof = (
        merged.groupby("distance_to_first_care", observed=False)[
            ["diff_unemployed", "diff_retired", "diff_not_working"]
        ]
        .mean()
        .reset_index()
        .sort_values("distance_to_first_care")
    )

    # Plot
    plt.figure(figsize=(12, 7))
    plt.title("Matched Differences by Distance to First Care Spell (No-Work Outcomes)")

    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_unemployed"],
        label="Unemployed (Orig - No Care)",
        color="tab:orange",
        linewidth=2,
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_retired"],
        label="Retired (Orig - No Care)",
        color="tab:purple",
        linewidth=2,
    )
    plt.plot(
        prof["distance_to_first_care"],
        prof["diff_not_working"],
        label="Not Working (Orig - No Care)",
        color="tab:brown",
        linewidth=2,
    )

    plt.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Distance to first care spell (periods)")
    plt.ylabel("Mean difference (Original - No Care)")
    plt.xlim(-window, window)
    plt.ylim(-0.025, 0.125)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()
