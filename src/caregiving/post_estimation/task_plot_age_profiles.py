"""Plot age profiles for wealth, savings, employment, and income for caregivers.

Creates separate plots for:
- Wealth (assets_begin_of_period)
- Savings decision (savings_dec)
- Employment (work indicator)
- Own income after SSC

All plots are conditioned on individuals who ever provided care (care_ever == True).
X-axis: age
"""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    N_WEEKS_IN_YEAR,
    PART_TIME,
    WORK,
)
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)


@pytask.mark.baseline_model
@pytask.mark.post_estimation_caregivers
def task_plot_age_profiles_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_simulated_data_no_care_demand: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "wealth_by_age_caregivers.png",
    path_to_plot_savings_dec: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "savings_dec_by_age_caregivers.png",
    path_to_plot_employment: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "employment_by_age_caregivers.png",
    path_to_plot_own_income_after_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "own_income_after_ssc_by_age_caregivers.png",
    # path_to_plot_bequest_from_parent: Annotated[Path, Product] = BLD
    # / "plots"
    # / "post_estimation"
    # / "age_profiles"
    # / "bequest_from_parent_by_age_caregivers.png",
    # path_to_plot_bequest_from_parent_positive: Annotated[Path, Product] = BLD
    # / "plots"
    # / "post_estimation"
    # / "age_profiles"
    # / "bequest_from_parent_positive_by_age_caregivers.png",
    path_to_plot_working_hours: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "working_hours_by_age_caregivers.png",
    path_to_plot_full_time: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "full_time_by_age_caregivers.png",
    path_to_plot_part_time: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "part_time_by_age_caregivers.png",
    path_to_plot_gets_inheritance: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "gets_inheritance_by_age_caregivers.png",
):
    """Plot age profiles for wealth, savings, employment, and income for caregivers.

    Creates separate plots for wealth, savings_dec, employment, and own_income_after_ssc
    by age, conditioned on individuals who ever provided care (care_ever == True),
    and overlays the no care demand counterfactual for the same individuals.

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to baseline simulated data pkl file
    path_to_simulated_data_no_care_demand : Path
        Path to no care demand counterfactual simulated data pkl file
    path_to_plot_wealth : Path
        Path to save the wealth plot
    path_to_plot_savings_dec : Path
        Path to save the savings decision plot
    path_to_plot_employment : Path
        Path to save the employment plot
    path_to_plot_own_income_after_ssc : Path
        Path to save the own income after SSC plot
    """
    # Load specs and simulated data
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df_baseline = pd.read_pickle(path_to_simulated_data)
    df_no_care_demand = pd.read_pickle(path_to_simulated_data_no_care_demand)

    # ------------------------------------------------------------------
    # Prepare baseline data: ensure agent/period, alive, age, caregivers
    # ------------------------------------------------------------------
    if isinstance(df_baseline.index, pd.MultiIndex):
        df_baseline = df_baseline.reset_index()
    elif "agent" not in df_baseline.columns:
        if hasattr(df_baseline.index, "names") and "agent" in df_baseline.index.names:
            df_baseline = df_baseline.reset_index()
        else:
            raise ValueError(
                f"Cannot find 'agent' column or index level. "
                f"Available columns: {df_baseline.columns.tolist()}, "
                f"Index names: {df_baseline.index.names if hasattr(df_baseline.index, 'names') else 'N/A'}"
            )

    # Verify agent and period columns exist
    if "agent" not in df_baseline.columns or "period" not in df_baseline.columns:
        raise ValueError(
            f"Missing required columns 'agent' or 'period'. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )

    # Filter to alive agents
    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()

    # Create age variable if not already present
    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]

    # Create care_ever variable: check if agent ever provided informal care
    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    df_baseline["informal_care"] = df_baseline["choice"].isin(care_codes)
    df_baseline["care_ever"] = df_baseline.groupby("agent")["informal_care"].transform(
        "any"
    )

    # Identify caregiver agents in the baseline model
    caregiver_ids = df_baseline.loc[
        df_baseline["care_ever"] == True, "agent"
    ].unique()  # noqa: E712

    # Filter baseline to only individuals who ever provided care
    df_baseline = df_baseline[df_baseline["agent"].isin(caregiver_ids)].copy()

    # ------------------------------------------------------------------
    # Prepare no care demand data for the same caregiver agents
    # ------------------------------------------------------------------
    if isinstance(df_no_care_demand.index, pd.MultiIndex):
        df_no_care_demand = df_no_care_demand.reset_index()
    elif "agent" not in df_no_care_demand.columns:
        if (
            hasattr(df_no_care_demand.index, "names")
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index()
        else:
            raise ValueError(
                f"Cannot find 'agent' column or index level in no care demand data. "
                f"Available columns: {df_no_care_demand.columns.tolist()}, "
                f"Index names: {df_no_care_demand.index.names if hasattr(df_no_care_demand.index, 'names') else 'N/A'}"
            )

    if (
        "agent" not in df_no_care_demand.columns
        or "period" not in df_no_care_demand.columns
    ):
        raise ValueError(
            f"Missing required columns 'agent' or 'period' in no care demand data. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    # Filter to alive agents
    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    # Create age variable if not already present
    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    # Restrict to the same caregiver agents identified from the baseline model
    df_no_care_demand = df_no_care_demand[
        df_no_care_demand["agent"].isin(caregiver_ids)
    ].copy()

    # Extract aux variables and add as columns (for own_income_after_ssc, bequests)
    df_baseline = _add_aux_variables_to_df(df_baseline)
    df_no_care_demand = _add_aux_variables_to_df(df_no_care_demand)

    # Create employment variable (work indicator) for both scenarios
    work_values_baseline = np.asarray(WORK).ravel().tolist()
    work_values_no_care_demand = np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist()
    df_baseline["employment"] = (
        df_baseline["choice"].isin(work_values_baseline).astype(float)
    )
    df_no_care_demand["employment"] = (
        df_no_care_demand["choice"].isin(work_values_no_care_demand).astype(float)
    )

    # Full-time and part-time indicators for both scenarios
    ft_values_baseline = np.asarray(FULL_TIME).ravel().tolist()
    pt_values_baseline = np.asarray(PART_TIME).ravel().tolist()
    ft_values_no_care_demand = np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist()
    pt_values_no_care_demand = np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist()

    df_baseline["full_time"] = (
        df_baseline["choice"].isin(ft_values_baseline).astype(float)
    )
    df_baseline["part_time"] = (
        df_baseline["choice"].isin(pt_values_baseline).astype(float)
    )
    df_no_care_demand["full_time"] = (
        df_no_care_demand["choice"].isin(ft_values_no_care_demand).astype(float)
    )
    df_no_care_demand["part_time"] = (
        df_no_care_demand["choice"].isin(pt_values_no_care_demand).astype(float)
    )

    # Weekly working hours (derived from annual hours)
    df_baseline["working_hours_weekly"] = df_baseline["working_hours"] / N_WEEKS_IN_YEAR
    df_no_care_demand["working_hours_weekly"] = (
        df_no_care_demand["working_hours"] / N_WEEKS_IN_YEAR
    )

    # Verify required columns exist
    required_cols = [
        "assets_begin_of_period",
        "savings_dec",
        "employment",
        "own_income_after_ssc",
        "gets_inheritance",
        "working_hours_weekly",
        "full_time",
        "part_time",
    ]
    missing_cols_baseline = [
        col for col in required_cols if col not in df_baseline.columns
    ]
    missing_cols_no_care_demand = [
        col for col in required_cols if col not in df_no_care_demand.columns
    ]
    if missing_cols_baseline:
        raise ValueError(
            f"Missing required columns in baseline data: {missing_cols_baseline}. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )
    if missing_cols_no_care_demand:
        raise ValueError(
            f"Missing required columns in no care demand data: {missing_cols_no_care_demand}. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    # Plot configurations: (column_name, ylabel, title, path)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Wealth (in 1,000€)",
            "Wealth by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_wealth,
        ),
        (
            "savings_dec",
            "Average Savings Decision (in 1,000€)",
            "Savings Decision by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_savings_dec,
        ),
        (
            "employment",
            "Average Employment Rate",
            "Employment by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_employment,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC (in 1,000€)",
            "Own Income After SSC by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_own_income_after_ssc,
        ),
        # (
        #     "bequest_from_parent",
        #     "Average Bequest from Parent (in 1,000€)",
        #     "Bequest from Parent by Age (Caregivers, baseline vs no care demand)",
        #     path_to_plot_bequest_from_parent,
        # ),
        # (
        #     "bequest_from_parent_positive",
        #     "Average Bequest from Parent, conditional on > 0 (in 1,000€)",
        #     "Bequest from Parent (conditional on > 0) by Age (Caregivers, baseline vs no care demand)",
        #     path_to_plot_bequest_from_parent_positive,
        # ),
        (
            "gets_inheritance",
            "Share Getting Inheritance",
            "Share Getting Inheritance by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_gets_inheritance,
        ),
        (
            "working_hours_weekly",
            "Average Weekly Working Hours",
            "Working Hours (weekly) by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_working_hours,
        ),
        (
            "full_time",
            "Share Full-time Employed",
            "Full-time Employment by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_full_time,
        ),
        (
            "part_time",
            "Share Part-time Employed",
            "Part-time Employment by Age (Caregivers, baseline vs no care demand)",
            path_to_plot_part_time,
        ),
    ]

    for col, ylabel, title, path in plot_configs:
        _plot_outcome_by_age_two_scenarios(
            df_baseline=df_baseline,
            df_no_care_demand=df_no_care_demand,
            outcome_col=col,
            ylabel=ylabel,
            title=title,
            path_to_plot=path,
        )


@pytask.mark.baseline_model
@pytask.mark.post_estimation_caregivers
def task_plot_age_profile_differences_caregivers(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_simulated_data_no_care_demand: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_wealth_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "wealth_diff_by_age_caregivers.png",
    path_to_plot_savings_dec_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "savings_dec_diff_by_age_caregivers.png",
    path_to_plot_employment_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "employment_diff_by_age_caregivers.png",
    path_to_plot_own_income_after_ssc_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "own_income_after_ssc_diff_by_age_caregivers.png",
    path_to_plot_bequest_from_parent_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "bequest_from_parent_diff_by_age_caregivers.png",
    path_to_plot_bequest_from_parent_positive_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "bequest_from_parent_positive_diff_by_age_caregivers.png",
    path_to_plot_working_hours_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "working_hours_diff_by_age_caregivers.png",
    path_to_plot_full_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "full_time_diff_by_age_caregivers.png",
    path_to_plot_part_time_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "part_time_diff_by_age_caregivers.png",
    path_to_plot_gets_inheritance_diff: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "age_profiles"
    / "differences"
    / "gets_inheritance_diff_by_age_caregivers.png",
):
    """Plot age-specific average differences between matched caregiver samples.

    Uses the same matched caregiver agents as in ``task_plot_age_profiles_caregivers``
    and plots baseline minus no-care-demand differences by age for:
    wealth, savings_dec, employment, and own_income_after_ssc.
    """
    # Load specs and simulated data
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df_baseline = pd.read_pickle(path_to_simulated_data)
    df_no_care_demand = pd.read_pickle(path_to_simulated_data_no_care_demand)

    # ------------------------------------------------------------------
    # Prepare baseline data: ensure agent/period, alive, age, caregivers
    # ------------------------------------------------------------------
    if isinstance(df_baseline.index, pd.MultiIndex):
        df_baseline = df_baseline.reset_index()
    elif "agent" not in df_baseline.columns:
        if hasattr(df_baseline.index, "names") and "agent" in df_baseline.index.names:
            df_baseline = df_baseline.reset_index()
        else:
            raise ValueError(
                f"Cannot find 'agent' column or index level. "
                f"Available columns: {df_baseline.columns.tolist()}, "
                f"Index names: {df_baseline.index.names if hasattr(df_baseline.index, 'names') else 'N/A'}"
            )

    if "agent" not in df_baseline.columns or "period" not in df_baseline.columns:
        raise ValueError(
            f"Missing required columns 'agent' or 'period'. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )

    df_baseline = df_baseline[df_baseline["health"] != DEAD].copy()

    if "age" not in df_baseline.columns:
        df_baseline["age"] = df_baseline["period"] + specs["start_age"]

    care_codes = np.asarray(INFORMAL_CARE).ravel().tolist()
    df_baseline["informal_care"] = df_baseline["choice"].isin(care_codes)
    df_baseline["care_ever"] = df_baseline.groupby("agent")["informal_care"].transform(
        "any"
    )

    caregiver_ids = df_baseline.loc[
        df_baseline["care_ever"] == True, "agent"
    ].unique()  # noqa: E712

    df_baseline = df_baseline[df_baseline["agent"].isin(caregiver_ids)].copy()

    # ------------------------------------------------------------------
    # Prepare no care demand data for the same caregiver agents
    # ------------------------------------------------------------------
    if isinstance(df_no_care_demand.index, pd.MultiIndex):
        df_no_care_demand = df_no_care_demand.reset_index()
    elif "agent" not in df_no_care_demand.columns:
        if (
            hasattr(df_no_care_demand.index, "names")
            and "agent" in df_no_care_demand.index.names
        ):
            df_no_care_demand = df_no_care_demand.reset_index()
        else:
            raise ValueError(
                f"Cannot find 'agent' column or index level in no care demand data. "
                f"Available columns: {df_no_care_demand.columns.tolist()}, "
                f"Index names: {df_no_care_demand.index.names if hasattr(df_no_care_demand.index, 'names') else 'N/A'}"
            )

    if (
        "agent" not in df_no_care_demand.columns
        or "period" not in df_no_care_demand.columns
    ):
        raise ValueError(
            f"Missing required columns 'agent' or 'period' in no care demand data. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    df_no_care_demand = df_no_care_demand[df_no_care_demand["health"] != DEAD].copy()

    if "age" not in df_no_care_demand.columns:
        df_no_care_demand["age"] = df_no_care_demand["period"] + specs["start_age"]

    df_no_care_demand = df_no_care_demand[
        df_no_care_demand["agent"].isin(caregiver_ids)
    ].copy()

    # Extract aux variables and add as columns (for own_income_after_ssc, bequests)
    df_baseline = _add_aux_variables_to_df(df_baseline)
    df_no_care_demand = _add_aux_variables_to_df(df_no_care_demand)

    # Create employment variable (work indicator) for both scenarios
    work_values_baseline = np.asarray(WORK).ravel().tolist()
    work_values_no_care_demand = np.asarray(WORK_NO_CARE_DEMAND).ravel().tolist()
    df_baseline["employment"] = (
        df_baseline["choice"].isin(work_values_baseline).astype(float)
    )
    df_no_care_demand["employment"] = (
        df_no_care_demand["choice"].isin(work_values_no_care_demand).astype(float)
    )

    # Full-time and part-time indicators for both scenarios
    ft_values_baseline = np.asarray(FULL_TIME).ravel().tolist()
    pt_values_baseline = np.asarray(PART_TIME).ravel().tolist()
    ft_values_no_care_demand = np.asarray(FULL_TIME_NO_CARE_DEMAND).ravel().tolist()
    pt_values_no_care_demand = np.asarray(PART_TIME_NO_CARE_DEMAND).ravel().tolist()

    df_baseline["full_time"] = (
        df_baseline["choice"].isin(ft_values_baseline).astype(float)
    )
    df_baseline["part_time"] = (
        df_baseline["choice"].isin(pt_values_baseline).astype(float)
    )
    df_no_care_demand["full_time"] = (
        df_no_care_demand["choice"].isin(ft_values_no_care_demand).astype(float)
    )
    df_no_care_demand["part_time"] = (
        df_no_care_demand["choice"].isin(pt_values_no_care_demand).astype(float)
    )

    required_cols = [
        "assets_begin_of_period",
        "savings_dec",
        "employment",
        "own_income_after_ssc",
        "bequest_from_parent",
        "working_hours",
        "full_time",
        "part_time",
        "gets_inheritance",
    ]
    missing_cols_baseline = [
        col for col in required_cols if col not in df_baseline.columns
    ]
    missing_cols_no_care_demand = [
        col for col in required_cols if col not in df_no_care_demand.columns
    ]
    if missing_cols_baseline:
        raise ValueError(
            f"Missing required columns in baseline data: {missing_cols_baseline}. "
            f"Available columns: {df_baseline.columns.tolist()}"
        )
    if missing_cols_no_care_demand:
        raise ValueError(
            f"Missing required columns in no care demand data: {missing_cols_no_care_demand}. "
            f"Available columns: {df_no_care_demand.columns.tolist()}"
        )

    # Plot configurations: (column_name, ylabel, title, path)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Baseline - No care demand (in 1,000€)",
            "Wealth difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_wealth_diff,
        ),
        (
            "savings_dec",
            "Baseline - No care demand (in 1,000€)",
            "Savings Decision difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_savings_dec_diff,
        ),
        (
            "employment",
            "Baseline - No care demand (employment rate)",
            "Employment difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_employment_diff,
        ),
        (
            "own_income_after_ssc",
            "Baseline - No care demand (in 1,000€)",
            "Own Income After SSC difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_own_income_after_ssc_diff,
        ),
        (
            "bequest_from_parent",
            "Baseline - No care demand (in 1,000€)",
            "Bequest from Parent difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_bequest_from_parent_diff,
        ),
        (
            "bequest_from_parent_positive",
            "Baseline - No care demand (in 1,000€)",
            "Bequest from Parent (conditional on > 0) difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_bequest_from_parent_positive_diff,
        ),
        (
            "working_hours_weekly",
            "Baseline - No care demand (weekly hours)",
            "Working Hours (weekly) difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_working_hours_diff,
        ),
        (
            "full_time",
            "Baseline - No care demand (share full-time)",
            "Full-time Employment difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_full_time_diff,
        ),
        (
            "part_time",
            "Baseline - No care demand (share part-time)",
            "Part-time Employment difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_part_time_diff,
        ),
        (
            "gets_inheritance",
            "Baseline - No care demand (share getting inheritance)",
            "Share Getting Inheritance difference by Age (Caregivers, baseline - no care demand)",
            path_to_plot_gets_inheritance_diff,
        ),
    ]

    for col, ylabel, title, path in plot_configs:
        _plot_difference_by_age_two_scenarios(
            df_baseline=df_baseline,
            df_no_care_demand=df_no_care_demand,
            outcome_col=col,
            ylabel=ylabel,
            title=title,
            path_to_plot=path,
        )


def _plot_outcome_by_age_two_scenarios(
    df_baseline,
    df_no_care_demand,
    outcome_col,
    ylabel,
    title,
    path_to_plot,
):
    """Helper function to plot an outcome by age.

    Parameters
    ----------
    df_baseline : pd.DataFrame
        Baseline simulated data (already filtered to caregiver agents)
    df_no_care_demand : pd.DataFrame
        No care demand simulated data (already filtered to the same caregiver agents)
    outcome_col : str
        Column name to plot
    ylabel : str
        Y-axis label
    title : str
        Plot title
    path_to_plot : Path
        Path to save the plot
    """
    # Calculate average by age for both scenarios
    avg_baseline = (
        df_baseline.groupby("age", observed=False)[outcome_col].mean().reset_index()
    )
    avg_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)[outcome_col]
        .mean()
        .reset_index()
    )

    # Get all unique ages across both scenarios
    ages = np.sort(
        np.union1d(df_baseline["age"].unique(), df_no_care_demand["age"].unique())
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline line
    values_baseline = avg_baseline.set_index("age")[outcome_col].reindex(ages).values
    mask_baseline = ~np.isnan(values_baseline)
    if mask_baseline.sum() > 0:
        ax.plot(
            ages[mask_baseline],
            values_baseline[mask_baseline],
            linewidth=2,
            color="steelblue",
            alpha=0.8,
            label="Baseline",
        )

    # No care demand line
    values_no_care_demand = (
        avg_no_care_demand.set_index("age")[outcome_col].reindex(ages).values
    )
    mask_no_care_demand = ~np.isnan(values_no_care_demand)
    if mask_no_care_demand.sum() > 0:
        ax.plot(
            ages[mask_no_care_demand],
            values_no_care_demand[mask_no_care_demand],
            linewidth=2,
            color="darkorange",
            alpha=0.8,
            linestyle="--",
            label="No care demand",
        )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_plot}")


def _plot_difference_by_age_two_scenarios(
    df_baseline,
    df_no_care_demand,
    outcome_col,
    ylabel,
    title,
    path_to_plot,
):
    """Helper to plot age-specific average differences between two scenarios.

    Plots baseline minus no care demand as a single line over age.
    """
    avg_baseline = (
        df_baseline.groupby("age", observed=False)[outcome_col].mean().reset_index()
    )
    avg_no_care_demand = (
        df_no_care_demand.groupby("age", observed=False)[outcome_col]
        .mean()
        .reset_index()
    )

    ages = np.sort(
        np.union1d(df_baseline["age"].unique(), df_no_care_demand["age"].unique())
    )

    # Align and compute differences
    base_vals = avg_baseline.set_index("age")[outcome_col].reindex(ages).values
    cf_vals = avg_no_care_demand.set_index("age")[outcome_col].reindex(ages).values
    diff_vals = base_vals - cf_vals
    mask = ~np.isnan(diff_vals)

    fig, ax = plt.subplots(figsize=(10, 6))

    if mask.sum() > 0:
        ax.plot(
            ages[mask],
            diff_vals[mask],
            linewidth=2,
            color="purple",
            alpha=0.8,
        )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Difference plot saved to {path_to_plot}")


def _extract_aux_variable(df_sim, var_name):
    """Extract a variable from either direct column or aux dictionary."""
    if var_name in df_sim.columns:
        return df_sim[var_name]
    elif "aux" in df_sim.columns:
        return df_sim["aux"].apply(
            lambda x: (x.get(var_name, np.nan) if isinstance(x, dict) else np.nan)
        )
    else:
        return pd.Series(np.nan, index=df_sim.index)


def _add_aux_variables_to_df(df_sim):
    """Extract aux variables and add them as columns to the dataframe."""
    df = df_sim.copy()

    # List of variables to extract from aux dict
    aux_vars = [
        "own_income_after_ssc",
        "bequest_from_parent",
        "gets_inheritance",
    ]

    for var in aux_vars:
        if var not in df.columns:
            df[var] = _extract_aux_variable(df_sim, var)

    return df
