"""Plot total income and gross own income by age from simulated model data.

Creates a single plot with two lines:
- Total income (net_hh_income: includes net household income, interest,
  care benefits, and bequest)
- Gross own income (gross_labor_income + gross_retirement_income)

For the baseline model. This plot helps identify retirement income spikes.
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
from caregiving.model.shared import RETIREMENT, WORK


@pytask.mark.debug_income
@pytask.mark.post_estimation
def task_plot_income_by_age(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "income_by_age.png",
) -> None:
    """Plot total income, gross own income, net own income, and post-tax incomes.

    Creates a single plot with five lines:
    - Total income (net_hh_income)
    - Gross own income (gross_labor_income + gross_retirement_income)
    - Net own income (own_income_after_ssc - combined labor and pension)
    - Labor income post tax (own_income_after_ssc when working)
    - Pension income post tax (own_income_after_ssc when retired)

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to baseline simulated data pkl file
    path_to_plot : Path
        Path to save the plot

    """
    # Load specs and simulated data
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df_sim = pd.read_pickle(path_to_simulated_data)

    # Check if agent and period are in the index (MultiIndex) or already columns
    if isinstance(df_sim.index, pd.MultiIndex):
        df_sim = df_sim.reset_index()
    elif "agent" not in df_sim.columns:
        if hasattr(df_sim.index, "names") and "agent" in df_sim.index.names:
            df_sim = df_sim.reset_index()
        else:
            index_names = (
                df_sim.index.names if hasattr(df_sim.index, "names") else "N/A"
            )
            raise ValueError(
                "Cannot find 'agent' column or index level. "
                f"Available columns: {df_sim.columns.tolist()}, "
                f"Index names: {index_names}"
            )

    # Verify agent and period columns exist
    if "agent" not in df_sim.columns or "period" not in df_sim.columns:
        raise ValueError(
            f"Missing required columns 'agent' or 'period'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Create age variable if not already present
    if "age" not in df_sim.columns:
        df_sim["age"] = df_sim["period"] + specs["start_age"]

    # Extract aux variables and add as columns
    df_sim = _add_aux_variables_to_df(df_sim)

    # Calculate gross own income
    if (
        "gross_labor_income" in df_sim.columns
        and "gross_retirement_income" in df_sim.columns
    ):
        df_sim["gross_own_income"] = (
            df_sim["gross_labor_income"] + df_sim["gross_retirement_income"]
        )
    else:
        raise ValueError(
            "Missing required columns 'gross_labor_income' or 'gross_retirement_income'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Use net_hh_income as total income (includes interest, care benefits, bequest)
    # net_hh_income = total_income + interest + care_benefits_and_costs + bequest_from_parent
    if "net_hh_income" not in df_sim.columns:
        raise ValueError(
            f"Missing required column 'net_hh_income'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Verify own_income_after_ssc exists
    if "own_income_after_ssc" not in df_sim.columns:
        raise ValueError(
            f"Missing required column 'own_income_after_ssc'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Extract assets_begin_of_period if available
    # It might be in the continuous_state column (normalized) or as a direct column
    if "assets_begin_of_period" not in df_sim.columns:
        # Try to extract from aux dict first
        df_sim["assets_begin_of_period"] = _extract_aux_variable(
            df_sim, "assets_begin_of_period"
        )
        # If still not available, try to calculate from continuous state
        if (
            df_sim["assets_begin_of_period"].isna().all()
            or df_sim["assets_begin_of_period"].isna().any()
        ):
            # Check if there's a continuous state column
            if "continuous_state" in df_sim.columns:
                # Convert from normalized (0-1) to actual value in 1000s of euros
                df_sim["assets_begin_of_period"] = (
                    df_sim["continuous_state"] * specs["wealth_unit"]
                ) / 1000.0  # Convert to thousands of euros for consistency with income
            else:
                # If not available, create a NaN column so the plot doesn't break
                df_sim["assets_begin_of_period"] = np.nan
    else:
        # If it exists, convert to thousands of euros if needed
        # Check if values are very large (likely in euros, not thousands)
        if df_sim["assets_begin_of_period"].max() > 10000:
            df_sim["assets_begin_of_period"] = df_sim["assets_begin_of_period"] / 1000.0

    # Identify working vs retired individuals
    # Use lagged_choice if available, otherwise use choice
    if "lagged_choice" in df_sim.columns:
        choice_col = "lagged_choice"
    elif "choice" in df_sim.columns:
        choice_col = "choice"
    else:
        raise ValueError(
            "Missing required column 'choice' or 'lagged_choice'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Convert JAX arrays to lists for use with pandas .isin()
    retirement_values = np.asarray(RETIREMENT).ravel().tolist()
    work_values = np.asarray(WORK).ravel().tolist()

    # Calculate labor income post tax and pension income post tax
    # own_income_after_ssc = labor_income_after_ssc (if working) or
    # retirement_income_after_ssc (if retired)
    df_sim["is_retired_flag"] = df_sim[choice_col].isin(retirement_values).astype(float)
    df_sim["is_working_flag"] = df_sim[choice_col].isin(work_values).astype(float)

    # Labor income post tax (own_income_after_ssc when working)
    df_sim["labor_income_post_tax"] = (
        df_sim["own_income_after_ssc"] * df_sim["is_working_flag"]
    )
    # Set to NaN when not working for cleaner averaging
    df_sim.loc[df_sim["is_working_flag"] == 0, "labor_income_post_tax"] = np.nan

    # Pension income post tax (own_income_after_ssc when retired)
    df_sim["pension_income_post_tax"] = (
        df_sim["own_income_after_ssc"] * df_sim["is_retired_flag"]
    )
    # Set to NaN when not retired for cleaner averaging
    df_sim.loc[df_sim["is_retired_flag"] == 0, "pension_income_post_tax"] = np.nan

    # Calculate average by age (pooled across all education levels)
    agg_dict = {
        "net_hh_income": "mean",
        "gross_own_income": "mean",
        "own_income_after_ssc": "mean",
        "labor_income_post_tax": "mean",
        "pension_income_post_tax": "mean",
    }

    # Add assets_begin_of_period if available
    if "assets_begin_of_period" in df_sim.columns:
        agg_dict["assets_begin_of_period"] = "mean"

    avg_by_age = df_sim.groupby("age", observed=False).agg(agg_dict).reset_index()

    # Get all unique ages
    ages = np.sort(df_sim["age"].unique())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot total income (net_hh_income includes interest, care benefits, bequest)
    total_income_values = (
        avg_by_age.set_index("age")["net_hh_income"].reindex(ages).values
    )
    total_income_mask = ~np.isnan(total_income_values)
    if total_income_mask.sum() > 0:
        ax.plot(
            ages[total_income_mask],
            total_income_values[total_income_mask],
            linewidth=2,
            color="blue",
            label="Total Income (net_hh_income)",
            alpha=0.8,
            linestyle="-",
        )

    # Plot gross own income
    gross_own_income_values = (
        avg_by_age.set_index("age")["gross_own_income"].reindex(ages).values
    )
    gross_own_income_mask = ~np.isnan(gross_own_income_values)
    if gross_own_income_mask.sum() > 0:
        ax.plot(
            ages[gross_own_income_mask],
            gross_own_income_values[gross_own_income_mask],
            linewidth=2,
            color="red",
            label="Gross Own Income",
            alpha=0.8,
            linestyle="--",
        )

    # Plot net own income (own_income_after_ssc - combined labor and pension)
    # Note: This is the average for ALL people (working + retired + unemployed)
    # at each age, so it's lower than labor income post tax which only includes workers
    net_own_income_values = (
        avg_by_age.set_index("age")["own_income_after_ssc"].reindex(ages).values
    )
    net_own_income_mask = ~np.isnan(net_own_income_values)
    if net_own_income_mask.sum() > 0:
        ax.plot(
            ages[net_own_income_mask],
            net_own_income_values[net_own_income_mask],
            linewidth=2,
            color="purple",
            label="Net Own Income (all people)",
            alpha=0.8,
            linestyle="-",
        )

    # Plot labor income post tax
    # Note: This is the average for ONLY working people at each age
    # (excludes retired and unemployed, which is why it's higher than net own income)
    labor_income_values = (
        avg_by_age.set_index("age")["labor_income_post_tax"].reindex(ages).values
    )
    labor_income_mask = ~np.isnan(labor_income_values)
    if labor_income_mask.sum() > 0:
        ax.plot(
            ages[labor_income_mask],
            labor_income_values[labor_income_mask],
            linewidth=2,
            color="green",
            label="Labor Income Post Tax (workers only)",
            alpha=0.8,
            linestyle="-.",
        )

    # Plot pension income post tax
    # Note: This is the average for ONLY retired people at each age
    pension_income_values = (
        avg_by_age.set_index("age")["pension_income_post_tax"].reindex(ages).values
    )
    pension_income_mask = ~np.isnan(pension_income_values)
    if pension_income_mask.sum() > 0:
        ax.plot(
            ages[pension_income_mask],
            pension_income_values[pension_income_mask],
            linewidth=2,
            color="orange",
            label="Pension Income Post Tax (retired only)",
            alpha=0.8,
            linestyle=":",
        )

    # Plot assets_begin_of_period if available
    if "assets_begin_of_period" in avg_by_age.columns:
        assets_values = (
            avg_by_age.set_index("age")["assets_begin_of_period"].reindex(ages).values
        )
        assets_mask = ~np.isnan(assets_values)
        if assets_mask.sum() > 0:
            ax.plot(
                ages[assets_mask],
                assets_values[assets_mask],
                linewidth=2,
                color="brown",
                label="Assets Begin of Period",
                alpha=0.8,
                linestyle="--",
            )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Income/Assets (in 1,000€)", fontsize=12)
    ax.set_title(
        "Total Income and Gross Own Income by Age (Baseline Model)", fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_plot}")


@pytask.mark.debug_income
@pytask.mark.post_estimation
def task_plot_assets_by_age(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_by_age.png",
) -> None:
    """Plot assets_begin_of_period and assets_end_of_period by age.

    Creates a single plot with two lines:
    - Assets begin of period (wealth at the start of the period)
    - Assets end of period (savings, wealth at the end of the period after consumption)

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to baseline simulated data pkl file
    path_to_plot : Path
        Path to save the plot

    """
    # Load specs and simulated data
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    df_sim = pd.read_pickle(path_to_simulated_data)

    # Check if agent and period are in the index (MultiIndex) or already columns
    if isinstance(df_sim.index, pd.MultiIndex):
        df_sim = df_sim.reset_index()
    elif "agent" not in df_sim.columns:
        if hasattr(df_sim.index, "names") and "agent" in df_sim.index.names:
            df_sim = df_sim.reset_index()
        else:
            index_names = (
                df_sim.index.names if hasattr(df_sim.index, "names") else "N/A"
            )
            raise ValueError(
                "Cannot find 'agent' column or index level. "
                f"Available columns: {df_sim.columns.tolist()}, "
                f"Index names: {index_names}"
            )

    # Verify agent and period columns exist
    if "agent" not in df_sim.columns or "period" not in df_sim.columns:
        raise ValueError(
            f"Missing required columns 'agent' or 'period'. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Create age variable if not already present
    if "age" not in df_sim.columns:
        df_sim["age"] = df_sim["period"] + specs["start_age"]

    # Extract assets_begin_of_period
    # It might be in the continuous_state column (normalized) or as a direct column
    if "assets_begin_of_period" not in df_sim.columns:
        # Try to extract from aux dict first
        df_sim["assets_begin_of_period"] = _extract_aux_variable(
            df_sim, "assets_begin_of_period"
        )
        # If still not available, try to calculate from continuous state
        if (
            df_sim["assets_begin_of_period"].isna().all()
            or df_sim["assets_begin_of_period"].isna().any()
        ):
            # Check if there's a continuous state column
            if "continuous_state" in df_sim.columns:
                # Convert from normalized (0-1) to actual value in 1000s of euros
                df_sim["assets_begin_of_period"] = (
                    df_sim["continuous_state"] * specs["wealth_unit"]
                ) / 1000.0  # Convert to thousands of euros
            else:
                # If not available, create a NaN column so the plot doesn't break
                df_sim["assets_begin_of_period"] = np.nan
    else:
        # If it exists, convert to thousands of euros if needed
        # Check if values are very large (likely in euros, not thousands)
        if df_sim["assets_begin_of_period"].max() > 10000:
            df_sim["assets_begin_of_period"] = df_sim["assets_begin_of_period"] / 1000.0

    # Extract assets_end_of_period
    # From the budget constraint and simulation code:
    # - assets_begin_of_period = savings + consumption
    # - assets_end_of_period = savings (wealth at end of period after consumption)
    # The "savings" column in simulated data is typically normalized (0-1) and needs
    # to be converted using wealth_unit, similar to continuous_state
    if "assets_end_of_period" not in df_sim.columns:
        # Try to extract from aux dict first
        df_sim["assets_end_of_period"] = _extract_aux_variable(
            df_sim, "assets_end_of_period"
        )

        # If not in aux, try savings column
        if (
            df_sim["assets_end_of_period"].isna().all()
            or df_sim["assets_end_of_period"].isna().any()
        ):
            if "savings" in df_sim.columns:
                # Check if savings is normalized (0-1) or already in actual units
                # If max value is <= 1, it's likely normalized
                if df_sim["savings"].max() <= 1.0:
                    # Convert from normalized (0-1) to actual value in 1000s of euros
                    df_sim["assets_end_of_period"] = (
                        df_sim["savings"] * specs["wealth_unit"]
                    ) / 1000.0
                else:
                    # Already in actual units, just convert to thousands
                    df_sim["assets_end_of_period"] = df_sim["savings"] / 1000.0
            else:
                # Try to calculate from assets_begin_of_period and consumption
                # assets_end_of_period = assets_begin_of_period - consumption
                if "consumption" in df_sim.columns:
                    # Check if consumption needs conversion
                    if df_sim["consumption"].max() <= 1.0:
                        # Normalized, convert using wealth_unit
                        consumption_actual = (
                            df_sim["consumption"] * specs["wealth_unit"]
                        ) / 1000.0
                    else:
                        # Already in actual units
                        consumption_actual = df_sim["consumption"] / 1000.0

                    assets_begin_thousands = df_sim["assets_begin_of_period"]
                    df_sim["assets_end_of_period"] = (
                        assets_begin_thousands - consumption_actual
                    )
                else:
                    # If not available, create a NaN column
                    df_sim["assets_end_of_period"] = np.nan
    else:
        # If it exists, check if it needs conversion
        # If max value is <= 1, it's likely normalized
        if df_sim["assets_end_of_period"].max() <= 1.0:
            # Convert from normalized to actual value
            df_sim["assets_end_of_period"] = (
                df_sim["assets_end_of_period"] * specs["wealth_unit"]
            ) / 1000.0
        elif df_sim["assets_end_of_period"].max() > 10000:
            # Already in actual units but needs conversion to thousands
            df_sim["assets_end_of_period"] = df_sim["assets_end_of_period"] / 1000.0

    # Calculate average by age (pooled across all education levels)
    agg_dict = {}
    if "assets_begin_of_period" in df_sim.columns:
        agg_dict["assets_begin_of_period"] = "mean"
    if "assets_end_of_period" in df_sim.columns:
        agg_dict["assets_end_of_period"] = "mean"

    if not agg_dict:
        raise ValueError(
            "Neither 'assets_begin_of_period' nor 'assets_end_of_period' "
            "could be extracted from the data."
        )

    avg_by_age = df_sim.groupby("age", observed=False).agg(agg_dict).reset_index()

    # Get all unique ages
    ages = np.sort(df_sim["age"].unique())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot assets_begin_of_period
    if "assets_begin_of_period" in avg_by_age.columns:
        assets_begin_values = (
            avg_by_age.set_index("age")["assets_begin_of_period"].reindex(ages).values
        )
        assets_begin_mask = ~np.isnan(assets_begin_values)
        if assets_begin_mask.sum() > 0:
            ax.plot(
                ages[assets_begin_mask],
                assets_begin_values[assets_begin_mask],
                linewidth=2,
                color="blue",
                label="Assets Begin of Period",
                alpha=0.8,
                linestyle="-",
            )

    # Plot assets_end_of_period
    if "assets_end_of_period" in avg_by_age.columns:
        assets_end_values = (
            avg_by_age.set_index("age")["assets_end_of_period"].reindex(ages).values
        )
        assets_end_mask = ~np.isnan(assets_end_values)
        if assets_end_mask.sum() > 0:
            ax.plot(
                ages[assets_end_mask],
                assets_end_values[assets_end_mask],
                linewidth=2,
                color="red",
                label="Assets End of Period (Savings)",
                alpha=0.8,
                linestyle="--",
            )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Assets (in 1,000€)", fontsize=12)
    ax.set_title("Assets Begin and End of Period by Age (Baseline Model)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_plot}")


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
        "net_hh_income",
        "hh_net_income_wo_interest",
        "interest",
        "joint_gross_labor_income",
        "joint_gross_retirement_income",
        "gross_labor_income",
        "gross_retirement_income",
        "bequest_from_parent",
        "income_tax",
        "own_ssc",
        "partner_ssc",
        "total_tax_revenue",
        "government_expenditures",
        "net_government_budget",
        "income_shock_previous_period",
        "income_shock_for_labor",
        "own_income_after_ssc",
    ]

    for var in aux_vars:
        if var not in df.columns:
            df[var] = _extract_aux_variable(df_sim, var)

    return df
