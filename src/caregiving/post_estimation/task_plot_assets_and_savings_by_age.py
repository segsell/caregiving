"""Plot average assets and savings by age and education from simulated model data.

Creates plots for variables from budget equation aux dict and related variables:
- Assets begin of period
- Savings decision (savings_dec)
- Total income
- Savings (raw savings/wealth level)
- Savings rate
- Net household income
- Household net income without interest
- Interest
- Joint gross labor income
- Joint gross retirement income
- Gross labor income
- Gross retirement income
- Bequest from parent
- Income tax
- Own SSC
- Partner SSC
- Total tax revenue
- Government expenditures
- Net government budget
- Experience years (exp_years)
- Own income after SSC
- Experience
- Already retired (already_retired)
- Is retired (is_retired)
- Fresh retired (fresh_retired = (already_retired == 0) & (is_retired == 1))
- Actual retirement age (actual_retirement_age)

For four models:
- Baseline model
- Caregiving leave counterfactual
- No care demand counterfactual
- Baseline model without inheritance

Each plot shows 2 lines: Low education and High education.

Note: Savings, savings_dec,
    and savings_rate plots exclude the maximum age (end_age = 100).
All plots are organized into subfolders: assets_and_savings/baseline/,
    assets_and_savings/caregiving_leave/, assets_and_savings/no_care_demand/
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


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.post_assets_and_savings
def task_plot_assets_and_savings_by_age_baseline(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot_assets_begin: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "assets_begin_of_period_by_age.png",
    path_to_plot_savings_dec: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "savings_dec_by_age.png",
    path_to_plot_total_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "total_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "savings_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "savings_rate_by_age.png",
    path_to_plot_net_hh_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "net_hh_income_by_age.png",
    path_to_plot_interest: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "interest_by_age.png",
    path_to_plot_joint_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "joint_gross_labor_income_by_age.png",
    path_to_plot_joint_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "joint_gross_retirement_income_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "gross_labor_income_by_age.png",
    path_to_plot_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "gross_retirement_income_by_age.png",
    path_to_plot_bequest_from_parent: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "bequest_from_parent_by_age.png",
    path_to_plot_income_tax: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "income_tax_by_age.png",
    path_to_plot_own_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "own_ssc_by_age.png",
    path_to_plot_partner_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "partner_ssc_by_age.png",
    path_to_plot_total_tax_revenue: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "total_tax_revenue_by_age.png",
    path_to_plot_government_expenditures: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "government_expenditures_by_age.png",
    path_to_plot_net_government_budget: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "net_government_budget_by_age.png",
    path_to_plot_exp_years: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "exp_years_by_age.png",
    path_to_plot_own_income_after_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "own_income_after_ssc_by_age.png",
    path_to_plot_experience: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "experience_by_age.png",
    path_to_plot_already_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "already_retired_by_age.png",
    path_to_plot_is_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "is_retired_by_age.png",
    path_to_plot_fresh_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "fresh_retired_by_age.png",
    path_to_plot_actual_retirement_age: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "baseline"
    / "actual_retirement_age_by_age.png",
):
    """Plot average assets and savings by age and education from baseline  # noqa: E501
    simulated data.

    Creates plots for all variables from budget equation aux dict and related variables.
    Each plot shows 2 lines (Low and High education).

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to baseline simulated data pkl file
    path_to_plot_* : Path
        Paths to save the plots

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

    # Calculate fresh_retired: (already_retired == 0) & (is_retired == 1)
    if "already_retired" in df_sim.columns and "is_retired" in df_sim.columns:
        df_sim["fresh_retired"] = (
            (df_sim["already_retired"] == 0) & (df_sim["is_retired"] == 1)
        ).astype(float)

    # Verify required columns exist
    required_cols = [
        "education",
        "assets_begin_of_period",
        "savings_dec",
        "total_income",
        "savings",
        "savings_rate",
    ]
    missing_cols = [col for col in required_cols if col not in df_sim.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Plot configurations: (column_name, ylabel, title_suffix, exclude_end_age)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period (in 1,000€)",
            "Assets Begin of Period (Baseline Model)",
            False,
        ),
        (
            "savings_dec",
            "Average Savings Decision (in 1,000€)",
            "Savings Decision (Baseline Model)",
            True,
        ),
        (
            "total_income",
            "Average Total Income (in 1,000€)",
            "Total Income (Baseline Model)",
            False,
        ),
        ("savings", "Average Savings (in 1,000€)", "Savings (Baseline Model)", True),
        ("savings_rate", "Average Savings Rate", "Savings Rate (Baseline Model)", True),
        (
            "net_hh_income",
            "Average Net Household Income (in 1,000€)",
            "Net Household Income (Baseline Model)",
            False,
        ),
        (
            "interest",
            "Average Interest (in 1,000€)",
            "Interest (Baseline Model)",
            False,
        ),
        (
            "joint_gross_labor_income",
            "Average Joint Gross Labor Income (in 1,000€)",
            "Joint Gross Labor Income (Baseline Model)",
            False,
        ),
        (
            "joint_gross_retirement_income",
            "Average Joint Gross Retirement Income (in 1,000€)",
            "Joint Gross Retirement Income (Baseline Model)",
            False,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income (in 1,000€)",
            "Gross Labor Income (Baseline Model)",
            False,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income (in 1,000€)",
            "Gross Retirement Income (Baseline Model)",
            False,
        ),
        (
            "bequest_from_parent",
            "Average Bequest from Parent (in 1,000€)",
            "Bequest from Parent (Baseline Model)",
            False,
        ),
        (
            "income_tax",
            "Average Income Tax (in 1,000€)",
            "Income Tax (Baseline Model)",
            False,
        ),
        ("own_ssc", "Average Own SSC (in 1,000€)", "Own SSC (Baseline Model)", False),
        (
            "partner_ssc",
            "Average Partner SSC (in 1,000€)",
            "Partner SSC (Baseline Model)",
            False,
        ),
        (
            "total_tax_revenue",
            "Average Total Tax Revenue (in 1,000€)",
            "Total Tax Revenue (Baseline Model)",
            False,
        ),
        (
            "government_expenditures",
            "Average Government Expenditures (in 1,000€)",
            "Government Expenditures (Baseline Model)",
            False,
        ),
        (
            "net_government_budget",
            "Average Net Government Budget (in 1,000€)",
            "Net Government Budget (Baseline Model)",
            False,
        ),
        (
            "exp_years",
            "Average Experience Years",
            "Experience Years (Baseline Model)",
            False,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC (in 1,000€)",
            "Own Income After SSC (Baseline Model)",
            False,
        ),
        (
            "experience",
            "Average Experience",
            "Experience (Baseline Model)",
            False,
        ),
        (
            "already_retired",
            "Average Already Retired",
            "Already Retired (Baseline Model)",
            False,
        ),
        (
            "is_retired",
            "Average Is Retired",
            "Is Retired (Baseline Model)",
            False,
        ),
        (
            "fresh_retired",
            "Average Fresh Retired",
            "Fresh Retired (Baseline Model)",
            False,
        ),
        (
            "actual_retirement_age",
            "Average Actual Retirement Age",
            "Actual Retirement Age (Baseline Model)",
            False,
        ),
    ]

    path_params = [
        path_to_plot_assets_begin,
        path_to_plot_savings_dec,
        path_to_plot_total_income,
        path_to_plot_savings,
        path_to_plot_savings_rate,
        path_to_plot_net_hh_income,
        path_to_plot_interest,
        path_to_plot_joint_gross_labor_income,
        path_to_plot_joint_gross_retirement_income,
        path_to_plot_gross_labor_income,
        path_to_plot_gross_retirement_income,
        path_to_plot_bequest_from_parent,
        path_to_plot_income_tax,
        path_to_plot_own_ssc,
        path_to_plot_partner_ssc,
        path_to_plot_total_tax_revenue,
        path_to_plot_government_expenditures,
        path_to_plot_net_government_budget,
        path_to_plot_exp_years,
        path_to_plot_own_income_after_ssc,
        path_to_plot_experience,
        path_to_plot_already_retired,
        path_to_plot_is_retired,
        path_to_plot_fresh_retired,
        path_to_plot_actual_retirement_age,
    ]

    for (col, ylabel, title, exclude_age), path in zip(
        plot_configs, path_params, strict=True
    ):
        _plot_asset_savings_outcome(
            df_sim=df_sim,
            specs=specs,
            outcome_col=col,
            ylabel=ylabel,
            title_suffix=title,
            path_to_plot=path,
            exclude_end_age=exclude_age,
        )


@pytask.mark.no_inheritance
@pytask.mark.baseline_model_no_inheritance
@pytask.mark.post_estimation
@pytask.mark.post_assets_and_savings
def task_plot_assets_and_savings_by_age_no_inheritance(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_inheritance.pkl",
    path_to_plot_assets_begin: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "assets_begin_of_period_by_age.png",
    path_to_plot_savings_dec: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "savings_dec_by_age.png",
    path_to_plot_total_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "total_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "savings_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "savings_rate_by_age.png",
    path_to_plot_net_hh_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "net_hh_income_by_age.png",
    path_to_plot_interest: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "interest_by_age.png",
    path_to_plot_joint_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "joint_gross_labor_income_by_age.png",
    path_to_plot_joint_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "joint_gross_retirement_income_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "gross_labor_income_by_age.png",
    path_to_plot_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "gross_retirement_income_by_age.png",
    path_to_plot_bequest_from_parent: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "bequest_from_parent_by_age.png",
    path_to_plot_income_tax: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "income_tax_by_age.png",
    path_to_plot_own_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "own_ssc_by_age.png",
    path_to_plot_partner_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "partner_ssc_by_age.png",
    path_to_plot_total_tax_revenue: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "total_tax_revenue_by_age.png",
    path_to_plot_government_expenditures: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "government_expenditures_by_age.png",
    path_to_plot_net_government_budget: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "net_government_budget_by_age.png",
    path_to_plot_exp_years: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "exp_years_by_age.png",
    path_to_plot_own_income_after_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "own_income_after_ssc_by_age.png",
    path_to_plot_experience: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "experience_by_age.png",
    path_to_plot_already_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "already_retired_by_age.png",
    path_to_plot_is_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "is_retired_by_age.png",
    path_to_plot_fresh_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "fresh_retired_by_age.png",
    path_to_plot_actual_retirement_age: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_inheritance"
    / "actual_retirement_age_by_age.png",
):
    """Plot average assets and savings by age and education from no  # noqa: E501
    inheritance simulated data.

    Creates plots for all variables from budget equation aux dict and related variables.
    Each plot shows 2 lines (Low and High education).

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to no inheritance simulated data pkl file
    path_to_plot_* : Path
        Paths to save the plots

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

    # Calculate fresh_retired: (already_retired == 0) & (is_retired == 1)
    if "already_retired" in df_sim.columns and "is_retired" in df_sim.columns:
        df_sim["fresh_retired"] = (
            (df_sim["already_retired"] == 0) & (df_sim["is_retired"] == 1)
        ).astype(float)

    # Verify required columns exist
    required_cols = [
        "education",
        "assets_begin_of_period",
        "savings_dec",
        "total_income",
        "savings",
        "savings_rate",
    ]
    missing_cols = [col for col in required_cols if col not in df_sim.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Plot configurations: (column_name, ylabel, title_suffix, exclude_end_age)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period (in 1,000€)",
            "Assets Begin of Period (No Inheritance)",
            False,
        ),
        (
            "savings_dec",
            "Average Savings Decision (in 1,000€)",
            "Savings Decision (No Inheritance)",
            True,
        ),
        (
            "total_income",
            "Average Total Income (in 1,000€)",
            "Total Income (No Inheritance)",
            False,
        ),
        ("savings", "Average Savings (in 1,000€)", "Savings (No Inheritance)", True),
        (
            "savings_rate",
            "Average Savings Rate",
            "Savings Rate (No Inheritance)",
            True,
        ),
        (
            "net_hh_income",
            "Average Net Household Income (in 1,000€)",
            "Net Household Income (No Inheritance)",
            False,
        ),
        (
            "interest",
            "Average Interest (in 1,000€)",
            "Interest (No Inheritance)",
            False,
        ),
        (
            "joint_gross_labor_income",
            "Average Joint Gross Labor Income (in 1,000€)",
            "Joint Gross Labor Income (No Inheritance)",
            False,
        ),
        (
            "joint_gross_retirement_income",
            "Average Joint Gross Retirement Income (in 1,000€)",
            "Joint Gross Retirement Income (No Inheritance)",
            False,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income (in 1,000€)",
            "Gross Labor Income (No Inheritance)",
            False,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income (in 1,000€)",
            "Gross Retirement Income (No Inheritance)",
            False,
        ),
        (
            "bequest_from_parent",
            "Average Bequest from Parent (in 1,000€)",
            "Bequest from Parent (No Inheritance)",
            False,
        ),
        (
            "income_tax",
            "Average Income Tax (in 1,000€)",
            "Income Tax (No Inheritance)",
            False,
        ),
        ("own_ssc", "Average Own SSC (in 1,000€)", "Own SSC (No Inheritance)", False),
        (
            "partner_ssc",
            "Average Partner SSC (in 1,000€)",
            "Partner SSC (No Inheritance)",
            False,
        ),
        (
            "total_tax_revenue",
            "Average Total Tax Revenue (in 1,000€)",
            "Total Tax Revenue (No Inheritance)",
            False,
        ),
        (
            "government_expenditures",
            "Average Government Expenditures (in 1,000€)",
            "Government Expenditures (No Inheritance)",
            False,
        ),
        (
            "net_government_budget",
            "Average Net Government Budget (in 1,000€)",
            "Net Government Budget (No Inheritance)",
            False,
        ),
        (
            "exp_years",
            "Average Experience Years",
            "Experience Years (No Inheritance)",
            False,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC (in 1,000€)",
            "Own Income After SSC (No Inheritance)",
            False,
        ),
        (
            "experience",
            "Average Experience",
            "Experience (No Inheritance)",
            False,
        ),
        (
            "already_retired",
            "Average Already Retired",
            "Already Retired (No Inheritance)",
            False,
        ),
        (
            "is_retired",
            "Average Is Retired",
            "Is Retired (No Inheritance)",
            False,
        ),
        (
            "fresh_retired",
            "Average Fresh Retired",
            "Fresh Retired (No Inheritance)",
            False,
        ),
        (
            "actual_retirement_age",
            "Average Actual Retirement Age",
            "Actual Retirement Age (No Inheritance)",
            False,
        ),
    ]

    path_params = [
        path_to_plot_assets_begin,
        path_to_plot_savings_dec,
        path_to_plot_total_income,
        path_to_plot_savings,
        path_to_plot_savings_rate,
        path_to_plot_net_hh_income,
        path_to_plot_interest,
        path_to_plot_joint_gross_labor_income,
        path_to_plot_joint_gross_retirement_income,
        path_to_plot_gross_labor_income,
        path_to_plot_gross_retirement_income,
        path_to_plot_bequest_from_parent,
        path_to_plot_income_tax,
        path_to_plot_own_ssc,
        path_to_plot_partner_ssc,
        path_to_plot_total_tax_revenue,
        path_to_plot_government_expenditures,
        path_to_plot_net_government_budget,
        path_to_plot_exp_years,
        path_to_plot_own_income_after_ssc,
        path_to_plot_experience,
        path_to_plot_already_retired,
        path_to_plot_is_retired,
        path_to_plot_fresh_retired,
        path_to_plot_actual_retirement_age,
    ]

    for (col, ylabel, title, exclude_age), path in zip(
        plot_configs, path_params, strict=True
    ):
        _plot_asset_savings_outcome(
            df_sim=df_sim,
            specs=specs,
            outcome_col=col,
            ylabel=ylabel,
            title_suffix=title,
            path_to_plot=path,
            exclude_end_age=exclude_age,
        )


@pytask.mark.caregiving_leave_with_job_retention_model
@pytask.mark.post_estimation
@pytask.mark.post_assets_and_savings
def task_plot_assets_and_savings_by_age_caregiving_leave(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_plot_assets_begin: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "assets_begin_of_period_by_age.png",
    path_to_plot_savings_dec: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "savings_dec_by_age.png",
    path_to_plot_total_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "total_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "savings_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "savings_rate_by_age.png",
    path_to_plot_net_hh_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "net_hh_income_by_age.png",
    path_to_plot_interest: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "interest_by_age.png",
    path_to_plot_joint_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "joint_gross_labor_income_by_age.png",
    path_to_plot_joint_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "joint_gross_retirement_income_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "gross_labor_income_by_age.png",
    path_to_plot_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "gross_retirement_income_by_age.png",
    path_to_plot_bequest_from_parent: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "bequest_from_parent_by_age.png",
    path_to_plot_income_tax: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "income_tax_by_age.png",
    path_to_plot_own_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "own_ssc_by_age.png",
    path_to_plot_partner_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "partner_ssc_by_age.png",
    path_to_plot_total_tax_revenue: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "total_tax_revenue_by_age.png",
    path_to_plot_government_expenditures: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "government_expenditures_by_age.png",
    path_to_plot_net_government_budget: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "net_government_budget_by_age.png",
    path_to_plot_exp_years: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "exp_years_by_age.png",
    path_to_plot_own_income_after_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "own_income_after_ssc_by_age.png",
    path_to_plot_experience: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "experience_by_age.png",
    path_to_plot_already_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "already_retired_by_age.png",
    path_to_plot_is_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "is_retired_by_age.png",
    path_to_plot_fresh_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "fresh_retired_by_age.png",
    path_to_plot_actual_retirement_age: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "caregiving_leave"
    / "actual_retirement_age_by_age.png",
):
    """Plot average assets and savings by age and education from  # noqa: E501
    caregiving leave counterfactual simulated data.

    Creates plots for all variables from budget equation aux dict and related variables.
    Each plot shows 2 lines (Low and High education).

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to caregiving leave counterfactual simulated data pkl file
    path_to_plot_* : Path
        Paths to save the plots

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

    # Calculate fresh_retired: (already_retired == 0) & (is_retired == 1)
    if "already_retired" in df_sim.columns and "is_retired" in df_sim.columns:
        df_sim["fresh_retired"] = (
            (df_sim["already_retired"] == 0) & (df_sim["is_retired"] == 1)
        ).astype(float)

    # Verify required columns exist
    required_cols = [
        "education",
        "assets_begin_of_period",
        "savings_dec",
        "total_income",
        "savings",
        "savings_rate",
    ]
    missing_cols = [col for col in required_cols if col not in df_sim.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Plot configurations: (column_name, ylabel, title_suffix, exclude_end_age)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period (in 1,000€)",
            "Assets Begin of Period (Caregiving Leave)",
            False,
        ),
        (
            "savings_dec",
            "Average Savings Decision (in 1,000€)",
            "Savings Decision (Caregiving Leave)",
            True,
        ),
        (
            "total_income",
            "Average Total Income (in 1,000€)",
            "Total Income (Caregiving Leave)",
            False,
        ),
        ("savings", "Average Savings (in 1,000€)", "Savings (Caregiving Leave)", True),
        (
            "savings_rate",
            "Average Savings Rate",
            "Savings Rate (Caregiving Leave)",
            True,
        ),
        (
            "net_hh_income",
            "Average Net Household Income (in 1,000€)",
            "Net Household Income (Caregiving Leave)",
            False,
        ),
        (
            "interest",
            "Average Interest (in 1,000€)",
            "Interest (Caregiving Leave)",
            False,
        ),
        (
            "joint_gross_labor_income",
            "Average Joint Gross Labor Income (in 1,000€)",
            "Joint Gross Labor Income (Caregiving Leave)",
            False,
        ),
        (
            "joint_gross_retirement_income",
            "Average Joint Gross Retirement Income (in 1,000€)",
            "Joint Gross Retirement Income (Caregiving Leave)",
            False,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income (in 1,000€)",
            "Gross Labor Income (Caregiving Leave)",
            False,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income (in 1,000€)",
            "Gross Retirement Income (Caregiving Leave)",
            False,
        ),
        (
            "bequest_from_parent",
            "Average Bequest from Parent (in 1,000€)",
            "Bequest from Parent (Caregiving Leave)",
            False,
        ),
        (
            "income_tax",
            "Average Income Tax (in 1,000€)",
            "Income Tax (Caregiving Leave)",
            False,
        ),
        ("own_ssc", "Average Own SSC (in 1,000€)", "Own SSC (Caregiving Leave)", False),
        (
            "partner_ssc",
            "Average Partner SSC (in 1,000€)",
            "Partner SSC (Caregiving Leave)",
            False,
        ),
        (
            "total_tax_revenue",
            "Average Total Tax Revenue (in 1,000€)",
            "Total Tax Revenue (Caregiving Leave)",
            False,
        ),
        (
            "government_expenditures",
            "Average Government Expenditures (in 1,000€)",
            "Government Expenditures (Caregiving Leave)",
            False,
        ),
        (
            "net_government_budget",
            "Average Net Government Budget (in 1,000€)",
            "Net Government Budget (Caregiving Leave)",
            False,
        ),
        (
            "exp_years",
            "Average Experience Years",
            "Experience Years (Caregiving Leave)",
            False,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC (in 1,000€)",
            "Own Income After SSC (Caregiving Leave)",
            False,
        ),
        (
            "experience",
            "Average Experience",
            "Experience (Caregiving Leave)",
            False,
        ),
        (
            "already_retired",
            "Average Already Retired",
            "Already Retired (Caregiving Leave)",
            False,
        ),
        (
            "is_retired",
            "Average Is Retired",
            "Is Retired (Caregiving Leave)",
            False,
        ),
        (
            "fresh_retired",
            "Average Fresh Retired",
            "Fresh Retired (Caregiving Leave)",
            False,
        ),
        (
            "actual_retirement_age",
            "Average Actual Retirement Age",
            "Actual Retirement Age (Caregiving Leave)",
            False,
        ),
    ]

    path_params = [
        path_to_plot_assets_begin,
        path_to_plot_savings_dec,
        path_to_plot_total_income,
        path_to_plot_savings,
        path_to_plot_savings_rate,
        path_to_plot_net_hh_income,
        path_to_plot_interest,
        path_to_plot_joint_gross_labor_income,
        path_to_plot_joint_gross_retirement_income,
        path_to_plot_gross_labor_income,
        path_to_plot_gross_retirement_income,
        path_to_plot_bequest_from_parent,
        path_to_plot_income_tax,
        path_to_plot_own_ssc,
        path_to_plot_partner_ssc,
        path_to_plot_total_tax_revenue,
        path_to_plot_government_expenditures,
        path_to_plot_net_government_budget,
        path_to_plot_exp_years,
        path_to_plot_own_income_after_ssc,
        path_to_plot_experience,
        path_to_plot_already_retired,
        path_to_plot_is_retired,
        path_to_plot_fresh_retired,
        path_to_plot_actual_retirement_age,
    ]

    for (col, ylabel, title, exclude_age), path in zip(
        plot_configs, path_params, strict=True
    ):
        _plot_asset_savings_outcome(
            df_sim=df_sim,
            specs=specs,
            outcome_col=col,
            ylabel=ylabel,
            title_suffix=title,
            path_to_plot=path,
            exclude_end_age=exclude_age,
        )


@pytask.mark.post_estimation_no_care_demand_model
@pytask.mark.no_care_demand_model
@pytask.mark.post_estimation
@pytask.mark.post_assets_and_savings
def task_plot_assets_and_savings_by_age_no_care_demand(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot_assets_begin: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "assets_begin_of_period_by_age.png",
    path_to_plot_savings_dec: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "savings_dec_by_age.png",
    path_to_plot_total_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "total_income_by_age.png",
    path_to_plot_savings: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "savings_by_age.png",
    path_to_plot_savings_rate: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "savings_rate_by_age.png",
    path_to_plot_net_hh_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "net_hh_income_by_age.png",
    path_to_plot_interest: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "interest_by_age.png",
    path_to_plot_joint_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "joint_gross_labor_income_by_age.png",
    path_to_plot_joint_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "joint_gross_retirement_income_by_age.png",
    path_to_plot_gross_labor_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "gross_labor_income_by_age.png",
    path_to_plot_gross_retirement_income: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "gross_retirement_income_by_age.png",
    path_to_plot_bequest_from_parent: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "bequest_from_parent_by_age.png",
    path_to_plot_income_tax: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "income_tax_by_age.png",
    path_to_plot_own_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "own_ssc_by_age.png",
    path_to_plot_partner_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "partner_ssc_by_age.png",
    path_to_plot_total_tax_revenue: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "total_tax_revenue_by_age.png",
    path_to_plot_government_expenditures: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "government_expenditures_by_age.png",
    path_to_plot_net_government_budget: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "net_government_budget_by_age.png",
    path_to_plot_income_shock_previous_period: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "income_shock_previous_period_by_age.png",
    path_to_plot_income_shock_for_labor: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "income_shock_for_labor_by_age.png",
    path_to_plot_exp_years: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "exp_years_by_age.png",
    path_to_plot_own_income_after_ssc: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "own_income_after_ssc_by_age.png",
    path_to_plot_experience: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "experience_by_age.png",
    path_to_plot_already_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "already_retired_by_age.png",
    path_to_plot_is_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "is_retired_by_age.png",
    path_to_plot_fresh_retired: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "fresh_retired_by_age.png",
    path_to_plot_actual_retirement_age: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "assets_and_savings"
    / "no_care_demand"
    / "actual_retirement_age_by_age.png",
):
    """Plot average assets and savings by age and education from no care  # noqa: E501
    demand counterfactual simulated data.

    Creates plots for all variables from budget equation aux dict and related variables.
    Each plot shows 2 lines (Low and High education).

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to no care demand counterfactual simulated data pkl file
    path_to_plot_* : Path
        Paths to save the plots

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

    # Calculate fresh_retired: (already_retired == 0) & (is_retired == 1)
    if "already_retired" in df_sim.columns and "is_retired" in df_sim.columns:
        df_sim["fresh_retired"] = (
            (df_sim["already_retired"] == 0) & (df_sim["is_retired"] == 1)
        ).astype(float)

    # Verify required columns exist
    required_cols = [
        "education",
        "assets_begin_of_period",
        "savings_dec",
        "total_income",
        "savings",
        "savings_rate",
    ]
    missing_cols = [col for col in required_cols if col not in df_sim.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Plot configurations: (column_name, ylabel, title_suffix, exclude_end_age)
    plot_configs = [
        (
            "assets_begin_of_period",
            "Average Assets Begin of Period (in 1,000€)",
            "Assets Begin of Period (No Care Demand)",
            False,
        ),
        (
            "savings_dec",
            "Average Savings Decision (in 1,000€)",
            "Savings Decision (No Care Demand)",
            True,
        ),
        (
            "total_income",
            "Average Total Income (in 1,000€)",
            "Total Income (No Care Demand)",
            False,
        ),
        ("savings", "Average Savings (in 1,000€)", "Savings (No Care Demand)", True),
        ("savings_rate", "Average Savings Rate", "Savings Rate (No Care Demand)", True),
        (
            "net_hh_income",
            "Average Net Household Income (in 1,000€)",
            "Net Household Income (No Care Demand)",
            False,
        ),
        (
            "interest",
            "Average Interest (in 1,000€)",
            "Interest (No Care Demand)",
            False,
        ),
        (
            "joint_gross_labor_income",
            "Average Joint Gross Labor Income (in 1,000€)",
            "Joint Gross Labor Income (No Care Demand)",
            False,
        ),
        (
            "joint_gross_retirement_income",
            "Average Joint Gross Retirement Income (in 1,000€)",
            "Joint Gross Retirement Income (No Care Demand)",
            False,
        ),
        (
            "gross_labor_income",
            "Average Gross Labor Income (in 1,000€)",
            "Gross Labor Income (No Care Demand)",
            False,
        ),
        (
            "gross_retirement_income",
            "Average Gross Retirement Income (in 1,000€)",
            "Gross Retirement Income (No Care Demand)",
            False,
        ),
        (
            "bequest_from_parent",
            "Average Bequest from Parent (in 1,000€)",
            "Bequest from Parent (No Care Demand)",
            False,
        ),
        (
            "income_tax",
            "Average Income Tax (in 1,000€)",
            "Income Tax (No Care Demand)",
            False,
        ),
        ("own_ssc", "Average Own SSC (in 1,000€)", "Own SSC (No Care Demand)", False),
        (
            "partner_ssc",
            "Average Partner SSC (in 1,000€)",
            "Partner SSC (No Care Demand)",
            False,
        ),
        (
            "total_tax_revenue",
            "Average Total Tax Revenue (in 1,000€)",
            "Total Tax Revenue (No Care Demand)",
            False,
        ),
        (
            "government_expenditures",
            "Average Government Expenditures (in 1,000€)",
            "Government Expenditures (No Care Demand)",
            False,
        ),
        (
            "net_government_budget",
            "Average Net Government Budget (in 1,000€)",
            "Net Government Budget (No Care Demand)",
            False,
        ),
        (
            "income_shock_previous_period",
            "Average Income Shock Previous Period",
            "Income Shock Previous Period (No Care Demand)",
            False,
        ),
        (
            "income_shock_for_labor",
            "Average Income Shock For Labor",
            "Income Shock For Labor (No Care Demand)",
            False,
        ),
        (
            "exp_years",
            "Average Experience Years",
            "Experience Years (No Care Demand)",
            False,
        ),
        (
            "own_income_after_ssc",
            "Average Own Income After SSC (in 1,000€)",
            "Own Income After SSC (No Care Demand)",
            False,
        ),
        (
            "experience",
            "Average Experience",
            "Experience (No Care Demand)",
            False,
        ),
        (
            "already_retired",
            "Average Already Retired",
            "Already Retired (No Care Demand)",
            False,
        ),
        (
            "is_retired",
            "Average Is Retired",
            "Is Retired (No Care Demand)",
            False,
        ),
        (
            "fresh_retired",
            "Average Fresh Retired",
            "Fresh Retired (No Care Demand)",
            False,
        ),
        (
            "actual_retirement_age",
            "Average Actual Retirement Age",
            "Actual Retirement Age (No Care Demand)",
            False,
        ),
    ]

    path_params = [
        path_to_plot_assets_begin,
        path_to_plot_savings_dec,
        path_to_plot_total_income,
        path_to_plot_savings,
        path_to_plot_savings_rate,
        path_to_plot_net_hh_income,
        path_to_plot_interest,
        path_to_plot_joint_gross_labor_income,
        path_to_plot_joint_gross_retirement_income,
        path_to_plot_gross_labor_income,
        path_to_plot_gross_retirement_income,
        path_to_plot_bequest_from_parent,
        path_to_plot_income_tax,
        path_to_plot_own_ssc,
        path_to_plot_partner_ssc,
        path_to_plot_total_tax_revenue,
        path_to_plot_government_expenditures,
        path_to_plot_net_government_budget,
        path_to_plot_income_shock_previous_period,
        path_to_plot_income_shock_for_labor,
        path_to_plot_exp_years,
        path_to_plot_own_income_after_ssc,
        path_to_plot_experience,
        path_to_plot_already_retired,
        path_to_plot_is_retired,
        path_to_plot_fresh_retired,
        path_to_plot_actual_retirement_age,
    ]

    for (col, ylabel, title, exclude_age), path in zip(
        plot_configs, path_params, strict=True
    ):
        _plot_asset_savings_outcome(
            df_sim=df_sim,
            specs=specs,
            outcome_col=col,
            ylabel=ylabel,
            title_suffix=title,
            path_to_plot=path,
            exclude_end_age=exclude_age,
        )


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


def _plot_asset_savings_outcome(  # noqa: PLR0912
    df_sim,
    specs,
    outcome_col,
    ylabel,
    title_suffix,
    path_to_plot,
    exclude_end_age=False,
):
    """Helper function to plot an asset/savings outcome by age and education.

    Parameters
    ----------
    df_sim : pd.DataFrame
        Simulated data
    specs : dict
        Model specifications
    outcome_col : str
        Column name to plot
    ylabel : str
        Y-axis label
    title_suffix : str
        Title suffix
    path_to_plot : Path
        Path to save the plot
    exclude_end_age : bool
        If True, exclude observations at the maximum age (end_age = 100)
    """
    # Filter out end_age if requested
    df_plot = df_sim.copy()
    if exclude_end_age:
        end_age = specs.get("end_age", df_sim["age"].max())
        df_plot = df_plot[df_plot["age"] < end_age].copy()

    # For experience, calculate min, max, and mean; otherwise just mean
    if outcome_col == "experience":
        stats_by_group = (
            df_plot.groupby(["age", "education"], observed=False)[outcome_col]
            .agg(["mean", "min", "max"])
            .reset_index()
        )
    else:
        # Calculate average by age and education
        avg_by_group = (
            df_plot.groupby(["age", "education"], observed=False)[outcome_col]
            .mean()
            .reset_index()
        )

    # Get all unique ages
    ages = np.sort(df_plot["age"].unique())

    # Education labels
    education_labels = specs.get("education_labels", ["Low", "High"])

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(education_labels))]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(education_labels):
        color = edu_colors[edu_var]

        if outcome_col == "experience":
            # Filter to this education level
            df_edu = stats_by_group.loc[stats_by_group["education"] == edu_var].copy()

            if len(df_edu) > 0:
                # Plot mean (average)
                mean_values = df_edu.set_index("age")["mean"].reindex(ages).values
                mean_mask = ~np.isnan(mean_values)
                if mean_mask.sum() > 0:
                    ax.plot(
                        ages[mean_mask],
                        mean_values[mean_mask],
                        linewidth=2,
                        color=color,
                        label=f"{edu_label} (Average)",
                        alpha=0.8,
                        linestyle="-",
                    )

                # Plot min (lowest)
                min_values = df_edu.set_index("age")["min"].reindex(ages).values
                min_mask = ~np.isnan(min_values)
                if min_mask.sum() > 0:
                    ax.plot(
                        ages[min_mask],
                        min_values[min_mask],
                        linewidth=1.5,
                        color=color,
                        label=f"{edu_label} (Lowest)",
                        alpha=0.6,
                        linestyle="--",
                    )

                # Plot max (highest)
                max_values = df_edu.set_index("age")["max"].reindex(ages).values
                max_mask = ~np.isnan(max_values)
                if max_mask.sum() > 0:
                    ax.plot(
                        ages[max_mask],
                        max_values[max_mask],
                        linewidth=1.5,
                        color=color,
                        label=f"{edu_label} (Highest)",
                        alpha=0.6,
                        linestyle=":",
                    )
        else:
            # Filter to this education level
            df_edu = avg_by_group.loc[avg_by_group["education"] == edu_var].copy()

            if len(df_edu) > 0:
                # Reindex to all ages and fill missing with NaN
                values = df_edu.set_index("age")[outcome_col].reindex(ages).values
                mask = ~np.isnan(values)
                if mask.sum() > 0:
                    ax.plot(
                        ages[mask],
                        values[mask],
                        linewidth=2,
                        color=color,
                        label=edu_label,
                        alpha=0.8,
                    )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # Update title for experience plots to reflect min/max/mean
    if outcome_col == "experience":
        ax.set_title(
            f"Experience (Min/Average/Max) {title_suffix} by Age and Education",
            fontsize=13,
        )
        ax.set_ylim(0, 1)
    else:
        ax.set_title(f"Average {title_suffix} by Age and Education", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {path_to_plot}")


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
