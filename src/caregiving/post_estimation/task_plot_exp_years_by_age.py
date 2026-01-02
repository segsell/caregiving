"""Plot average experience years by age and education from simulated model data.

Creates three plots showing average experience years with 2 lines per plot:
- Baseline model
- Caregiving leave counterfactual
- No care demand counterfactual

Each plot shows 2 lines: Low education and High education.
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
def task_plot_exp_years_by_age_baseline(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "exp_years_by_age_baseline.png",
):
    """Plot average experience years by age and education from baseline simulated data.

    Creates one plot with 2 lines (Low and High education).

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
            raise ValueError(
                f"Cannot find 'agent' column or index level. "
                f"Available columns: {df_sim.columns.tolist()}, "
                f"Index names: {df_sim.index.names if hasattr(df_sim.index, 'names') else 'N/A'}"
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

    # Verify exp_years column exists
    if "exp_years" not in df_sim.columns:
        raise ValueError(
            f"Missing 'exp_years' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Verify education column exists
    if "education" not in df_sim.columns:
        raise ValueError(
            f"Missing 'education' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Calculate average experience years by age and education
    avg_exp_by_group = (
        df_sim.groupby(["age", "education"], observed=False)["exp_years"]
        .mean()
        .reset_index()
    )

    # Get all unique ages
    ages = np.sort(df_sim["age"].unique())

    # Education labels
    education_labels = specs.get("education_labels", ["Low", "High"])

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(education_labels))]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(education_labels):
        color = edu_colors[edu_var]

        # Filter to this education level
        df_edu = avg_exp_by_group.loc[avg_exp_by_group["education"] == edu_var].copy()

        if len(df_edu) > 0:
            # Reindex to all ages and fill missing with NaN
            exp_values = df_edu.set_index("age")["exp_years"].reindex(ages).values
            mask = ~np.isnan(exp_values)
            if mask.sum() > 0:
                ax.plot(
                    ages[mask],
                    exp_values[mask],
                    linewidth=2,
                    color=color,
                    label=edu_label,
                    alpha=0.8,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Experience Years", fontsize=12)
    ax.set_title(
        "Average Experience Years by Age and Education (Baseline Model)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Experience years by age plot (baseline) saved to {path_to_plot}")


@pytask.mark.caregiving_leave_with_job_retention_model
@pytask.mark.post_estimation
def task_plot_exp_years_by_age_caregiving_leave(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "exp_years_by_age_caregiving_leave.png",
):
    """Plot average experience years by age and education from caregiving leave counterfactual simulated data.

    Creates one plot with 2 lines (Low and High education).

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to caregiving leave counterfactual simulated data pkl file
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
            raise ValueError(
                f"Cannot find 'agent' column or index level. "
                f"Available columns: {df_sim.columns.tolist()}, "
                f"Index names: {df_sim.index.names if hasattr(df_sim.index, 'names') else 'N/A'}"
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

    # Verify exp_years column exists
    if "exp_years" not in df_sim.columns:
        raise ValueError(
            f"Missing 'exp_years' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Verify education column exists
    if "education" not in df_sim.columns:
        raise ValueError(
            f"Missing 'education' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Calculate average experience years by age and education
    avg_exp_by_group = (
        df_sim.groupby(["age", "education"], observed=False)["exp_years"]
        .mean()
        .reset_index()
    )

    # Get all unique ages
    ages = np.sort(df_sim["age"].unique())

    # Education labels
    education_labels = specs.get("education_labels", ["Low", "High"])

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(education_labels))]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(education_labels):
        color = edu_colors[edu_var]

        # Filter to this education level
        df_edu = avg_exp_by_group.loc[avg_exp_by_group["education"] == edu_var].copy()

        if len(df_edu) > 0:
            # Reindex to all ages and fill missing with NaN
            exp_values = df_edu.set_index("age")["exp_years"].reindex(ages).values
            mask = ~np.isnan(exp_values)
            if mask.sum() > 0:
                ax.plot(
                    ages[mask],
                    exp_values[mask],
                    linewidth=2,
                    color=color,
                    label=edu_label,
                    alpha=0.8,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Experience Years", fontsize=12)
    ax.set_title(
        "Average Experience Years by Age and Education (Caregiving Leave)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Experience years by age plot (caregiving leave) saved to {path_to_plot}")


@pytask.mark.no_care_demand_model
@pytask.mark.post_estimation
def task_plot_exp_years_by_age_no_care_demand(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_no_care_demand.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "exp_years_by_age_no_care_demand.png",
):
    """Plot average experience years by age and education from no care demand counterfactual simulated data.

    Creates one plot with 2 lines (Low and High education).

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to no care demand counterfactual simulated data pkl file
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
            raise ValueError(
                f"Cannot find 'agent' column or index level. "
                f"Available columns: {df_sim.columns.tolist()}, "
                f"Index names: {df_sim.index.names if hasattr(df_sim.index, 'names') else 'N/A'}"
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

    # Verify exp_years column exists
    if "exp_years" not in df_sim.columns:
        raise ValueError(
            f"Missing 'exp_years' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Verify education column exists
    if "education" not in df_sim.columns:
        raise ValueError(
            f"Missing 'education' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Calculate average experience years by age and education
    avg_exp_by_group = (
        df_sim.groupby(["age", "education"], observed=False)["exp_years"]
        .mean()
        .reset_index()
    )

    # Get all unique ages
    ages = np.sort(df_sim["age"].unique())

    # Education labels
    education_labels = specs.get("education_labels", ["Low", "High"])

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(education_labels))]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(education_labels):
        color = edu_colors[edu_var]

        # Filter to this education level
        df_edu = avg_exp_by_group.loc[avg_exp_by_group["education"] == edu_var].copy()

        if len(df_edu) > 0:
            # Reindex to all ages and fill missing with NaN
            exp_values = df_edu.set_index("age")["exp_years"].reindex(ages).values
            mask = ~np.isnan(exp_values)
            if mask.sum() > 0:
                ax.plot(
                    ages[mask],
                    exp_values[mask],
                    linewidth=2,
                    color=color,
                    label=edu_label,
                    alpha=0.8,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Experience Years", fontsize=12)
    ax.set_title(
        "Average Experience Years by Age and Education (No Care Demand)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Experience years by age plot (no care demand) saved to {path_to_plot}")
