"""Plot caregiving leave top-up by age, education, and job before caregiving.

Creates multiple plots showing average caregiving leave top-up with different conditions:
- Conditional on caregiving_type == 1 (all observations)
- Conditional on caregiving_type == 1 AND caregiving_leave_top_up > 0
- Conditional on providing informal care (current period) AND caregiving_type == 1
- Conditional on currently working (PT or FT) AND caregiving_type == 1
- Conditional on caregiving_type == 1 only (explicit labeling)
- No conditioning (all caregiving_types)

All plots show 4 lines: Low/High education × PT/FT job before caregiving.
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
    FULL_TIME,
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    PART_TIME,
)


@pytask.mark.caregiving_leave_with_job_retention_model
@pytask.mark.post_estimation
@pytask.mark.post_caregiving_leave_top_up
def task_plot_caregiving_leave_top_up(  # noqa: PLR0913
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_caregiving_leave_with_job_retention_estimated_params.pkl",
    path_to_plot_all: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "caregiving_leave_top_up_by_age.png",
    path_to_plot_positive: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "caregiving_leave_top_up_by_age_positive_only.png",
    path_to_plot_caregiver: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "caregiving_leave_top_up_by_age_caregiver.png",
    path_to_plot_working: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "caregiving_leave_top_up_by_age_working.png",
    path_to_plot_type1_only: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "caregiving_leave_top_up_by_age_type1_only.png",
    path_to_plot_no_conditioning: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "caregiving_leave_top_up_by_age_no_conditioning.png",
):
    """Plot average caregiving leave top-up by age, education, and job before caregiving.

    Creates multiple plots with different conditioning:
    1. Average top-up conditional on caregiving_type == 1 (all observations)
    2. Average top-up conditional on caregiving_type == 1 AND top_up > 0
    3. Average top-up conditional on providing informal care (current period) AND caregiving_type == 1
    4. Average top-up conditional on currently working (PT or FT) AND caregiving_type == 1
    5. Average top-up conditional on caregiving_type == 1 only (explicit labeling)
    6. Average top-up with no conditioning (all caregiving_types)

    Each plot shows 4 lines:
    - Low education + Part-time job before caregiving
    - Low education + Full-time job before caregiving
    - High education + Part-time job before caregiving
    - High education + Full-time job before caregiving

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to caregiving leave counterfactual simulated data pkl file
    path_to_plot_all : Path
        Path to save the plot for all caregiving_type == 1 observations
    path_to_plot_positive : Path
        Path to save the plot for observations with top_up > 0
    path_to_plot_caregiver : Path
        Path to save the plot for observations where agent is providing informal care
    path_to_plot_working : Path
        Path to save the plot for observations where agent is currently working (PT or FT)
    path_to_plot_type1_only : Path
        Path to save the plot for caregiving_type == 1 only (explicit labeling)
    path_to_plot_no_conditioning : Path
        Path to save the plot with no conditioning (all caregiving_types)

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

    # Handle aux variables - they may be stored directly as columns or in an "aux" dict
    if "caregiving_leave_top_up" not in df_sim.columns:
        if "aux" in df_sim.columns:
            # Extract from aux dictionary column
            df_sim["caregiving_leave_top_up"] = df_sim["aux"].apply(
                lambda x: (
                    x.get("caregiving_leave_top_up", np.nan)
                    if isinstance(x, dict)
                    else np.nan
                )
            )
        else:
            raise ValueError(
                f"Missing 'caregiving_leave_top_up' column and no 'aux' column found. "
                f"Available columns: {df_sim.columns.tolist()}"
            )

    # Verify other required columns exist
    required_cols = [
        "job_before_caregiving",
        "caregiving_type",
        "education",
    ]
    missing_cols = [col for col in required_cols if col not in df_sim.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Create age variable if not already present
    if "age" not in df_sim.columns:
        df_sim["age"] = df_sim["period"] + specs["start_age"]

    # Filter to caregiving_type == 1 (only these can provide informal care and go on leave)
    df_filtered = df_sim.loc[df_sim["caregiving_type"] == 1].copy()

    # Create job_before_caregiving category
    # job_before_caregiving: 0 = none, 1 = PT, 2 = FT
    df_filtered["job_before_category"] = "None"
    df_filtered.loc[
        df_filtered["job_before_caregiving"] == 1, "job_before_category"
    ] = "PT"
    df_filtered.loc[
        df_filtered["job_before_caregiving"] == 2, "job_before_category"
    ] = "FT"

    # Filter to only PT and FT (exclude None)
    df_filtered = df_filtered.loc[
        df_filtered["job_before_caregiving"].isin([1, 2])
    ].copy()

    # Create education labels
    education_labels = specs.get("education_labels", ["Low", "High"])
    df_filtered["education_label"] = df_filtered["education"].map(
        {i: education_labels[i] for i in range(len(education_labels))}
    )

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(education_labels))]

    # Line styles for job before categories
    job_linestyles = {"PT": ":", "FT": "-"}  # PT: dotted, FT: solid

    # Prepare data for plotting
    def create_plot_data(data):
        """Create grouped data for plotting."""
        # Group by age, education, and job_before_category
        grouped = (
            data.groupby(
                ["age", "education_label", "job_before_category"], observed=False
            )["caregiving_leave_top_up"]
            .mean()
            .reset_index()
        )

        return grouped

    # Plot 1: All observations with caregiving_type == 1
    plot_data_all = create_plot_data(df_filtered)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each combination
    for edu_idx, edu_label in enumerate(education_labels):
        edu_color = edu_colors[edu_idx]
        for job_type in ["PT", "FT"]:
            subset = plot_data_all[
                (plot_data_all["education_label"] == edu_label)
                & (plot_data_all["job_before_category"] == job_type)
            ]
            if len(subset) > 0:
                label = f"{edu_label} edu, {job_type}"
                ax.plot(
                    subset["age"],
                    subset["caregiving_leave_top_up"],
                    label=label,
                    color=edu_color,
                    linestyle=job_linestyles[job_type],
                    linewidth=2,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Caregiving Leave Top-Up (in 1,000€)", fontsize=12)
    ax.set_title(
        "Average Caregiving Leave Top-Up by Age\n(Caregiving Type 1, PT/FT Job Before Caregiving)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure plot directory exists
    path_to_plot_all.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_to_plot_all, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Only observations with top_up > 0
    df_positive = df_filtered.loc[df_filtered["caregiving_leave_top_up"] > 0].copy()

    plot_data_positive = create_plot_data(df_positive)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each combination
    for edu_idx, edu_label in enumerate(education_labels):
        edu_color = edu_colors[edu_idx]
        for job_type in ["PT", "FT"]:
            subset = plot_data_positive[
                (plot_data_positive["education_label"] == edu_label)
                & (plot_data_positive["job_before_category"] == job_type)
            ]
            if len(subset) > 0:
                label = f"{edu_label} edu, {job_type}"
                ax.plot(
                    subset["age"],
                    subset["caregiving_leave_top_up"],
                    label=label,
                    color=edu_color,
                    linestyle=job_linestyles[job_type],
                    linewidth=2,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Caregiving Leave Top-Up (in 1,000€)", fontsize=12)
    ax.set_title(
        "Average Caregiving Leave Top-Up by Age (Positive Top-Up Only)\n"
        "(Caregiving Type 1, PT/FT Job Before Caregiving, Top-Up > 0)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure plot directory exists
    path_to_plot_positive.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_to_plot_positive, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Only observations where agent is providing informal care
    # Get informal care choice values
    light_care_values = LIGHT_INFORMAL_CARE.ravel().tolist()
    intensive_care_values = INTENSIVE_INFORMAL_CARE.ravel().tolist()
    informal_care_values = light_care_values + intensive_care_values

    # Filter to observations providing informal care AND caregiving_type == 1
    # AND job_before_caregiving is PT or FT
    df_caregiver = df_filtered.loc[
        df_filtered["choice"].isin(informal_care_values)
    ].copy()

    plot_data_caregiver = create_plot_data(df_caregiver)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each combination
    for edu_idx, edu_label in enumerate(education_labels):
        edu_color = edu_colors[edu_idx]
        for job_type in ["PT", "FT"]:
            subset = plot_data_caregiver[
                (plot_data_caregiver["education_label"] == edu_label)
                & (plot_data_caregiver["job_before_category"] == job_type)
            ]
            if len(subset) > 0:
                label = f"{edu_label} edu, {job_type}"
                ax.plot(
                    subset["age"],
                    subset["caregiving_leave_top_up"],
                    label=label,
                    color=edu_color,
                    linestyle=job_linestyles[job_type],
                    linewidth=2,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Caregiving Leave Top-Up (in 1,000€)", fontsize=12)
    ax.set_title(
        "Average Caregiving Leave Top-Up by Age (Providing Informal Care)\n"
        "(Caregiving Type 1, PT/FT Job Before Caregiving, Currently Providing Informal Care)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure plot directory exists
    path_to_plot_caregiver.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_to_plot_caregiver, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Only observations where agent is currently working (PT or FT)
    # Get part-time and full-time choice values
    part_time_values = PART_TIME.ravel().tolist()
    full_time_values = FULL_TIME.ravel().tolist()
    working_values = part_time_values + full_time_values

    # Filter to observations where agent is working AND caregiving_type == 1
    # AND job_before_caregiving is PT or FT
    df_working = df_filtered.loc[df_filtered["choice"].isin(working_values)].copy()

    plot_data_working = create_plot_data(df_working)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each combination
    for edu_idx, edu_label in enumerate(education_labels):
        edu_color = edu_colors[edu_idx]
        for job_type in ["PT", "FT"]:
            subset = plot_data_working[
                (plot_data_working["education_label"] == edu_label)
                & (plot_data_working["job_before_category"] == job_type)
            ]
            if len(subset) > 0:
                label = f"{edu_label} edu, {job_type}"
                ax.plot(
                    subset["age"],
                    subset["caregiving_leave_top_up"],
                    label=label,
                    color=edu_color,
                    linestyle=job_linestyles[job_type],
                    linewidth=2,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Caregiving Leave Top-Up (in 1,000€)", fontsize=12)
    ax.set_title(
        "Average Caregiving Leave Top-Up by Age (Currently Working)\n"
        "(Caregiving Type 1, PT/FT Job Before Caregiving, Currently Working PT/FT)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure plot directory exists
    path_to_plot_working.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_to_plot_working, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 5: Only caregiving_type == 1 (explicit labeling, same as Plot 1 but with different title)
    # This is the same data as Plot 1, but with explicit labeling in the title
    plot_data_type1 = create_plot_data(df_filtered)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each combination
    for edu_idx, edu_label in enumerate(education_labels):
        edu_color = edu_colors[edu_idx]
        for job_type in ["PT", "FT"]:
            subset = plot_data_type1[
                (plot_data_type1["education_label"] == edu_label)
                & (plot_data_type1["job_before_category"] == job_type)
            ]
            if len(subset) > 0:
                label = f"{edu_label} edu, {job_type}"
                ax.plot(
                    subset["age"],
                    subset["caregiving_leave_top_up"],
                    label=label,
                    color=edu_color,
                    linestyle=job_linestyles[job_type],
                    linewidth=2,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Caregiving Leave Top-Up (in 1,000€)", fontsize=12)
    ax.set_title(
        "Average Caregiving Leave Top-Up by Age (Caregiving Type 1 Only)\n"
        "(PT/FT Job Before Caregiving)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure plot directory exists
    path_to_plot_type1_only.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_to_plot_type1_only, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 6: No conditioning (all caregiving_types)
    # Prepare data with no conditioning on caregiving_type
    df_all_types = df_sim.copy()

    # Create job_before_caregiving category for all types
    df_all_types["job_before_category"] = "None"
    df_all_types.loc[
        df_all_types["job_before_caregiving"] == 1, "job_before_category"
    ] = "PT"
    df_all_types.loc[
        df_all_types["job_before_caregiving"] == 2, "job_before_category"
    ] = "FT"

    # Filter to only PT and FT (exclude None)
    df_all_types = df_all_types.loc[
        df_all_types["job_before_caregiving"].isin([1, 2])
    ].copy()

    # Create education labels
    df_all_types["education_label"] = df_all_types["education"].map(
        {i: education_labels[i] for i in range(len(education_labels))}
    )

    plot_data_no_conditioning = create_plot_data(df_all_types)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each combination
    for edu_idx, edu_label in enumerate(education_labels):
        edu_color = edu_colors[edu_idx]
        for job_type in ["PT", "FT"]:
            subset = plot_data_no_conditioning[
                (plot_data_no_conditioning["education_label"] == edu_label)
                & (plot_data_no_conditioning["job_before_category"] == job_type)
            ]
            if len(subset) > 0:
                label = f"{edu_label} edu, {job_type}"
                ax.plot(
                    subset["age"],
                    subset["caregiving_leave_top_up"],
                    label=label,
                    color=edu_color,
                    linestyle=job_linestyles[job_type],
                    linewidth=2,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Caregiving Leave Top-Up (in 1,000€)", fontsize=12)
    ax.set_title(
        "Average Caregiving Leave Top-Up by Age (No Conditioning)\n"
        "(All Caregiving Types, PT/FT Job Before Caregiving)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure plot directory exists
    path_to_plot_no_conditioning.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_to_plot_no_conditioning, dpi=300, bbox_inches="tight")
    plt.close()
