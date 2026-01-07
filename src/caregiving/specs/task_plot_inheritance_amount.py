"""Plot inheritance amount by age, education, and caregiving type for women."""

import pickle
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP


@pytask.mark.inheritance_specs
def task_plot_inheritance_amount_by_age(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "specs"
    / "inheritance_amount_by_age.png",
) -> None:
    """Plot inheritance amount for women by age, education, and caregiving type.

    Plots inheritance_amount from specs for women only, broken down by:
    - Education: Low (blue) and High (orange)
    - Caregiving type: Intensive care (solid), Light care (dashed), No care (dotted)

    Legend has two columns:
    - Left: Low education (blue) with intensive, light, no care
    - Right: High education (orange) with intensive, light, no care
    """
    # Load specs
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    # Get inheritance amount matrix
    # Shape: (n_sexes, n_periods, n_education, 3)
    # Last dimension: [no_care, light_care, intensive_care]
    inheritance_amount_mat = np.array(specs["inheritance_amount_mat"])

    # Get metadata
    start_age = specs["start_age"]
    end_age_caregiving = specs["end_age_caregiving"]
    sex_labels = specs["sex_labels"]  # ["Men", "Women"]
    education_labels = specs["education_labels"]  # ["Low", "High"]

    # Map sex to index
    sex_to_idx = {sex: idx for idx, sex in enumerate(sex_labels)}
    women_idx = sex_to_idx["Women"]

    # Create age array from start_age to 80 (inclusive)
    age_max = 80
    ages = np.arange(start_age, min(age_max + 1, specs["end_age"] + 1))

    # Get data for all periods
    n_periods_full = specs["end_age"] - specs["start_age"] + 1
    women_data_full = inheritance_amount_mat[women_idx, :n_periods_full, :, :]

    # Extract data for the relevant ages
    # Map ages to periods (period = age - start_age)
    periods = ages - start_age
    women_data = women_data_full[periods, :, :]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define line styles
    line_styles = {
        2: "-",  # intensive care: solid
        1: "--",  # light care: dashed
        0: ":",  # no care: dotted
    }

    # Define care type labels
    care_labels = {
        2: "Intensive care",
        1: "Light care",
        0: "No informal care",
    }

    # Plot lines for each combination and create legend
    # Order: Intensive (2), Light (1), No care (0) for each education
    legend_handles = []
    legend_labels = []

    for care_type_idx in [2, 1, 0]:  # Intensive, Light, No care
        for edu_idx in [0, 1]:  # Low education, High education
            color = JET_COLOR_MAP[edu_idx]  # Blue for low, orange for high

            # Extract data for this education and care type
            data = women_data[:, edu_idx, care_type_idx]

            # Plot
            line = ax.plot(
                ages,
                data,
                color=color,
                linestyle=line_styles[care_type_idx],
                linewidth=2,
            )[0]

            # Add to legend: education type + comma + care type
            # Use education label as-is (assuming it's already "Low" or "High")
            edu_label = education_labels[edu_idx]
            legend_handles.append(line)
            legend_labels.append(f"{edu_label}, {care_labels[care_type_idx]}")

    # Create single vertical legend
    leg = ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        frameon=True,
        fontsize=9,
        handlelength=2.5,
    )

    # Set labels and limits
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("Bequest", fontsize=11)

    # X-axis: start at start_age with padding, end at 80 with padding
    x_padding = 2
    ax.set_xlim(start_age - x_padding, age_max + x_padding)

    # Y-axis: start at 0
    ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.3)

    # Format y-axis with euro sign and thousand comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"€{x:,.0f}"))

    # Save plot
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {path_to_save_plot}")

    plt.close(fig)


@pytask.mark.inheritance_specs
def task_plot_inheritance_probability_by_age(
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "specs"
    / "inheritance_probability_by_age.png",
) -> None:
    """Plot inheritance probability for women by age, education, and caregiving type.

    Plots inheritance_prob_mat from specs for women only, broken down by:
    - Education: Low (blue) and High (orange)
    - Caregiving type: Informal care (solid), No informal care (dashed)
    Note: Light and intensive care have the same probability, so combined as "Informal care"
    """
    # Load specs
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    # Get inheritance probability matrix
    # Shape: (n_sexes, n_periods, n_education, 3)
    # Last dimension: [no_care, light_care, intensive_care]
    # Note: light_care and intensive_care have the same probability (both use any_care=1)
    inheritance_prob_mat = np.array(specs["inheritance_prob_mat"])

    # Get metadata
    start_age = specs["start_age"]
    sex_labels = specs["sex_labels"]  # ["Men", "Women"]
    education_labels = specs["education_labels"]  # ["Low", "High"]

    # Map sex to index
    sex_to_idx = {sex: idx for idx, sex in enumerate(sex_labels)}
    women_idx = sex_to_idx["Women"]

    # Create age array from start_age to 80 (inclusive)
    age_max = 80
    ages = np.arange(start_age, min(age_max + 1, specs["end_age"] + 1))

    # Get data for all periods
    n_periods_full = specs["end_age"] - specs["start_age"] + 1
    women_data_full = inheritance_prob_mat[women_idx, :n_periods_full, :, :]

    # Extract data for the relevant ages
    # Map ages to periods (period = age - start_age)
    periods = ages - start_age
    women_data = women_data_full[periods, :, :]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define line styles
    # Only two categories: Informal care and No informal care
    line_styles = {
        "informal_care": "-",  # solid
        "no_informal_care": "--",  # dashed
    }

    # Define care type labels
    care_labels = {
        "informal_care": "Informal care",
        "no_informal_care": "No informal care",
    }

    # Plot lines for each combination and create legend
    # Order: Informal care, No informal care for each education
    legend_handles = []
    legend_labels = []

    # Care type order: Informal care first, then No informal care
    care_types = [
        ("informal_care", 1),  # Use index 1 (light_care) - same as intensive_care
        ("no_informal_care", 0),  # Use index 0 (no_care)
    ]

    for care_type_name, care_type_idx in care_types:
        for edu_idx in [0, 1]:  # Low education, High education
            color = JET_COLOR_MAP[edu_idx]  # Blue for low, orange for high

            # Extract data for this education and care type
            data = women_data[:, edu_idx, care_type_idx]

            # Plot
            line = ax.plot(
                ages,
                data,
                color=color,
                linestyle=line_styles[care_type_name],
                linewidth=2,
            )[0]

            # Add to legend: education type + comma + care type
            edu_label = education_labels[edu_idx]
            legend_handles.append(line)
            legend_labels.append(f"{edu_label}, {care_labels[care_type_name]}")

    # Create single vertical legend
    leg = ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        frameon=True,
        fontsize=9,
        handlelength=2.5,
    )

    # Set labels and limits
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)

    # X-axis: start at start_age with padding, end at 80 with padding
    x_padding = 2
    ax.set_xlim(start_age - x_padding, age_max + x_padding)

    # Y-axis: start at 0, end at 20%
    ax.set_ylim(0, 0.2)

    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

    # Save plot
    path_to_save_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_to_save_plot, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {path_to_save_plot}")

    plt.close(fig)
