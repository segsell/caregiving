"""Test functions to plot inheritance probability and amount from specs dictionary.

This test reproduces the inheritance plots using the precomputed matrices
stored in the specs dictionary, similar to the task_plot_inheritance_* modules.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from caregiving.config import BLD, JET_COLOR_MAP


def test_plot_inheritance_probability_from_specs():
    """Plot inheritance probability from specs dictionary.

    Creates a plot showing inheritance probability by age, education, and care type.
    Different lines for:
    - Education levels (low/high)
    - Care types (no care, formal care, light care, intensive care)

    Plot is saved in tests/plotting/ directory.
    """
    # Load specs dictionary
    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    # Get age range
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    ages = np.arange(start_age, end_age + 1)

    # Get probability matrix
    inheritance_prob_mat = specs["inheritance_prob_mat"]

    # Care type labels matching the order in shared.py
    # Index 0: no_care, Index 1: formal_care, Index 2: light_care,
    # Index 3: intensive_care
    care_type_labels = ["no care", "formal care", "light care", "intensive care"]
    care_type_indices = [0, 1, 2, 3]

    # Line styles for different care types
    care_line_styles = {
        0: ":",  # no care: dotted
        1: (0, (3, 1, 1, 1, 1, 1)),  # formal care: dotted and dashed mix
        2: "--",  # light care: dashed
        3: "-",  # intensive care: solid
    }

    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]

        # Plot for each education level
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            color = JET_COLOR_MAP[edu_var]

            # Plot for each care type
            for care_idx, care_label in zip(
                care_type_indices, care_type_labels, strict=True
            ):
                # Extract probabilities for this sex, education, and care type
                # Shape: (n_sexes, n_periods, n_education, n_care_types)
                probs = inheritance_prob_mat[sex_var, :, edu_var, care_idx]

                ax.plot(
                    ages,
                    probs,
                    linewidth=2.5,
                    color=color,
                    linestyle=care_line_styles[care_idx],
                    label=f"{edu_label}, {care_label}",
                )

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Probability of Inheritance", fontsize=12)
        ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(start_age, end_age)
        ax.set_ylim(0, None)

    plt.suptitle(
        "Inheritance Probability by Age, Education, and Care Type\n"
        "(From Precomputed Specs Matrix)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    path_to_save = output_dir / "inheritance_probability_from_specs.png"
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved inheritance probability plot to: {path_to_save}")


def test_plot_inheritance_amount_from_specs():
    """Plot inheritance amount from specs dictionary.

    Creates a plot showing inheritance amount by age, education, and care type.
    Different lines for:
    - Education levels (low/high)
    - Care types (no care, formal care, light care, intensive care)

    Plot is saved in tests/plotting/ directory.
    """
    # Load specs dictionary
    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"
    with path_to_specs.open("rb") as f:
        specs = pickle.load(f)

    # Get age range
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    ages = np.arange(start_age, end_age + 1)

    # Get amount matrix
    inheritance_amount_mat = specs["inheritance_amount_mat"]

    # Care type labels matching the order in shared.py
    # Index 0: no_care, Index 1: formal_care, Index 2: light_care,
    # Index 3: intensive_care
    care_type_labels = ["no care", "formal care", "light care", "intensive care"]
    care_type_indices = [0, 1, 2, 3]

    # Line styles for different care types
    care_line_styles = {
        0: ":",  # no care: dotted
        1: (0, (3, 1, 1, 1, 1, 1)),  # formal care: dotted and dashed mix
        2: "--",  # light care: dashed
        3: "-",  # intensive care: solid
    }

    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]

        # Plot for each education level
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            color = JET_COLOR_MAP[edu_var]

            # Plot for each care type
            for care_idx, care_label in zip(
                care_type_indices, care_type_labels, strict=True
            ):
                # Extract amounts for this sex, education, and care type
                # Shape: (n_sexes, n_periods, n_education, n_care_types)
                amounts = inheritance_amount_mat[sex_var, :, edu_var, care_idx]

                ax.plot(
                    ages,
                    amounts,
                    linewidth=2.5,
                    color=color,
                    linestyle=care_line_styles[care_idx],
                    label=f"{edu_label}, {care_label}",
                )

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Inheritance Amount (€)", fontsize=12)
        ax.set_title(str(sex_label), fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(start_age, end_age)
        ax.set_ylim(0, None)

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"€{x:,.0f}"))

    plt.suptitle(
        "Inheritance Amount by Age, Education, and Care Type\n"
        "(From Precomputed Specs Matrix)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    path_to_save = output_dir / "inheritance_amount_from_specs.png"
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved inheritance amount plot to: {path_to_save}")
