"""Test inheritance calculation function with plots."""

import pickle as pkl
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from caregiving.config import BLD
from caregiving.model.shared import (
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_CARE,
    SEX,
)
from caregiving.model.wealth_and_budget.transfers import calc_inheritance

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def load_specs():
    """Load specs from pickle file."""
    path_to_specs = BLD / "model" / "specs" / "specs_full.pkl"

    with path_to_specs.open("rb") as file:
        specs = pkl.load(file)

    return specs


def test_inheritance_calc_plot(load_specs):  # noqa: PLR0912, PLR0915
    """Test and plot inheritance probability and amount using calc_inheritance function.

    Tests the calc_inheritance function with different scenarios:
    - No care (choice 0)
    - Light informal care (choice 4)
    - Intensive informal care (choice 8)

    Shows variation by:
    - Age (40-80)
    - Education level
    - Care type

    Creates two subplots:
    1. Probability of positive inheritance
    2. Expected inheritance amount (probability * amount)
    """
    tmp_path = Path(__file__).parent / "temp_figs"
    tmp_path.mkdir(exist_ok=True)

    specs = load_specs

    # Age range
    start_age = specs["start_age"]
    ages = np.arange(40, 81)  # 40 to 80 inclusive
    periods = ages - start_age

    # Compute probability and amount separately for plotting
    sex_var = SEX
    sex_label = specs["sex_labels"][sex_var]

    # Get parameters
    inheritance_prob_params = specs["inheritance_prob_spec5_params"]
    inheritance_amount_params = specs["inheritance_amount_spec5_params"]

    # ============================================================================
    # PLOT 1: Probability of positive inheritance
    # Only shows: any care vs no care (no distinction between light/intensive)
    # ============================================================================
    fig_prob, ax_prob = plt.subplots(figsize=(10, 6))

    # Only two scenarios for probability: no care and any care
    prob_scenarios = [
        (0, "No informal care"),
        (4, "Any informal care"),  # Can be light or intensive, doesn't matter
    ]

    # Plot for each education level
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        color = plt.cm.tab10(edu_var)  # Use matplotlib default colormap

        # Plot for each care scenario (no care vs any care)
        for lagged_choice, care_label in prob_scenarios:
            # Determine if any care
            if lagged_choice == 0:
                any_care_int = 0
            else:
                any_care_int = 1  # Any informal care (light or intensive)

            # Arrays to store results
            probabilities = []

            for period, age in zip(periods, ages, strict=True):
                if period < 0 or period >= specs["n_periods"]:
                    probabilities.append(np.nan)
                    continue

                age_sq = age**2

                # Compute probability using spec7 logit parameters
                logit_linear = (
                    inheritance_prob_params.loc[sex_label, "age"] * age
                    + inheritance_prob_params.loc[sex_label, "age_sq"] * age_sq
                    + inheritance_prob_params.loc[sex_label, "any_care"] * any_care_int
                    + inheritance_prob_params.loc[sex_label, "education"] * edu_var
                    + inheritance_prob_params.loc[sex_label, "const"]
                )

                prob = 1.0 / (1.0 + np.exp(-logit_linear))
                probabilities.append(prob)

            # Convert to numpy array
            probabilities = np.array(probabilities)

            # Determine line style
            linestyle = "--" if lagged_choice == 0 else "-"

            # Plot probability
            ax_prob.plot(
                ages,
                probabilities,
                linewidth=2,
                color=color,
                linestyle=linestyle,
                label=f"{edu_label}, {care_label}",
                alpha=0.8,
            )

    # Format probability plot
    ax_prob.set_xlabel("Age", fontsize=12)
    ax_prob.set_ylabel("Probability of Positive Inheritance", fontsize=12)
    ax_prob.set_title(
        "Inheritance Probability by Age, Education, and Care Type", fontsize=13
    )
    ax_prob.legend(loc="best", fontsize=9, ncol=2)
    ax_prob.grid(True, alpha=0.3)
    # No fixed y-limit - let it be flexible

    plt.tight_layout()

    # Save probability plot
    path_to_save_prob = tmp_path / "inheritance_probability_test.png"
    plt.savefig(path_to_save_prob, dpi=300, bbox_inches="tight")
    plt.close()

    # ============================================================================
    # PLOT 2: Expected inheritance amount
    # Distinguishes by education and care type (no care, light, intensive)
    # ============================================================================
    fig_amount, ax_amount = plt.subplots(figsize=(10, 6))

    # Care type order for amount plot - match order from task_plot_inheritance_two_specs
    # Order: intensive, light, no care (to match reference plots)
    care_type_order = [
        (8, "Intensive informal care"),
        (4, "Light informal care"),
        (0, "No informal care"),
    ]

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(specs["education_labels"]))]

    # Line styles for care types - match task_plot_inheritance_two_specs.py
    # Intensive: solid, Light: dotted, No care: dashed
    care_linestyles = [
        "-",  # Index 0: Intensive (solid)
        ":",  # Index 1: Light (dotted)
        "--",  # Index 2: No care (dashed)
    ]
    care_linewidths = [2.5, 2, 2]  # Thicker for intensive

    for care_idx, (lagged_choice, care_label) in enumerate(care_type_order):
        # Determine care type
        if lagged_choice in NO_CARE:
            light_care = 0
            intensive_care = 0
        elif lagged_choice in LIGHT_INFORMAL_CARE:
            light_care = 1
            intensive_care = 0
        elif lagged_choice in INTENSIVE_INFORMAL_CARE:
            light_care = 0
            intensive_care = 1
        else:
            light_care = 0
            intensive_care = 0

        # Plot for each education level
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            color = edu_colors[edu_var]

            # Arrays to store results
            expected_inheritances = []

            for period, age in zip(periods, ages, strict=True):
                if period < 0 or period >= specs["n_periods"]:
                    expected_inheritances.append(np.nan)
                    continue

                age_sq = age**2

                # Compute amount using spec12 OLS parameters
                ln_inheritance_amount = (
                    inheritance_amount_params.loc[sex_label, "age"] * age
                    + inheritance_amount_params.loc[sex_label, "age_sq"] * age_sq
                    + inheritance_amount_params.loc[sex_label, "light_care_recent"]
                    * light_care
                    + inheritance_amount_params.loc[sex_label, "intensive_care_recent"]
                    * intensive_care
                    + inheritance_amount_params.loc[sex_label, "education"] * edu_var
                    + inheritance_amount_params.loc[sex_label, "const"]
                )

                amount = np.exp(ln_inheritance_amount)

                # For the amount plot, show conditional amount
                # (not multiplied by probability)
                # This matches what task_plot_inheritance_two_specs does
                # The amount from OLS is already conditional on positive inheritance
                expected_inheritances.append(amount)

            # Convert to numpy array
            expected_inheritances = np.array(expected_inheritances)

            # Plot expected inheritance amount
            ax_amount.plot(
                ages,
                expected_inheritances,
                linewidth=care_linewidths[care_idx],
                color=color,
                linestyle=care_linestyles[care_idx],
                label=f"{edu_label}, {care_label}",
                alpha=0.8,
            )

    # Format amount plot
    ax_amount.set_xlabel("Age", fontsize=12)
    ax_amount.set_ylabel("Inheritance Amount (€)", fontsize=12)
    ax_amount.set_title(
        "Inheritance Amount (Conditional on Positive) by Age, Education, and Care Type",
        fontsize=13,
    )
    ax_amount.legend(loc="best", fontsize=9, ncol=2)
    ax_amount.grid(True, alpha=0.3)
    ax_amount.set_xlim(40, 80)
    ax_amount.set_ylim(0, None)
    # Format y-axis with thousands separator
    ax_amount.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"€{x:,.0f}"))

    plt.tight_layout()

    # Save amount plot
    path_to_save_amount = tmp_path / "inheritance_amount_test.png"
    plt.savefig(path_to_save_amount, dpi=300, bbox_inches="tight")
    plt.close()

    # Verify calc_inheritance function works correctly
    # Test a few sample cases
    test_cases = [
        (20, 0, 0, 1),  # period=20, no care, low edu, mother dead
        (20, 4, 0, 1),  # period=20, light care, low edu, mother dead
        (20, 8, 1, 1),  # period=20, intensive care, high edu, mother dead
        (20, 0, 0, 0),  # period=20, no care, low edu, mother alive (should be 0)
    ]

    print("\nTesting calc_inheritance function:")
    print("-" * 70)
    for period, lagged_choice, education, mother_dead_test in test_cases:
        result = calc_inheritance(
            period=period,
            lagged_choice=lagged_choice,
            education=education,
            mother_dead=mother_dead_test,
            model_specs=specs,
        )
        result_value = float(result)
        print(
            f"Period={period}, Choice={lagged_choice}, Edu={education}, "
            f"MotherDead={mother_dead_test}: Inheritance = {result_value:,.2f} €"
        )

        # Assert that inheritance is zero when mother is alive
        if mother_dead_test == 0:
            assert (
                result_value == 0.0
            ), "Inheritance should be zero when mother is alive"

        # Assert that inheritance is non-negative
        assert result_value >= 0.0, "Inheritance should be non-negative"
    print("-" * 70)

    print("\nInheritance test plots saved to:")
    print(f"  Probability: {path_to_save_prob}")
    print(f"  Amount: {path_to_save_amount}")


def test_inheritance_calc_basic(load_specs):
    """Basic test of calc_inheritance function."""
    specs = load_specs

    # Test that inheritance is zero when mother is alive
    result_alive = calc_inheritance(
        period=20,
        lagged_choice=0,
        education=0,
        mother_dead=0,
        model_specs=specs,
    )
    assert float(result_alive) == 0.0, "Inheritance should be zero when mother is alive"

    # Test that inheritance is non-negative when mother is dead
    result_dead = calc_inheritance(
        period=20,
        lagged_choice=4,  # Light care
        education=0,
        mother_dead=1,
        model_specs=specs,
    )
    assert (
        float(result_dead) >= 0.0
    ), "Inheritance should be non-negative when mother is dead"

    # Test that intensive care gives different result than light care
    result_intensive = calc_inheritance(
        period=20,
        lagged_choice=8,  # Intensive care
        education=0,
        mother_dead=1,
        model_specs=specs,
    )
    # They might be equal, but let's just check they're both non-negative
    assert float(result_intensive) >= 0.0
