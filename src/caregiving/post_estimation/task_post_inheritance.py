"""Plot inheritance by age from simulated baseline model data.

Creates plots showing inheritance patterns with 6 lines per plot:
- 2 education levels (Low, High) × 3 care types (Intensive, Light, No care)
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
    INTENSIVE_INFORMAL_CARE,
    LIGHT_INFORMAL_CARE,
    NO_CARE,
    PARENT_RECENTLY_DEAD,
)


@pytask.mark.baseline_model
@pytask.mark.post_estimation
@pytask.mark.post_estimation_inheritance
def task_plot_inheritance_by_age(  # noqa: PLR0915
    path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_simulated_data: Path = BLD
    / "solve_and_simulate"
    / "simulated_data_estimated_params.pkl",
    path_to_save_share_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "inheritance_share_by_age.png",
    path_to_save_share_lagged_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "inheritance_share_by_age_lagged_mother_dead.png",
    path_to_save_mother_dead_share_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "mother_recently_dead_share_by_age.png",
    path_to_save_amount_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "inheritance_amount_by_age.png",
    path_to_save_amount_unconditional_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "inheritance_amount_by_age_unconditional.png",
    path_to_save_amount_no_inheritance_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "inheritance_amount_by_age_no_inheritance.png",
    path_to_save_amount_positive_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "post_estimation"
    / "inheritance_amount_by_age_positive.png",
):
    """Plot inheritance share and amount by age from baseline simulated data.

    Creates seven plots with 6 lines each (2 education × 3 care types):
    1. Share of people with positive inheritance (gets_inheritance == 1)
       conditional on mother_dead == RECENTLY_DEAD, by age, education, and care type.
    2. Share of people with positive inheritance (gets_inheritance == 1)
       conditional on lagged mother_dead == RECENTLY_DEAD, by age, education, and care type.
    3. Share of individuals with mother recently dead (whole sample), by age and education.
    4. Average inheritance amount (conditional on mother recently dead AND positive inheritance)
       by age, education, and care type.
    5. Average inheritance amount (unconditional, whole sample) by age and education.
    6. Average inheritance amount (conditional on mother recently dead AND gets_inheritance == 0)
       by age, education, and care type.
    7. Average inheritance amount (conditional on bequest_from_parent > 0, whole sample)
       by age, education, and care type.

    Parameters
    ----------
    path_to_specs : Path
        Path to full specs pkl file containing model parameters
    path_to_simulated_data : Path
        Path to baseline simulated data pkl file
    path_to_save_share_plot : Path
        Path to save the share plot (conditional on current period mother recently dead)
    path_to_save_share_lagged_plot : Path
        Path to save the lagged share plot (conditional on previous period mother recently dead)
    path_to_save_mother_dead_share_plot : Path
        Path to save the mother recently dead share plot (whole sample)
    path_to_save_amount_plot : Path
        Path to save the inheritance amount plot (conditional on positive inheritance)
    path_to_save_amount_unconditional_plot : Path
        Path to save the unconditional inheritance amount plot (whole sample)
    path_to_save_amount_no_inheritance_plot : Path
        Path to save the inheritance amount plot (conditional on gets_inheritance == 0)
    path_to_save_amount_positive_plot : Path
        Path to save the inheritance amount plot (conditional on bequest_from_parent > 0)

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

    # Ensure data is sorted by agent and period
    df_sim = df_sim.sort_values(["agent", "period"]).copy()

    # Create lagged choice (previous period's choice) BEFORE filtering
    # This ensures we can access the previous period's choice even if mother wasn't dead then
    df_sim["lagged_choice"] = df_sim.groupby("agent", observed=False)["choice"].shift(1)

    # Create care category based on lagged choice
    no_care_values = NO_CARE.ravel().tolist()
    light_care_values = LIGHT_INFORMAL_CARE.ravel().tolist()
    intensive_care_values = INTENSIVE_INFORMAL_CARE.ravel().tolist()

    df_sim["care_category"] = "no_care"
    df_sim.loc[df_sim["lagged_choice"].isin(light_care_values), "care_category"] = (
        "light_care"
    )
    df_sim.loc[
        df_sim["lagged_choice"].isin(intensive_care_values), "care_category"
    ] = "intensive_care"

    # Create indicator for mother recently dead (whole sample)
    df_sim["mother_recently_dead"] = (
        df_sim["mother_dead"] == PARENT_RECENTLY_DEAD
    ).astype(int)

    # Create lagged mother_dead indicator (previous period's mother_dead)
    df_sim["lagged_mother_dead"] = df_sim.groupby("agent", observed=False)[
        "mother_dead"
    ].shift(1)
    df_sim["lagged_mother_recently_dead"] = (
        df_sim["lagged_mother_dead"] == PARENT_RECENTLY_DEAD
    ).astype(int)

    # Filter to mother recently dead (PARENT_RECENTLY_DEAD = 1) in current period
    df_recently_dead = df_sim.loc[df_sim["mother_dead"] == PARENT_RECENTLY_DEAD].copy()

    # Filter to lagged mother recently dead (PARENT_RECENTLY_DEAD = 1) in previous period
    df_lagged_recently_dead = df_sim.loc[
        df_sim["lagged_mother_recently_dead"] == 1
    ].copy()

    # Check if gets_inheritance column exists
    if "gets_inheritance" not in df_recently_dead.columns:
        raise ValueError(
            f"Missing 'gets_inheritance' column. "
            f"Available columns: {df_recently_dead.columns.tolist()}"
        )

    # Care type order for plots - match style from task_plot_inheritance_by_age.py
    # Order: intensive, light, no care
    care_type_order = [
        ("intensive_care", "Intensive informal care"),
        ("light_care", "Light informal care"),
        ("no_care", "No informal care"),
    ]

    # Colors for education levels
    edu_colors = [plt.cm.tab10(i) for i in range(len(specs["education_labels"]))]

    # Line styles for care types - match task_plot_inheritance_by_age.py
    # Intensive: solid, Light: dotted, No care: dashed
    care_linestyles = ["-", ":", "--"]  # Index 0: Intensive, 1: Light, 2: No care
    care_linewidths = [2.5, 2, 2]  # Thicker for intensive

    # Calculate share with positive inheritance (gets_inheritance == 1) by age, education, and care category
    share_by_group = (
        df_recently_dead.groupby(
            ["age", "education", "care_category"], observed=False
        )["gets_inheritance"]
        .mean()
        .reset_index()
    )

    # Get all unique ages
    ages = np.sort(df_recently_dead["age"].unique())

    # =================================================================================
    # Plot: Share with positive inheritance (gets_inheritance == 1) by age
    # Conditional on mother recently dead, grouped by education and care type
    # =================================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for each care type, then for each education level
    for care_idx, (care_category, care_label) in enumerate(care_type_order):
        linestyle = care_linestyles[care_idx]
        linewidth = care_linewidths[care_idx]

        # Filter to this care category
        df_care = share_by_group.loc[
            share_by_group["care_category"] == care_category
        ].copy()

        # Plot for each education level
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            color = edu_colors[edu_var]

            # Filter to this education level within this care category
            df_edu_care = df_care.loc[df_care["education"] == edu_var].copy()

            if len(df_edu_care) > 0:
                # Reindex to all ages and fill missing with NaN
                shares = (
                    df_edu_care.set_index("age")["gets_inheritance"]
                    .reindex(ages)
                    .values
                )
                mask = ~np.isnan(shares)
                if mask.sum() > 0:
                    ax.plot(
                        ages[mask],
                        shares[mask],
                        linewidth=linewidth,
                        color=color,
                        linestyle=linestyle,
                        label=f"{edu_label}, {care_label}",
                        alpha=0.8,
                    )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel(
        "Share with Positive Inheritance (gets_inheritance == 1)", fontsize=12
    )
    ax.set_title(
        "Inheritance Receipt Probability by Age, Education, and Care Type (Baseline Model)\n"
        "Conditional on Mother Recently Dead",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)

    plt.tight_layout()
    path_to_save_share_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_share_plot, dpi=300)
    plt.close(fig)

    print(f"Inheritance share by age plot saved to {path_to_save_share_plot}")

    # =================================================================================
    # Plot: Share with positive inheritance (gets_inheritance == 1) by age
    # Conditional on lagged mother recently dead, grouped by education and care type
    # =================================================================================
    # Calculate share with positive inheritance by age, education, and care category (lagged condition)
    share_lagged_by_group = (
        df_lagged_recently_dead.groupby(
            ["age", "education", "care_category"], observed=False
        )["gets_inheritance"]
        .mean()
        .reset_index()
    )

    # Get all unique ages for lagged plot
    ages_lagged = np.sort(df_lagged_recently_dead["age"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for each care type, then for each education level
    for care_idx, (care_category, care_label) in enumerate(care_type_order):
        linestyle = care_linestyles[care_idx]
        linewidth = care_linewidths[care_idx]

        # Filter to this care category
        df_care = share_lagged_by_group.loc[
            share_lagged_by_group["care_category"] == care_category
        ].copy()

        # Plot for each education level
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            color = edu_colors[edu_var]

            # Filter to this education level within this care category
            df_edu_care = df_care.loc[df_care["education"] == edu_var].copy()

            if len(df_edu_care) > 0:
                # Reindex to all ages and fill missing with NaN
                shares = (
                    df_edu_care.set_index("age")["gets_inheritance"]
                    .reindex(ages_lagged)
                    .values
                )
                mask = ~np.isnan(shares)
                if mask.sum() > 0:
                    ax.plot(
                        ages_lagged[mask],
                        shares[mask],
                        linewidth=linewidth,
                        color=color,
                        linestyle=linestyle,
                        label=f"{edu_label}, {care_label}",
                        alpha=0.8,
                    )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel(
        "Share with Positive Inheritance (gets_inheritance == 1)", fontsize=12
    )
    ax.set_title(
        "Inheritance Receipt Probability by Age, Education, and Care Type (Baseline Model)\n"
        "Conditional on Lagged Mother Recently Dead",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)

    plt.tight_layout()
    path_to_save_share_lagged_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_share_lagged_plot, dpi=300)
    plt.close(fig)

    print(
        f"Inheritance share by age (lagged) plot saved to {path_to_save_share_lagged_plot}"
    )

    # =================================================================================
    # Plot: Share with mother recently dead (whole sample) by age
    # Grouped by education (no care distinction for this plot)
    # =================================================================================
    # Calculate share with mother recently dead by age and education (whole sample)
    mother_dead_share_by_group = (
        df_sim.groupby(["age", "education"], observed=False)["mother_recently_dead"]
        .mean()
        .reset_index()
    )

    # Get all unique ages for mother dead plot
    ages_mother_dead = np.sort(df_sim["age"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(specs["education_labels"]):
        color = edu_colors[edu_var]

        # Filter to this education level
        df_edu = mother_dead_share_by_group.loc[
            mother_dead_share_by_group["education"] == edu_var
        ].copy()

        if len(df_edu) > 0:
            # Reindex to all ages and fill missing with NaN
            shares = (
                df_edu.set_index("age")["mother_recently_dead"]
                .reindex(ages_mother_dead)
                .values
            )
            mask = ~np.isnan(shares)
            if mask.sum() > 0:
                ax.plot(
                    ages_mother_dead[mask],
                    shares[mask],
                    linewidth=2,
                    color=color,
                    label=edu_label,
                    alpha=0.8,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Share with Mother Recently Dead", fontsize=12)
    ax.set_title(
        "Share with Mother Recently Dead by Age and Education (Baseline Model)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    path_to_save_mother_dead_share_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_mother_dead_share_plot, dpi=300)
    plt.close(fig)

    print(
        f"Mother recently dead share by age plot saved to {path_to_save_mother_dead_share_plot}"
    )

    # =================================================================================
    # Plot: Average inheritance amount (conditional on positive) by age
    # Conditional on mother recently dead AND positive inheritance, grouped by education and care type
    # =================================================================================
    # Check if bequest_from_parent column exists
    if "bequest_from_parent" not in df_recently_dead.columns:
        raise ValueError(
            f"Missing 'bequest_from_parent' column. "
            f"Available columns: {df_recently_dead.columns.tolist()}"
        )

    # Calculate average inheritance amount (conditional on positive) by age, education, and care category
    # First, filter to positive inheritance only
    df_positive = df_recently_dead.loc[df_recently_dead["gets_inheritance"] == 1].copy()

    if len(df_positive) > 0:
        avg_amount_by_group = (
            df_positive.groupby(
                ["age", "education", "care_category"], observed=False
            )["bequest_from_parent"]
            .mean()
            .reset_index()
        )

        # Get all unique ages for amount plot
        ages_amount = np.sort(df_positive["age"].unique())

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot for each care type, then for each education level
        for care_idx, (care_category, care_label) in enumerate(care_type_order):
            linestyle = care_linestyles[care_idx]
            linewidth = care_linewidths[care_idx]

            # Filter to this care category
            df_care = avg_amount_by_group.loc[
                avg_amount_by_group["care_category"] == care_category
            ].copy()

            # Plot for each education level
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                color = edu_colors[edu_var]

                # Filter to this education level within this care category
                df_edu_care = df_care.loc[df_care["education"] == edu_var].copy()

                if len(df_edu_care) > 0:
                    # Reindex to all ages and fill missing with NaN
                    # Convert from wealth units to Euros
                    amounts = (
                        df_edu_care.set_index("age")["bequest_from_parent"]
                        .reindex(ages_amount)
                        .values
                        * specs["wealth_unit"]
                    )
                    mask = ~np.isnan(amounts)
                    if mask.sum() > 0:
                        ax.plot(
                            ages_amount[mask],
                            amounts[mask],
                            linewidth=linewidth,
                            color=color,
                            linestyle=linestyle,
                            label=f"{edu_label}, {care_label}",
                            alpha=0.8,
                        )

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Inheritance Amount (€)", fontsize=12)
        ax.set_title(
            "Average Inheritance Amount by Age, Education, and Care Type (Baseline Model)\n"
            "Conditional on Mother Recently Dead AND Positive Inheritance",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncol=2)
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        plt.tight_layout()
        path_to_save_amount_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save_amount_plot, dpi=300)
        plt.close(fig)

        print(f"Inheritance amount by age plot saved to {path_to_save_amount_plot}")

        # Print summary statistics
        print("\n" + "=" * 70)
        print("INHERITANCE AMOUNT SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total observations with positive inheritance: {len(df_positive)}")
        print(
            f"Average inheritance amount (conditional on positive): €{df_positive['bequest_from_parent'].mean() * specs['wealth_unit']:,.0f}"
        )
        print(
            f"Median inheritance amount (conditional on positive): €{df_positive['bequest_from_parent'].median() * specs['wealth_unit']:,.0f}"
        )
        print("=" * 70 + "\n")
    else:
        print("Warning: No observations with positive inheritance found.")

    # =================================================================================
    # Plot: Average inheritance amount (unconditional) by age
    # Whole sample (not filtered), grouped by education (no care distinction)
    # =================================================================================
    # Check if bequest_from_parent column exists in whole sample
    if "bequest_from_parent" not in df_sim.columns:
        raise ValueError(
            f"Missing 'bequest_from_parent' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Calculate average inheritance amount (unconditional) by age and education
    # This includes zeros (where no inheritance was received)
    avg_amount_unconditional_by_group = (
        df_sim.groupby(["age", "education"], observed=False)["bequest_from_parent"]
        .mean()
        .reset_index()
    )

    # Get all unique ages for unconditional amount plot
    ages_unconditional = np.sort(df_sim["age"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for edu_var, edu_label in enumerate(specs["education_labels"]):
        color = edu_colors[edu_var]

        # Filter to this education level
        df_edu = avg_amount_unconditional_by_group.loc[
            avg_amount_unconditional_by_group["education"] == edu_var
        ].copy()

        if len(df_edu) > 0:
            # Reindex to all ages and fill missing with NaN
            # Convert from wealth units to Euros
            amounts = (
                df_edu.set_index("age")["bequest_from_parent"]
                .reindex(ages_unconditional)
                .values
                * specs["wealth_unit"]
            )
            mask = ~np.isnan(amounts)
            if mask.sum() > 0:
                ax.plot(
                    ages_unconditional[mask],
                    amounts[mask],
                    linewidth=2,
                    color=color,
                    label=edu_label,
                    alpha=0.8,
                )

    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Average Inheritance Amount (€)", fontsize=12)
    ax.set_title(
        "Average Inheritance Amount by Age and Education (Baseline Model)\n"
        "Unconditional (Whole Sample)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()
    path_to_save_amount_unconditional_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save_amount_unconditional_plot, dpi=300)
    plt.close(fig)

    print(
        f"Inheritance amount (unconditional) by age plot saved to {path_to_save_amount_unconditional_plot}"
    )

    # =================================================================================
    # Plot: Average inheritance amount (conditional on gets_inheritance == 0) by age
    # Conditional on mother recently dead AND gets_inheritance == 0, grouped by education and care type
    # =================================================================================
    # Filter to gets_inheritance == 0 within mother recently dead sample
    df_no_inheritance = df_recently_dead.loc[
        df_recently_dead["gets_inheritance"] == 0
    ].copy()

    if len(df_no_inheritance) > 0:
        # Calculate average inheritance amount by age, education, and care category
        avg_amount_no_inheritance_by_group = (
            df_no_inheritance.groupby(
                ["age", "education", "care_category"], observed=False
            )["bequest_from_parent"]
            .mean()
            .reset_index()
        )

        # Get all unique ages for no inheritance amount plot
        ages_no_inheritance = np.sort(df_no_inheritance["age"].unique())

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot for each care type, then for each education level
        for care_idx, (care_category, care_label) in enumerate(care_type_order):
            linestyle = care_linestyles[care_idx]
            linewidth = care_linewidths[care_idx]

            # Filter to this care category
            df_care = avg_amount_no_inheritance_by_group.loc[
                avg_amount_no_inheritance_by_group["care_category"] == care_category
            ].copy()

            # Plot for each education level
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                color = edu_colors[edu_var]

                # Filter to this education level within this care category
                df_edu_care = df_care.loc[df_care["education"] == edu_var].copy()

                if len(df_edu_care) > 0:
                    # Reindex to all ages and fill missing with NaN
                    # Convert from wealth units to Euros
                    amounts = (
                        df_edu_care.set_index("age")["bequest_from_parent"]
                        .reindex(ages_no_inheritance)
                        .values
                        * specs["wealth_unit"]
                    )
                    mask = ~np.isnan(amounts)
                    if mask.sum() > 0:
                        ax.plot(
                            ages_no_inheritance[mask],
                            amounts[mask],
                            linewidth=linewidth,
                            color=color,
                            linestyle=linestyle,
                            label=f"{edu_label}, {care_label}",
                            alpha=0.8,
                        )

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Average Inheritance Amount (€)", fontsize=12)
        ax.set_title(
            "Average Inheritance Amount by Age, Education, and Care Type (Baseline Model)\n"
            "Conditional on Mother Recently Dead AND gets_inheritance == 0",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncol=2)
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        plt.tight_layout()
        path_to_save_amount_no_inheritance_plot.parent.mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(path_to_save_amount_no_inheritance_plot, dpi=300)
        plt.close(fig)

        print(
            f"Inheritance amount (no inheritance) by age plot saved to {path_to_save_amount_no_inheritance_plot}"
        )
    else:
        print("Warning: No observations with gets_inheritance == 0 found.")

    # =================================================================================
    # Plot: Average inheritance amount (conditional on bequest_from_parent > 0) by age
    # Whole sample, filtered to positive bequest_from_parent, grouped by education and care type
    # =================================================================================
    # Check if bequest_from_parent column exists in whole sample
    if "bequest_from_parent" not in df_sim.columns:
        raise ValueError(
            f"Missing 'bequest_from_parent' column. "
            f"Available columns: {df_sim.columns.tolist()}"
        )

    # Filter to positive bequest_from_parent (which already incorporates both conditions
    # from the budget equation: mother_dead == PARENT_RECENTLY_DEAD AND gets_inheritance == 1)
    df_positive_bequest = df_sim.loc[df_sim["bequest_from_parent"] > 0].copy()

    if len(df_positive_bequest) > 0:
        # Calculate average inheritance amount by age, education, and care category
        avg_amount_positive_by_group = (
            df_positive_bequest.groupby(
                ["age", "education", "care_category"], observed=False
            )["bequest_from_parent"]
            .mean()
            .reset_index()
        )

        # Get all unique ages for positive bequest plot
        ages_positive = np.sort(df_positive_bequest["age"].unique())

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot for each care type, then for each education level
        for care_idx, (care_category, care_label) in enumerate(care_type_order):
            linestyle = care_linestyles[care_idx]
            linewidth = care_linewidths[care_idx]

            # Filter to this care category
            df_care = avg_amount_positive_by_group.loc[
                avg_amount_positive_by_group["care_category"] == care_category
            ].copy()

            # Plot for each education level
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                color = edu_colors[edu_var]

                # Filter to this education level within this care category
                df_edu_care = df_care.loc[df_care["education"] == edu_var].copy()

                if len(df_edu_care) > 0:
                    # Reindex to all ages and fill missing with NaN
                    # Convert from wealth units to Euros
                    amounts = (
                        df_edu_care.set_index("age")["bequest_from_parent"]
                        .reindex(ages_positive)
                        .values
                        * specs["wealth_unit"]
                    )
                    mask = ~np.isnan(amounts)
                    if mask.sum() > 0:
                        ax.plot(
                            ages_positive[mask],
                            amounts[mask],
                            linewidth=linewidth,
                            color=color,
                            linestyle=linestyle,
                            label=f"{edu_label}, {care_label}",
                            alpha=0.8,
                        )

        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Average Inheritance Amount (€)", fontsize=12)
        ax.set_title(
            "Average Inheritance Amount by Age, Education, and Care Type (Baseline Model)\n"
            "Conditional on bequest_from_parent > 0 (Whole Sample)",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncol=2)
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        plt.tight_layout()
        path_to_save_amount_positive_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save_amount_positive_plot, dpi=300)
        plt.close(fig)

        print(
            f"Inheritance amount (positive bequest) by age plot saved to {path_to_save_amount_positive_plot}"
        )
    else:
        print("Warning: No observations with bequest_from_parent > 0 found.")

