"""Plot care demand patterns from simulated data."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import DEAD


def task_plot_care_demand_by_age(
    path_to_simulated_data: Path = BLD / "solve_and_simulate"
    # / "simulated_data.pkl",
    / "simulated_data_estimated_params.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "counterfactual"
    / "care_demand_by_age.png",
    has_sister: bool = True,
    education: bool = True,
) -> None:
    """Plot simulated care demand share by age."""

    # Load simulated data
    df_sim = pd.read_pickle(path_to_simulated_data)

    # Create plot
    create_care_demand_plot(df_sim, path_to_plot, has_sister=False, education=True)


def create_care_demand_plot(  # noqa: PLR0912, PLR0915
    df_sim: pd.DataFrame,
    path_to_plot: Path,
    has_sister: bool = True,
    education: bool = True,
) -> None:
    """Create plot showing care demand share by age, education, and sister status."""

    # Age range from specs: start_age_msm to end_age_caregiving
    age_min = 40  # start_age_msm
    age_max = 80  # end_age_caregiving

    # Filter out dead individuals and restrict to relevant age range
    df_sim = df_sim.loc[df_sim["health"] != DEAD].copy()
    df_sim = df_sim.loc[(df_sim["age"] >= age_min) & (df_sim["age"] <= age_max)].copy()

    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors and styles
    colors = {0: "blue", 1: "orange"}  # 0 = low education, 1 = high education
    linestyles = {0: "--", 1: "-"}  # 0 = no sister, 1 = has sister
    edu_labels = {0: "Low Education", 1: "High Education"}
    sister_labels = {0: "No Sister", 1: "Has Sister"}

    # Determine grouping variables based on arguments
    group_vars = ["age"]
    if education:
        group_vars.append("education")
    if has_sister:
        group_vars.append("has_sister")

    # Calculate care demand share by selected variables
    care_demand_shares = df_sim.groupby(group_vars)["care_demand"].mean().reset_index()

    # Generate title based on arguments
    title_parts = ["Simulated Care Demand Share by Age"]
    if education and has_sister:
        title_parts.append("Education, and Sister Status")
    elif education:
        title_parts.append("and Education")
    elif has_sister:
        title_parts.append("and Sister Status")
    ax.set_title(", ".join(title_parts))

    # Plot based on arguments
    if education and has_sister:
        # Plot all four combinations
        for edu in (0, 1):
            for sister in (0, 1):
                subset = care_demand_shares[
                    (care_demand_shares["education"] == edu)
                    & (care_demand_shares["has_sister"] == sister)
                ]
                if not subset.empty:
                    ax.plot(
                        subset["age"],
                        subset["care_demand"],
                        color=colors[edu],
                        linestyle=linestyles[sister],
                        linewidth=2,
                        label=f"{edu_labels[edu]}, {sister_labels[sister]}",
                    )
    elif education:
        # Plot by education only
        for edu in (0, 1):
            subset = care_demand_shares[care_demand_shares["education"] == edu]
            if not subset.empty:
                ax.plot(
                    subset["age"],
                    subset["care_demand"],
                    color=colors[edu],
                    linewidth=2,
                    label=edu_labels[edu],
                )
    elif has_sister:
        # Plot by sister status only
        for sister in (0, 1):
            subset = care_demand_shares[care_demand_shares["has_sister"] == sister]
            if not subset.empty:
                ax.plot(
                    subset["age"],
                    subset["care_demand"],
                    linestyle=linestyles[sister],
                    linewidth=2,
                    label=sister_labels[sister],
                )
    else:
        # Plot overall average
        overall_shares = df_sim.groupby("age")["care_demand"].mean().reset_index()
        ax.plot(
            overall_shares["age"],
            overall_shares["care_demand"],
            linewidth=2,
            label="Overall",
        )

    # Styling
    ax.set_xlabel("Age")
    ax.set_ylabel("Share with Care Demand")
    ax.grid(True, alpha=0.2)
    ax.legend()

    # Calculate padding (same amount as bottom padding)
    x_pad = 0.005  # Same as bottom padding

    # Set axis limits with padding
    ax.set_xlim(age_min - x_pad, age_max + x_pad)
    # Add small padding at bottom so zero is not exactly on x-axis
    ax.set_ylim(-0.005, None)

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300, transparent=False, bbox_inches="tight")
    plt.close()

    print(f"Care demand plot saved to: {path_to_plot}")
