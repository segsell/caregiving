#!/usr/bin/env python
"""Draft script to test different black and white plotting styles.

This script generates different style variations for the employment rate plot
by age bins at first care demand, suitable for publication in top economic journals.

To run this script, ensure you have activated your conda environment and run:
    python src/caregiving/counterfactual/draft_plot_styles.py

The plots will be saved to: bld/plots/counterfactual/plot_style_draft/
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.config import BLD
from caregiving.counterfactual.plotting_helpers import (
    get_age_at_first_event,
)
from caregiving.counterfactual.plotting_utils import (
    calculate_outcomes,
    calculate_working_hours_weekly,
    create_outcome_columns,
    merge_and_compute_differences,
    prepare_dataframes_for_comparison,
)
from caregiving.counterfactual.task_plot_labor_supply_differences_no_care_demand import (  # noqa: E501
    _add_distance_to_first_care_demand,
)

# ============================================================================
# Data loading (replicating the actual plotting function)
# ============================================================================


def load_plot_data():
    """Load and prepare data for the employment rate plot."""
    path_to_original_data = (
        BLD / "solve_and_simulate" / "simulated_data_estimated_params.pkl"
    )
    path_to_no_care_demand_data = (
        BLD / "solve_and_simulate" / "simulated_data_no_care_demand.pkl"
    )
    path_to_options = BLD / "model" / "options.pkl"

    ever_caregivers = True
    window = 20
    min_age = 50
    max_age = 62
    bin_width = 3

    # Load and prepare data
    df_o, df_c = prepare_dataframes_for_comparison(
        pd.read_pickle(path_to_original_data),
        pd.read_pickle(path_to_no_care_demand_data),
        ever_caregivers=ever_caregivers,
    )

    # Calculate outcomes
    o_outcomes = calculate_outcomes(df_o, choice_set_type="original")
    c_outcomes = calculate_outcomes(df_c, choice_set_type="no_care_demand")

    # Calculate working hours
    options = pickle.load(path_to_options.open("rb"))
    model_params = options["model_params"]
    o_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_o, model_params, choice_set_type="original"
    )
    c_outcomes["hours_weekly"] = calculate_working_hours_weekly(
        df_c, model_params, choice_set_type="no_care_demand"
    )

    # Create outcome columns and merge
    o_cols = create_outcome_columns(df_o, o_outcomes, "_o")
    c_cols = create_outcome_columns(df_c, c_outcomes, "_c")

    # Merge and compute differences
    outcome_names = ["work", "ft", "pt", "job_offer", "hours_weekly", "care"]
    merged = merge_and_compute_differences(o_cols, c_cols, outcome_names)

    # Compute distance and age at first care demand from original
    df_o_dist = _add_distance_to_first_care_demand(df_o)

    # Get first care demand period for each agent
    dist_map = (
        df_o_dist.groupby("agent", observed=False)["first_care_demand_period"]
        .first()
        .reset_index()
    )

    # Get age at first care demand period
    care_demand_mask = df_o["care_demand"] > 0
    first_care_demand_with_age = get_age_at_first_event(
        df_o, care_demand_mask, "age_at_first_care_demand"
    )

    # Merge distance and age information
    merged = merged.merge(dist_map, on="agent", how="left")
    merged["distance_to_first_care_demand"] = (
        merged["period"] - merged["first_care_demand_period"]
    )
    merged = merged.merge(first_care_demand_with_age, on="agent", how="left")

    # Filter to age range
    merged = merged[
        (merged["age_at_first_care_demand"] >= min_age)
        & (merged["age_at_first_care_demand"] <= max_age)
    ]

    # Create age bins
    merged["age_bin_start"] = (
        (merged["age_at_first_care_demand"] // bin_width) * bin_width
    ).astype(int)
    merged["age_bin_end"] = merged["age_bin_start"] + bin_width - 1
    merged["age_bin_label"] = (
        merged["age_bin_start"].astype(str) + "-" + merged["age_bin_end"].astype(str)
    )

    # Trim to window
    merged = merged[
        (merged["distance_to_first_care_demand"] >= -window)
        & (merged["distance_to_first_care_demand"] <= window)
    ]

    # Average differences by distance and age_bin
    prof = (
        merged.groupby(
            ["distance_to_first_care_demand", "age_bin_label"], observed=False
        )[["diff_work"]]
        .mean()
        .reset_index()
    )

    # Add age_bin_start back for sorting
    prof["age_bin_start"] = prof["age_bin_label"].str.split("-").str[0].astype(int)
    prof = prof.sort_values(["age_bin_start", "distance_to_first_care_demand"])

    # Get unique age bins in order (sorted by bin start)
    unique_bins = (
        merged[["age_bin_label", "age_bin_start"]]
        .drop_duplicates()
        .sort_values("age_bin_start")["age_bin_label"]
        .tolist()
    )

    return prof, unique_bins, window


# ============================================================================
# Style variations
# ============================================================================


def style_a_dashed_lines(prof, unique_bins, window, output_path):
    """Style A: Solid and different types of dashed lines."""
    # Define line styles for different age bins
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]
    # All black - no color variation
    line_color = "#000000"

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bin_label in enumerate(unique_bins):
        prof_group = prof[prof["age_bin_label"] == bin_label].sort_values(
            "distance_to_first_care_demand"
        )
        ax.plot(
            prof_group["distance_to_first_care_demand"],
            prof_group["diff_work"],
            label=f"Age {bin_label}",
            color=line_color,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,  # Using Style E's line width
        )

    ax.axvline(x=0, color="k", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Year relative to first care demand", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Proportion Working\nDeviation from Counterfactual",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-window, window)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.legend(
        loc="best",
        prop={"size": 11},
        title="Age at first care demand",
        title_fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def style_b_symbols(prof, unique_bins, window, output_path):
    """Style B: Solid lines with symbols at fixed intervals."""
    # Different marker styles
    markers = ["o", "s", "^", "D", "v", "p"]
    # All black - no color variation
    line_color = "#000000"
    # Marker interval (every N points)
    marker_interval = 3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bin_label in enumerate(unique_bins):
        prof_group = prof[prof["age_bin_label"] == bin_label].sort_values(
            "distance_to_first_care_demand"
        )
        x_vals = prof_group["distance_to_first_care_demand"].values
        y_vals = prof_group["diff_work"].values

        # Plot line
        ax.plot(
            x_vals,
            y_vals,
            label=f"Age {bin_label}",
            color=line_color,
            linestyle="-",
            linewidth=2,  # Using Style E's line width
        )

        # Add markers at fixed intervals - solid black filled
        marker_indices = np.arange(0, len(x_vals), marker_interval)
        ax.plot(
            x_vals[marker_indices],
            y_vals[marker_indices],
            marker=markers[i % len(markers)],
            color=line_color,
            linestyle="None",
            markersize=7,
            markeredgewidth=1.5,
            markeredgecolor=line_color,
            markerfacecolor=line_color,  # Solid black filled
        )

    ax.axvline(x=0, color="k", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Year relative to first care demand", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Proportion Working\nDeviation from Counterfactual",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-window, window)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.legend(
        loc="best",
        prop={"size": 11},
        title="Age at first care demand",
        title_fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def style_c_linewidth_variation(prof, unique_bins, window, output_path):
    """Style C: Different line styles (using Style E's line width)."""
    # Different line styles
    line_styles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
    # All black - no color variation
    line_color = "#000000"

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bin_label in enumerate(unique_bins):
        prof_group = prof[prof["age_bin_label"] == bin_label].sort_values(
            "distance_to_first_care_demand"
        )
        ax.plot(
            prof_group["distance_to_first_care_demand"],
            prof_group["diff_work"],
            label=f"Age {bin_label}",
            color=line_color,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,  # Using Style E's line width
        )

    ax.axvline(x=0, color="k", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Year relative to first care demand", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Proportion Working\nDeviation from Counterfactual",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-window, window)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.legend(
        loc="best",
        prop={"size": 11},
        title="Age at first care demand",
        title_fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def style_d_combined(prof, unique_bins, window, output_path):
    """Style D: Combined approach - dashed lines with occasional markers."""
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]
    markers = ["o", "s", "^", "D", "v", "p"]
    # All black - no color variation
    line_color = "#000000"
    marker_interval = 4

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bin_label in enumerate(unique_bins):
        prof_group = prof[prof["age_bin_label"] == bin_label].sort_values(
            "distance_to_first_care_demand"
        )
        x_vals = prof_group["distance_to_first_care_demand"].values
        y_vals = prof_group["diff_work"].values

        # Plot dashed line
        ax.plot(
            x_vals,
            y_vals,
            label=f"Age {bin_label}",
            color=line_color,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,  # Using Style E's line width
        )

        # Add markers - solid black filled
        marker_indices = np.arange(0, len(x_vals), marker_interval)
        ax.plot(
            x_vals[marker_indices],
            y_vals[marker_indices],
            marker=markers[i % len(markers)],
            color=line_color,
            linestyle="None",
            markersize=6,
            markeredgewidth=1.2,
            markeredgecolor=line_color,
            markerfacecolor=line_color,  # Solid black filled
        )

    ax.axvline(x=0, color="k", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Year relative to first care demand", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Proportion Working\nDeviation from Counterfactual",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-window, window)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.legend(
        loc="best",
        prop={"size": 11},
        title="Age at first care demand",
        title_fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def style_e_minimalist(prof, unique_bins, window, output_path):
    """Style E: Minimalist with subtle variations."""
    # Very subtle line style variations
    line_styles = ["-", "--", "-.", ":", (0, (8, 4)), (0, (4, 2, 1, 2))]
    # All black - no color variation
    line_color = "#000000"

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bin_label in enumerate(unique_bins):
        prof_group = prof[prof["age_bin_label"] == bin_label].sort_values(
            "distance_to_first_care_demand"
        )
        ax.plot(
            prof_group["distance_to_first_care_demand"],
            prof_group["diff_work"],
            label=f"Age {bin_label}",
            color=line_color,
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,
        )

    ax.axvline(x=0, color="k", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.2, linewidth=0.6)
    ax.set_xlabel("Year relative to first care demand", fontsize=13)
    ax.set_ylabel(
        "Proportion Working\nDeviation from Counterfactual",
        fontsize=13,
    )
    ax.set_xlim(-window, window)
    ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.4)
    ax.tick_params(labelsize=11)
    ax.legend(
        loc="best",
        prop={"size": 10},
        title="Age at first care demand",
        title_fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor="gray",
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def style_f_markers_at_points(prof, unique_bins, window, output_path):
    """Style F: Black lines with markers at every data point (like reference plot)."""
    # Different marker styles matching the reference plot style
    markers = ["s", "D", "o", "^", "x"]  # square, diamond, circle, triangle, cross
    # All black - no color variation
    line_color = "#000000"

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bin_label in enumerate(unique_bins):
        prof_group = prof[prof["age_bin_label"] == bin_label].sort_values(
            "distance_to_first_care_demand"
        )
        x_vals = prof_group["distance_to_first_care_demand"].values
        y_vals = prof_group["diff_work"].values

        # Determine if marker should be hollow (all except 'x')
        current_marker = markers[i % len(markers)]
        is_hollow = current_marker != "x"
        marker_face_color = "white" if is_hollow else line_color

        # Plot line with markers at every point
        ax.plot(
            x_vals,
            y_vals,
            label=f"Age {bin_label}",
            color=line_color,
            linestyle="-",
            linewidth=1,  # Linewidth of 1
            marker=current_marker,
            markersize=6,
            markeredgewidth=1.5,
            markeredgecolor=line_color,
            markerfacecolor=marker_face_color,  # Hollow for most, solid for 'x'
        )

    ax.axvline(x=0, color="k", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Year relative to first care demand", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Proportion Working\nDeviation from Counterfactual",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-window, window)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=12)
    ax.legend(
        loc="best",
        prop={"size": 11},
        title="Age at first care demand",
        title_fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================


def main():
    """Generate all style variations."""
    print("Loading data...")
    prof, unique_bins, window = load_plot_data()
    print(f"Loaded data with {len(unique_bins)} age bins: {unique_bins}")

    # Create output directory
    output_dir = BLD / "plots" / "counterfactual" / "plot_style_draft"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate all style variations
    print("\nGenerating style variations...")

    print("  Style A: Dashed lines...")
    style_a_dashed_lines(
        prof,
        unique_bins,
        window,
        output_dir / "style_a_dashed_lines.png",
    )

    print("  Style B: Symbols at intervals...")
    style_b_symbols(
        prof,
        unique_bins,
        window,
        output_dir / "style_b_symbols.png",
    )

    print("  Style C: Different line styles...")
    style_c_linewidth_variation(
        prof,
        unique_bins,
        window,
        output_dir / "style_c_line_styles.png",
    )

    print("  Style D: Combined (dashed + markers)...")
    style_d_combined(
        prof,
        unique_bins,
        window,
        output_dir / "style_d_combined.png",
    )

    print("  Style E: Minimalist...")
    style_e_minimalist(
        prof,
        unique_bins,
        window,
        output_dir / "style_e_minimalist.png",
    )

    print("  Style F: Markers at every point (reference style)...")
    style_f_markers_at_points(
        prof,
        unique_bins,
        window,
        output_dir / "style_f_markers_at_points.png",
    )

    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
