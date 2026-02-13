"""Helper functions for plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    FULL_TIME_CHOICES,
    INFORMAL_CARE,
    PART_TIME,
    PART_TIME_CHOICES,
    RETIREMENT,
    RETIREMENT_CHOICES,
    SEX,
    UNEMPLOYED,
    UNEMPLOYED_CHOICES,
    WORK_CHOICES,
)


def plot_choice_shares_by_education_bw(  # noqa: PLR0912, PLR0915
    data_emp,
    data_sim,
    specs,
    age_min=None,
    age_max=None,
    choice_groups_sim=None,
    path_to_save_plot=None,
    standard_deviation=False,
):
    """Plot choice-specific shares by age and education in black and white.

    Observed lines are dashed, simulated lines are black solid.
    Only plots over the age range [start_age, end_age_msm].
    If standard_deviation is True, plot bands of 1.96 * standard
    deviation around the empirical mean per age. SD is computed from the raw
    observations at each age (general formula for any mean, not proportion-specific).
    """

    # ---------- 1. Map raw codes → 4-way choice ----------------------------
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }
    if choice_groups_sim is None:
        choice_groups_sim = {
            0: RETIREMENT,
            1: UNEMPLOYED,
            2: PART_TIME,
            3: FULL_TIME,
        }

    data_sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    data_emp = data_emp.copy()

    for agg_code, raw_codes in choice_groups_sim.items():
        data_sim.loc[
            data_sim["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    for agg_code, raw_codes in choice_groups_emp.items():
        data_emp.loc[
            data_emp["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    data_sim["choice_group"] = data_sim["choice_group"].fillna(0).astype(int)
    data_emp["choice_group"] = data_emp["choice_group"].astype(int)

    # ---------- 2. Plotting setup ------------------------------------------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    sex = SEX  # assumed scalar {0,1}

    n_edu = len(specs["education_labels"])
    n_choices = len(specs["choice_labels"])
    # Layout: 4 rows (choices) x 2 columns (education)
    # Left column: low education, Right column: high education
    fig, axs = plt.subplots(
        n_choices, n_edu, figsize=(16, 20), sharex=True, sharey=True
    )
    axs = np.atleast_2d(axs)

    # Add padding between subplots
    plt.subplots_adjust(
        left=0.12, right=0.92, top=0.96, bottom=0.08, wspace=0.25, hspace=0.4
    )

    # ---------- 3. Pre-compute shares for all education groups --------------
    shares_by_edu = {}
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        emp_edu = data_emp[
            (data_emp["sex"] == sex) & (data_emp["education"] == edu_var)
        ]
        sim_edu = data_sim[data_sim["education"] == edu_var]

        # shares by age × aggregated choice
        sim_shares = (
            sim_edu.groupby("age", observed=False)["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        emp_shares = (
            emp_edu.groupby("age", observed=False)["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        shares_by_edu[edu_var] = {
            "sim": sim_shares,
            "emp": emp_shares,
            "emp_edu": emp_edu,
            "label": edu_label,
        }

    # ---------- 4. Loop over choices (rows) and education (columns) ---------
    for choice_var in range(n_choices):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axs[choice_var, edu_var]

            ages = range(age_min, age_max + 1)
            sim_shares = shares_by_edu[edu_var]["sim"]
            emp_shares = shares_by_edu[edu_var]["emp"]
            vals_sim = sim_shares.reindex(ages, fill_value=0)[choice_var]
            vals_emp = emp_shares.reindex(ages, fill_value=0)[choice_var]

            # Optional: band of 1.96 * standard deviation around empirical mean.
            # Use general formula: std from raw observations at each age (any variable).
            if standard_deviation:
                emp_edu = shares_by_edu[edu_var]["emp_edu"]
                std_by_age = []
                for age in ages:
                    at_age = emp_edu.loc[emp_edu["age"] == age, "choice_group"]
                    vals = (at_age == choice_var).astype(float)
                    std_by_age.append(
                        float(np.std(vals, ddof=0)) if len(vals) > 0 else 0.0
                    )
                std_emp = pd.Series(std_by_age, index=ages)
                half_width = 1.96 * std_emp
                lower = (vals_emp - half_width).clip(0, 1)
                upper = (vals_emp + half_width).clip(0, 1)
                ax.fill_between(
                    ages,
                    lower,
                    upper,
                    color="black",
                    alpha=0.2,
                )

            # Black and white: simulated = black solid, observed = dashed
            # Solid line thicker, dashed line thinner
            ax.plot(ages, vals_sim, color="black", label="Simulated", linewidth=1.8)
            ax.plot(
                ages, vals_emp, color="black", ls="--", label="Observed", linewidth=1.4
            )

            # Add padding within each plot (space between content and axes)
            # Calculate padding as percentage of range
            y_pad = 0.05  # 5% padding on top and bottom
            x_pad = (age_max - age_min) * 0.02  # 2% padding on left and right

            # Set limits with padding
            ax.set_ylim(0 - y_pad, 1 + y_pad)
            ax.set_xlim(age_min - x_pad, age_max + x_pad)

            # All subplots get full axes with labels (slightly smaller fonts)
            ax.set_title(edu_label, fontsize=16)  # Normal font, not bold

            # Get choice label and rename if needed
            choice_label = specs["choice_labels"][choice_var]
            if choice_label == "Retired":
                choice_label = "Retirement"
            elif choice_label == "Unemployed":
                choice_label = "Unemployment"

            ax.set_ylabel(f"{choice_label}\nShare", fontsize=14)
            ax.set_xlabel("Age", fontsize=14)

            # Tick labels with ticks and numbers on all axes (slightly smaller)
            ax.tick_params(
                labelsize=12,
                width=1.5,
                length=6,
                bottom=True,
                top=False,
                left=True,
                right=False,
                labelbottom=True,
                labeltop=False,
                labelleft=True,
                labelright=False,
            )

            # Remove top and right spines (box), keep only bottom and left axes
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add faint horizontal grid lines from y-axis ticks
            ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

            # Add legend on first subplot only (slightly smaller font)
            if choice_var == edu_var == 0:
                ax.legend(prop={"size": 12}, frameon=True)

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=600, transparent=False, bbox_inches="tight")
    plt.close(fig)


def plot_aggregated_share_by_education_bw(  # noqa: PLR0915
    data_emp,
    data_sim,
    specs,
    choice_group_codes,
    ylabel="Share",
    age_min=None,
    age_max=None,
    choice_groups_sim=None,
    path_to_save_plot=None,
    standard_deviation=False,
):
    """Plot one aggregated share (sum of given choice_group codes) in 1×2 layout.

    Left: low education, Right: high education. Observed dashed, simulated solid.
    choice_group_codes: e.g. [2, 3] for employment (PT+FT), [0, 1] for non-work.
    """
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }
    if choice_groups_sim is None:
        choice_groups_sim = {
            0: RETIREMENT,
            1: UNEMPLOYED,
            2: PART_TIME,
            3: FULL_TIME,
        }

    data_sim = data_sim.loc[data_sim["health"] != DEAD].copy()
    data_emp = data_emp.copy()

    for agg_code, raw_codes in choice_groups_sim.items():
        data_sim.loc[
            data_sim["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    for agg_code, raw_codes in choice_groups_emp.items():
        data_emp.loc[
            data_emp["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"
        ] = agg_code

    data_sim["choice_group"] = data_sim["choice_group"].fillna(0).astype(int)
    data_emp["choice_group"] = data_emp["choice_group"].astype(int)

    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    sex = SEX
    n_edu = len(specs["education_labels"])

    fig, axs = plt.subplots(1, n_edu, figsize=(10, 5), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    plt.subplots_adjust(left=0.12, right=0.92, top=0.88, bottom=0.14, wspace=0.25)

    shares_by_edu = {}
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        emp_edu = data_emp[
            (data_emp["sex"] == sex) & (data_emp["education"] == edu_var)
        ]
        sim_edu = data_sim[data_sim["education"] == edu_var]

        sim_shares = (
            sim_edu.groupby("age", observed=False)["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        emp_shares = (
            emp_edu.groupby("age", observed=False)["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        shares_by_edu[edu_var] = {
            "sim": sim_shares,
            "emp": emp_shares,
            "emp_edu": emp_edu,
            "label": edu_label,
        }

    ages = range(age_min, age_max + 1)
    y_pad = 0.05
    x_pad = (age_max - age_min) * 0.02

    for edu_var, edu_label in enumerate(specs["education_labels"]):
        ax = axs[edu_var]
        sim_shares = shares_by_edu[edu_var]["sim"]
        emp_shares = shares_by_edu[edu_var]["emp"]

        sim_reindexed = sim_shares.reindex(ages, fill_value=0).reindex(
            columns=choice_group_codes, fill_value=0
        )
        emp_reindexed = emp_shares.reindex(ages, fill_value=0).reindex(
            columns=choice_group_codes, fill_value=0
        )
        vals_sim = sim_reindexed.sum(axis=1)
        vals_emp = emp_reindexed.sum(axis=1)

        if standard_deviation:
            emp_edu = shares_by_edu[edu_var]["emp_edu"]
            std_by_age = []
            for age in ages:
                at_age = emp_edu.loc[emp_edu["age"] == age, "choice_group"]
                vals = at_age.isin(choice_group_codes).astype(float)
                std_by_age.append(float(np.std(vals, ddof=0)) if len(vals) > 0 else 0.0)
            std_emp = pd.Series(std_by_age, index=ages)
            half_width = 1.96 * std_emp
            lower = (vals_emp - half_width).clip(0, 1)
            upper = (vals_emp + half_width).clip(0, 1)
            ax.fill_between(ages, lower, upper, color="black", alpha=0.2)

        ax.plot(ages, vals_sim, color="black", label="Simulated", linewidth=1.8)
        ax.plot(ages, vals_emp, color="black", ls="--", label="Observed", linewidth=1.4)
        ax.set_ylim(0 - y_pad, 1 + y_pad)
        ax.set_xlim(age_min - x_pad, age_max + x_pad)
        ax.set_title(edu_label, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel("Age", fontsize=14)
        ax.tick_params(
            labelsize=12,
            width=1.5,
            length=6,
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
            labeltop=False,
            labelleft=True,
            labelright=False,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
        if edu_var == 0:
            ax.legend(prop={"size": 12}, frameon=True)

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=600, transparent=False, bbox_inches="tight")
    plt.close(fig)
