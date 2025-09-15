"""Plot raw SOEP data."""

from itertools import product
from pathlib import Path
from typing import Annotated, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import (
    BAD_HEALTH,
    DEAD,
    FILL_VALUE,
    FULL_TIME_CHOICES,
    GOOD_HEALTH,
    NOT_WORKING,
    PARENT_WEIGHTS_SHARE,
    PART_TIME,
    PART_TIME_CHOICES,
    RETIREMENT,
    RETIREMENT_CHOICES,
    SEX,
    UNEMPLOYED,
    UNEMPLOYED_CHOICES,
    WORK,
)
from caregiving.specs.task_write_specs import read_and_derive_specs
from caregiving.utils import table

DEGREES_OF_FREEDOM = 1


def task_plot_empirical_soep_moments(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_labor_supply_all_by_age: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_by_age.png",
    path_to_save_labor_supply_non_caregivers_by_age: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_non_caregivers_by_age.png",
    path_to_save_labor_supply_good_health_by_age: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_good_health_by_age.png",
    path_to_save_labor_supply_bad_health_by_age: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_bad_health_by_age.png",
    path_to_save_labor_supply_caregivers_by_age: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_caregivers_by_age.png",
    path_to_save_labor_supply_all_by_age_bins: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_by_age_bins.png",
    path_to_save_labor_supply_caregivers_by_age_bins: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_caregivers_by_age_bins.png",
    path_to_save_labor_supply_light_caregivers_by_age_bins: Annotated[
        Path, Product
    ] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_light_caregivers_by_age_bins.png",
    path_to_save_labor_supply_intensive_caregivers_by_age_bins: Annotated[
        Path, Product
    ] = BLD
    / "plots"
    / "raw_moments"
    / "labor_shares_intensive_caregivers_by_age_bins.png",
) -> None:
    """Create moments for MSM estimation."""

    specs = read_and_derive_specs(path_to_specs)
    start_age = specs["start_age"]
    end_age = specs["end_age_msm"]

    df = pd.read_csv(path_to_main_sample, index_col=[0])
    df = df[(df["sex"] == 1) & (df["age"] <= end_age + 10)].copy()  # women only

    df_non_caregivers = df[df["any_care"] == 0].copy()

    df_good_health = df[df["health"] == GOOD_HEALTH].copy()
    df_bad_health = df[df["health"] == BAD_HEALTH].copy()

    df_caregivers = pd.read_csv(path_to_caregivers_sample, index_col=[0])
    df_caregivers = df_caregivers[
        (df_caregivers["sex"] == 1)
        & (df_caregivers["age"] <= end_age + 10)
        & (df_caregivers["any_care"] == 1)
    ]
    df_light_caregivers = df_caregivers[df_caregivers["light_care"] == 1]
    df_intensive_caregivers = df_caregivers[df_caregivers["intensive_care"] == 1]

    _df_year = df[df["syear"] == 2012]  # 2016 # noqa: PLR2004
    # # df_year = df[df["syear"].between(2012, 2018)]

    df["kidage_youngest"] = df["kidage_youngest"] - 1

    plot_choice_shares_by_education_emp(
        data_emp=df,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        path_to_save_plot=path_to_save_labor_supply_all_by_age,
    )
    plot_choice_shares_by_education_emp(
        data_emp=df_non_caregivers,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        path_to_save_plot=path_to_save_labor_supply_non_caregivers_by_age,
    )
    plot_choice_shares_by_education_emp(
        data_emp=df_caregivers,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        path_to_save_plot=path_to_save_labor_supply_caregivers_by_age,
    )

    # Health
    plot_choice_shares_by_education_emp(
        data_emp=df_good_health,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        path_to_save_plot=path_to_save_labor_supply_good_health_by_age,
    )
    plot_choice_shares_by_education_emp(
        data_emp=df_bad_health,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        path_to_save_plot=path_to_save_labor_supply_bad_health_by_age,
    )

    plot_choice_shares_by_education_age_bins_emp(
        data_emp=df,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        bin_width=5,
        path_to_save_plot=path_to_save_labor_supply_all_by_age_bins,
    )

    plot_choice_shares_by_education_age_bins_emp(
        data_emp=df_caregivers,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        bin_width=5,
        path_to_save_plot=path_to_save_labor_supply_caregivers_by_age_bins,
    )
    plot_choice_shares_by_education_age_bins_emp(
        data_emp=df_light_caregivers,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        bin_width=5,
        path_to_save_plot=path_to_save_labor_supply_light_caregivers_by_age_bins,
    )
    plot_choice_shares_by_education_age_bins_emp(
        data_emp=df_intensive_caregivers,
        specs=specs,
        age_min=start_age,
        age_max=end_age,
        bin_width=5,
        path_to_save_plot=path_to_save_labor_supply_intensive_caregivers_by_age_bins,
    )


def plot_choice_shares_by_education_emp(
    data_emp, specs, age_min=None, age_max=None, path_to_save_plot=None
):
    """
    Plot observed (empirical) choice-specific shares by age and education,
    only using the `data_emp` dataset.

    Expects:
        - `data_emp` has columns: ["sex", "education", "age", "choice"].
        - `specs` contains:
            * "start_age", "end_age_msm"
            * "education_labels" (list of edu labels, index = edu code)
            * "choice_labels" (list of labels for the 4 aggregated choices)
    Uses global constants:
        - SEX (scalar {0,1})
        - RETIREMENT_CHOICES, UNEMPLOYED_CHOICES, PART_TIME_CHOICES, FULL_TIME_CHOICES
    """

    # ---------- 1. Map raw codes → 4-way aggregated choice ------------------
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }

    df = data_emp.copy()

    for agg_code, raw_codes in choice_groups_emp.items():
        df.loc[df["choice"].isin(np.asarray(raw_codes).tolist()), "choice_group"] = (
            agg_code
        )

    # if any rows didn't match mapping, this will raise; ensure mapping is complete
    df["choice_group"] = df["choice_group"].astype(int)

    # ---------- 2. Plotting setup ------------------------------------------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    sex = SEX  # assumed scalar {0,1}

    n_edu = len(specs["education_labels"])
    n_choices = len(specs["choice_labels"])  # after aggregation (should be 4)

    fig, axs = plt.subplots(n_edu, n_choices, figsize=(16, 6), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)  # robust indexing when n_edu==1 or n_choices==1

    # ---------- 3. Loop over education groups -------------------------------
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        emp_edu = df[(df["sex"] == sex) & (df["education"] == edu_var)]

        # shares by age × aggregated choice
        emp_shares = (
            emp_edu.groupby("age")["choice_group"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # Ensure we have a value for each age and each aggregated choice column
        ages = range(age_min, age_max + 1)
        emp_shares = emp_shares.reindex(index=ages, fill_value=0)
        emp_shares = emp_shares.reindex(columns=range(n_choices), fill_value=0)

        # ---------- 4. Plot each aggregated choice --------------------------
        for choice_var in range(n_choices):
            ax = axs[edu_var, choice_var]
            vals_emp = emp_shares[choice_var]

            ax.plot(ages, vals_emp, ls="--", label="Observed")

            ax.set_title(specs["choice_labels"][choice_var])
            ax.set_ylim(0, 1)
            ax.set_xlim(age_min, age_max)
            ax.set_xlabel("Age")
            ax.tick_params(labelbottom=True)

            if choice_var == 0:
                ax.set_ylabel(f"{edu_label}\nShare")
                ax.legend()

    plt.tight_layout()
    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_choice_shares_by_education_age_bins_emp(  # noqa: PLR0912, PLR0915
    data_emp,
    specs,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    bin_width: int = 5,
    path_to_save_plot: str | None = None,
):
    """
    Plot observed (empirical) aggregated-choice shares by *age bins* and education.
    Each panel contains one aggregated choice; rows are education groups.

    X-axis labels are range-style (e.g., "55-59"), rotated 45°, and shown at the bottom
    of every subplot. Adds small left/right/top/bottom whitespace so lines remain
    visible when they hit the axes.

    """

    # ── 1. Map raw choice codes → 4 aggregated groups ────────────────────────
    choice_groups_emp = {
        0: RETIREMENT_CHOICES,
        1: UNEMPLOYED_CHOICES,
        2: PART_TIME_CHOICES,
        3: FULL_TIME_CHOICES,
    }

    emp = data_emp.copy()
    for g, raw in choice_groups_emp.items():
        emp.loc[emp["choice"].isin(np.atleast_1d(raw)), "choice_group"] = g
    emp["choice_group"] = emp["choice_group"].astype(int)

    # ── 2. Build age bins & labels ───────────────────────────────────────────
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]

    # Bin edges: [start, next_start, ..., age_max+1]
    edges = list(range(age_min, age_max + 1, bin_width))
    if edges[-1] <= age_max:
        edges.append(age_max + 1)

    # Left-edge x positions and labels like "55-59"
    bin_starts = edges[:-1]
    bin_labels = [
        f"{start}-{end - 1}" for start, end in zip(edges[:-1], edges[1:], strict=False)
    ]

    sex = SEX
    edu_labels = specs["education_labels"]
    choice_labels = specs["choice_labels"]
    n_edu, n_choices = len(edu_labels), len(choice_labels)

    # sharey only—no sharex so we can force labels on each row cleanly
    fig, axs = plt.subplots(
        n_edu, n_choices, figsize=(4 * n_choices, 3 * n_edu), sharey=True
    )

    # Ensure axs is 2D with shape (n_edu, n_choices)
    if n_edu == n_choices == 1:
        axs = np.array([[axs]])
    elif n_edu == 1:
        axs = axs[np.newaxis, :]
    elif n_choices == 1:
        axs = axs[:, np.newaxis]

    # --- Padding amounts ---
    y_pad = 0.03  # adds whitespace above 1 and below 0
    # compute a representative bin step for x padding
    if len(bin_starts) > 1:
        step = float(np.median(np.diff(bin_starts)))
    else:
        step = float(bin_width) if bin_width is not None else 1.0
    x_pad = 0.35 * step  # "little" left/right whitespace (~35% of one bin)

    # ── 3. Loop over education × choice bins ─────────────────────────────────
    for i, edu_label in enumerate(edu_labels):
        emp_sub = emp[(emp["sex"] == sex) & (emp["education"] == i)]

        for j in range(n_choices):
            ax = axs[i, j]
            emp_rates = []

            for start, end in zip(edges[:-1], edges[1:], strict=False):
                emp_bin = emp_sub[(emp_sub["age"] >= start) & (emp_sub["age"] < end)]
                emp_rates.append(
                    (emp_bin["choice_group"] == j).mean() if len(emp_bin) else np.nan
                )

            ax.plot(bin_starts, emp_rates, ls="--", label="Observed")
            ax.set_title(choice_labels[j])

            # --- Uniform axes + whitespace on all sides ---
            ax.set_ylim(-y_pad, 1 + y_pad)  # bottom/top whitespace
            if len(bin_starts) > 1:
                ax.set_xlim(
                    bin_starts[0] - x_pad, bin_starts[-1] + x_pad
                )  # left/right whitespace
            else:
                # single bin: give symmetric whitespace
                ax.set_xlim(bin_starts[0] - x_pad, bin_starts[0] + x_pad)

            # --- Range-style x-axis labels (e.g., "55-59"), rotated 45° ---
            ax.set_xticks(bin_starts)
            ax.set_xticklabels(bin_labels)
            ax.tick_params(
                axis="x", labelbottom=True, labeltop=False, bottom=True, top=False
            )
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_ha("right")
                lbl.set_rotation_mode("anchor")

            if i == n_edu - 1:
                ax.set_xlabel("Age (bin range)")
            if j == 0:
                ax.set_ylabel(f"{edu_label}\nShare")
                ax.legend()

    plt.tight_layout()
    # If labels are still tight, you can nudge margins a bit:
    # plt.subplots_adjust(bottom=0.15, left=0.10, right=0.98, top=0.92)

    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300)
