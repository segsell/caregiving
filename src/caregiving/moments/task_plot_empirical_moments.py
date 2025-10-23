"""Plot raw SOEP data."""

import re
from itertools import product
from pathlib import Path
from typing import Annotated, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
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
    WEALTH_MOMENTS_SCALE,
    WEALTH_QUANTILE_CUTOFF,
    WORK,
)
from caregiving.moments.task_create_soep_moments import adjust_and_trim_wealth_data
from caregiving.specs.task_write_specs import read_and_derive_specs
from caregiving.utils import table

DEGREES_OF_FREEDOM = 1


@pytask.mark.emp_moms
def task_plot_empirical_soep_moments(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_soep_moments: Path = BLD / "moments" / "soep_moments_new.csv",
    path_to_main_sample: Path = BLD / "data" / "soep_structural_estimation_sample.csv",
    path_to_caregivers_sample: Path = BLD
    / "data"
    / "soep_structural_caregivers_sample.csv",
    path_to_save_wealth: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "wealth_by_age_and_education.png",
    path_to_save_wealth_from_moments: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "wealth_by_age_and_education_from_moments.png",
    path_to_save_wealth_empirical_vs_moments: Annotated[Path, Product] = BLD
    / "plots"
    / "raw_moments"
    / "wealth_by_age_and_education_empirical_vs_moments.png",
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

    soep_moments = pd.read_csv(path_to_soep_moments, index_col=[0])

    df_full = pd.read_csv(path_to_main_sample, index_col=[0])
    df = df_full[
        (df_full["sex"] == 1) & (df_full["age"] <= end_age + 10)
    ].copy()  # women only

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

    # df_wealth = pd.read_csv(path_to_wealth_sample, index_col=[0])
    df_wealth = df_full[df_full["sex"] == SEX].copy()
    trimmed = adjust_and_trim_wealth_data(df=df_wealth, specs=specs)

    # Wealth
    plot_wealth_emp(
        data_emp=trimmed,
        # data_emp=df_wealth,
        specs=specs,
        wealth_var_emp="adjusted_wealth",
        median=False,
        age_min=30,
        age_max=100,
        path_to_save_plot=path_to_save_wealth,
    )
    plot_wealth_from_moments(
        soep_moments,
        specs=specs,
        age_min=30,
        age_max=100,
        path_to_save_plot=path_to_save_wealth_from_moments,
    )
    plot_wealth_emp_vs_moments(
        data_emp=trimmed,
        moments=soep_moments / WEALTH_MOMENTS_SCALE,
        specs=specs,
        wealth_var_emp="adjusted_wealth",
        median=False,
        age_min=30,
        age_max=100,
        path_to_save_plot=path_to_save_wealth_empirical_vs_moments,
    )

    # Labor supply
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


# ======================================================================================
# Wealth
# ======================================================================================


def plot_wealth_emp(
    data_emp: pd.DataFrame,
    specs: dict,
    *,
    wealth_var_emp: str,
    median: bool = False,
    age_min: int | None = None,
    age_max: int | None = None,
    path_to_save_plot: str | None = None,
):
    """
    Plot empirical average/median wealth by age and education only.
    Parameters
    ----------
    data_emp : DataFrame
        Must include columns: 'age', 'education', and the wealth variable.
    specs : dict
        Expects:
          - 'start_age', 'end_age_msm'
          - 'education_labels' : list[str]
    wealth_var_emp : str
        Column name of wealth in the empirical dataset.
    median : bool, default False
        If True, plot median; else plot mean.
    age_min, age_max : int | None
        Plot range. Defaults to [specs['start_age'], specs['end_age_msm']].
    path_to_save_plot : str | None
        If provided, saves the figure.
    """
    # ---------- 0. Setup ----------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    ages = range(age_min, age_max + 1)
    stat_name = "Median" if median else "Average"

    n_edu = len(specs["education_labels"])
    fig, axs = plt.subplots(1, n_edu, figsize=(5 * n_edu, 4), sharex=True, sharey=True)
    if n_edu == 1:
        axs = np.array([axs])

    agg = (
        (lambda s: s.median(skipna=True)) if median else (lambda s: s.mean(skipna=True))
    )

    # ---------- 1. Loop over education groups ----------
    all_values = []
    for edu_idx, edu_label in enumerate(specs["education_labels"]):
        ax = axs[edu_idx]

        emp_edu = data_emp[data_emp["education"] == edu_idx]
        emp_series = (
            emp_edu.groupby("age")[wealth_var_emp]
            .apply(agg)
            .reindex(ages, fill_value=np.nan)
        )

        all_values.extend(emp_series.dropna().tolist())

        ax.plot(ages, emp_series.values, color="black", ls="-", label="Observed")

        # Add internal x padding (5% of the range)
        xrange = age_max - age_min
        pad = int(0.05 * xrange)
        ax.set_xlim(age_min - pad, age_max + pad)

        ax.set_xlabel("Age")
        ax.set_title(edu_label)
        ax.grid(True, alpha=0.2)

        if edu_idx == 0:
            ax.set_ylabel(f"{stat_name} wealth")
            # ax.legend()
        else:
            ax.tick_params(labelleft=False)

    # ---------- 2. Common y-limits ----------
    if all_values:
        ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
        pad = 0.05 * (ymax - ymin)
        for ax in axs:
            ax.set_ylim(ymin - pad, ymax + pad)

    # ---------- 3. Add left y-axis to right panel too ----------
    axs[-1].yaxis.set_ticks_position("left")
    axs[-1].yaxis.set_label_position("left")
    axs[-1].set_ylabel(f"{stat_name} wealth")

    plt.tight_layout()

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)


def plot_wealth_from_moments(  # noqa: PLR0912, PLR0915
    moments: pd.DataFrame,
    specs: dict,
    *,
    age_min: int | None = None,
    age_max: int | None = None,
    path_to_save_plot: str | None = None,
):
    """
    Plot mean wealth moments by age and education (low/high) in the
    same style as `plot_wealth_emp`.

    Parameters
    ----------
    df_moments : DataFrame
        Loaded from CSV. Expect index column 0 to contain moment keys like:
        'mean_wealth_low_education_wealth_age_30'
        and a 'value' column with the numeric moment.
    specs : dict
        Expects:
          - 'start_age', 'end_age_msm'
          - 'education_labels' : list[str] (ordered like [low_label, high_label])
    age_min, age_max : int | None
        Plot range. Defaults to [specs['start_age'], specs['end_age_msm']].
    path_to_save_plot : str | None
        If provided, saves the figure.
    """
    # ---------- 0. Setup ----------
    if moments.index.name is None or moments.index.name != moments.columns[0]:
        # Ensure the moment strings live in the index if user didn't pass index_col=0
        if moments.columns[0] != "value":  # typical load without index_col=0
            moments = moments.set_index(moments.columns[0])

    if "value" not in moments.columns:
        raise ValueError("Expected a 'value' column in df_moments.")

    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    ages = range(age_min, age_max + 1)

    edu_labels = specs["education_labels"]
    n_edu = len(edu_labels)
    fig, axs = plt.subplots(1, n_edu, figsize=(5 * n_edu, 4), sharex=True, sharey=True)
    if n_edu == 1:
        axs = np.array([axs])

    # Helper to extract a Series (index=int age) for a given education tag
    # ("low" or "high")
    def extract_series_for(tag: str) -> pd.Series:
        # pattern captures trailing age
        pattern = re.compile(rf"^mean_wealth_{tag}_education_wealth_age_(\d+)$")
        # extract ages that match
        extracted = moments.index.to_series().str.extract(pattern)
        ages_str = extracted[0]
        mask = ages_str.notna()
        s = moments.loc[mask, "value"].copy()
        s.index = ages_str[mask].astype(int)
        return s.sort_index()

    # We assume two education groups (low/high). If you have more, extend this mapping.
    tag_map = {
        0: "low",
        1: "high",
    }

    # ---------- 1. Loop over education groups ----------
    all_values = []
    for edu_idx in range(n_edu):
        ax = axs[edu_idx]
        tag = tag_map.get(edu_idx)
        if tag is None:
            # If more than 2 groups, skip gracefully
            ax.set_visible(False)
            continue

        s = extract_series_for(tag)
        s = s.reindex(ages, fill_value=np.nan)
        all_values.extend(s.dropna().tolist())

        # plot style: black line, like in plot_wealth_emp
        ax.plot(ages, s.values, color="black", ls="-", label="Moments")

        # Add internal x padding (5% of the range)
        xrange = age_max - age_min
        pad = int(0.05 * xrange)
        ax.set_xlim(age_min - pad, age_max + pad)

        ax.set_xlabel("Age")
        ax.set_title(
            edu_labels[edu_idx] if edu_idx < len(edu_labels) else f"Edu {edu_idx}"
        )
        ax.grid(True, alpha=0.2)

        if edu_idx == 0:
            ax.set_ylabel("Average wealth")
        else:
            ax.tick_params(labelleft=False)

    # ---------- 2. Common y-limits ----------
    if all_values:
        ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        for ax in axs:
            if ax.get_visible():
                ax.set_ylim(ymin - pad, ymax + pad)

    # ---------- 3. Add left y-axis to right panel too ----------
    axs[-1].yaxis.set_ticks_position("left")
    axs[-1].yaxis.set_label_position("left")
    axs[-1].set_ylabel("Average wealth")

    plt.tight_layout()

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)

    return fig, axs


def plot_wealth_emp_vs_moments(  # noqa: PLR0912, PLR0915
    data_emp: pd.DataFrame,
    moments: pd.DataFrame,
    specs: dict,
    *,
    wealth_var_emp: str,
    median: bool = False,
    age_min: int | None = None,
    age_max: int | None = None,
    path_to_save_plot: str | None = None,
):
    """
    Overlay empirical wealth (mean/median) and moment-based mean wealth by age,
    with one subplot per education group (left: low, right: high).

    Parameters
    ----------
    data_emp : DataFrame
        Must include columns: 'age', 'education', and the wealth variable.
    moments : DataFrame
        Loaded from CSV. Expect index col to contain moment keys like:
        'mean_wealth_low_education_wealth_age_30' and a 'value' column.
    specs : dict
        Expects:
          - 'start_age', 'end_age_msm'
          - 'education_labels' : list[str], e.g. ["Low education", "High education"]
    wealth_var_emp : str
        Column name of wealth in the empirical dataset.
    median : bool, default False
        If True, plot median for empirical; else plot mean.
    age_min, age_max : int | None
        Plot range. Defaults to [specs['start_age'], specs['end_age_msm']].
    path_to_save_plot : str | None
        If provided, saves the figure.
    """
    # ---------- 0) Setup ----------
    if age_min is None:
        age_min = specs["start_age"]
    if age_max is None:
        age_max = specs["end_age_msm"]
    ages = range(age_min, age_max + 1)
    stat_name = "Median" if median else "Average"

    # Make sure the moments have the moment keys as index and a 'value' column
    if moments.index.name is None or (
        moments.index.name != moments.columns[0] and "value" in moments.columns
    ):
        # If user didn't load with index_col=0, set first column as index,
        # the moment key column
        if "value" not in moments.columns:
            # Typical CSV: moment,value  (so first column is "moment")
            moments = moments.set_index(moments.columns[0])
        # else: already fine

    if "value" not in moments.columns:
        raise ValueError("Expected a 'value' column in the moments DataFrame.")

    edu_labels = specs["education_labels"]
    n_edu = len(edu_labels)
    fig, axs = plt.subplots(1, n_edu, figsize=(5 * n_edu, 4), sharex=True, sharey=True)
    if n_edu == 1:
        axs = np.array([axs])

    agg = (
        (lambda s: s.median(skipna=True)) if median else (lambda s: s.mean(skipna=True))
    )

    # Helper: parse a Series (index=int age) from moments for a given education tag
    def extract_moment_series(tag: str) -> pd.Series:
        pattern = re.compile(rf"^mean_wealth_{tag}_education_wealth_age_(\d+)$")
        extracted = moments.index.to_series().str.extract(pattern)
        ages_str = extracted[0]
        mask = ages_str.notna()
        s = moments.loc[mask, "value"].copy()
        s.index = ages_str[mask].astype(int)
        return s.sort_index()

    # Map panel index → tag used in moment keys
    tag_map = {0: "low", 1: "high"}

    # ---------- 1) Plot per education ----------
    all_values = []
    for edu_idx, edu_label in enumerate(edu_labels):
        ax = axs[edu_idx]

        # Empirical series
        emp_edu = data_emp[data_emp["education"] == edu_idx]
        emp_series = (
            emp_edu.groupby("age")[wealth_var_emp]
            .apply(agg)
            .reindex(ages, fill_value=np.nan)
        )

        # Moments series
        tag = tag_map.get(edu_idx)
        if tag is None:
            # If more than two education groups exist, hide extra panels gracefully
            ax.set_visible(False)
            continue
        mom_series = extract_moment_series(tag).reindex(ages, fill_value=np.nan)

        # Collect for common y-limits
        all_values.extend(emp_series.dropna().tolist())
        all_values.extend(mom_series.dropna().tolist())

        # Lines (match style spirit: black empirical; dashed moments)
        ax.plot(ages, emp_series.values, color="black", ls="-", label="Observed")
        ax.plot(ages, mom_series.values, color="C1", ls="--", label="Moments")

        # Ax cosmetics
        xrange = age_max - age_min
        pad = int(0.05 * xrange)
        ax.set_xlim(age_min - pad, age_max + pad)
        ax.set_xlabel("Age")
        ax.set_title(edu_label)
        ax.grid(True, alpha=0.2)

        if edu_idx == 0:
            ax.set_ylabel(f"{stat_name} wealth")
        else:
            ax.tick_params(labelleft=False)

    # ---------- 2) Common y-limits ----------
    if all_values:
        ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        for ax in axs:
            if ax.get_visible():
                ax.set_ylim(ymin - pad, ymax + pad)

    # ---------- 3) Mirror left y-axis on right panel ----------
    axs[-1].yaxis.set_ticks_position("left")
    axs[-1].yaxis.set_label_position("left")
    axs[-1].set_ylabel(f"{stat_name} wealth")

    # ---------- 4) Single legend (top center) ----------
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300, transparent=False)

    return fig, axs


# ======================================================================================
# Labor supply choices
# ======================================================================================


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
