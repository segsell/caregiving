"""Plot income processes."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
from matplotlib.lines import Line2D
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.model.shared import (
    FULL_TIME,
    FULL_TIME_CARE,
    PART_TIME,
    PART_TIME_CARE,
    UNEMPLOYED_CARE,
)
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.budget_equation_caregiving_leave_with_job_retention import (  # noqa: E501
    calc_caregiving_leave_top_up,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_gross_pension_income,
    calc_pensions_after_ssc,
)
from caregiving.model.wealth_and_budget.transfers import calc_child_benefits
from caregiving.model.wealth_and_budget.wages import (
    calc_labor_income_after_ssc,
    calculate_gross_labor_income,
)

EXP_MOD_5 = 5
EXP_MOD_7 = 7


@pytask.mark.budget_constraint
def task_plot_caregiving_leave_top_ups(  # noqa: PLR0915
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "caregiving_leave_top_ups.png",
):
    """Plot caregiving-leave wage top-ups by experience and prior job status.

    For a representative sex and education type, we plot the annual top-up amounts
    across experience years for the following combinations:

    - Prior none, now unemployed caregiver
    - Prior PT, now unemployed caregiver
    - Prior FT, now unemployed caregiver
    - Prior FT, now PT caregiver
    """

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    # Representative sex and education (e.g. women, medium education)
    sex_var = 1  # as in task_plot_total_household_income
    edu_var = 1 if len(specs["education_labels"]) > 1 else 0

    # Experience years grid
    exp_levels = np.arange(0, 51)

    # Representative caregiving choices (with informal care)
    choice_unemp_care = int(UNEMPLOYED_CARE[0])
    choice_pt_care = int(PART_TIME_CARE[0])
    # Representative non-caregiving work choices
    choice_pt_noncg = int(PART_TIME[0])
    choice_ft_noncg = int(FULL_TIME[0])

    # Containers for top-ups
    topup_prior_none_unemp = np.zeros_like(exp_levels, dtype=float)
    topup_prior_pt_unemp = np.zeros_like(exp_levels, dtype=float)
    topup_prior_ft_unemp = np.zeros_like(exp_levels, dtype=float)
    topup_prior_ft_pt = np.zeros_like(exp_levels, dtype=float)

    for i, exp in enumerate(exp_levels):
        experience_years = float(exp)
        income_shock = 0.0

        # Labor income for PT caregiving (used for FT→PT gap)
        labor_income_pt_care = calc_labor_income_after_ssc(
            lagged_choice=choice_pt_care,
            experience_years=experience_years,
            education=edu_var,
            sex=sex_var,
            income_shock=income_shock,
            model_specs=specs,
        )

        # Prior none (0), now unemployed caregiver
        topup_prior_none_unemp[i] = float(
            calc_caregiving_leave_top_up(
                lagged_choice=choice_unemp_care,
                education=edu_var,
                job_before_caregiving=0,
                experience_years=experience_years,
                income_shock_previous_period=income_shock,
                sex=sex_var,
                labor_income_after_ssc=0.0,
                model_specs=specs,
            )
        )

        # Prior PT (1), now unemployed caregiver
        topup_prior_pt_unemp[i] = float(
            calc_caregiving_leave_top_up(
                lagged_choice=choice_unemp_care,
                education=edu_var,
                job_before_caregiving=1,
                experience_years=experience_years,
                income_shock_previous_period=income_shock,
                sex=sex_var,
                labor_income_after_ssc=0.0,
                model_specs=specs,
            )
        )

        # Prior FT (2), now unemployed caregiver
        topup_prior_ft_unemp[i] = float(
            calc_caregiving_leave_top_up(
                lagged_choice=choice_unemp_care,
                education=edu_var,
                job_before_caregiving=2,
                experience_years=experience_years,
                income_shock_previous_period=income_shock,
                sex=sex_var,
                labor_income_after_ssc=0.0,
                model_specs=specs,
            )
        )

        # Prior FT (2), now PT caregiver
        topup_prior_ft_pt[i] = float(
            calc_caregiving_leave_top_up(
                lagged_choice=choice_pt_care,
                education=edu_var,
                job_before_caregiving=2,
                experience_years=experience_years,
                income_shock_previous_period=income_shock,
                sex=sex_var,
                labor_income_after_ssc=labor_income_pt_care,
                model_specs=specs,
            )
        )

    # Baseline labor income (after SSC) in the caregiving states
    labor_prior_none_unemp = np.zeros_like(exp_levels, dtype=float)
    labor_prior_pt_unemp = np.zeros_like(exp_levels, dtype=float)
    labor_prior_ft_unemp = np.zeros_like(exp_levels, dtype=float)
    labor_prior_ft_pt = np.zeros_like(exp_levels, dtype=float)

    for i, exp in enumerate(exp_levels):
        experience_years = float(exp)
        income_shock = 0.0

        # Labor income for unemployed caregiver
        # (should be zero but computed for clarity)
        labor_unemp_care = calc_labor_income_after_ssc(
            lagged_choice=choice_unemp_care,
            experience_years=experience_years,
            education=edu_var,
            sex=sex_var,
            income_shock=income_shock,
            model_specs=specs,
        )

        labor_pt_care = calc_labor_income_after_ssc(
            lagged_choice=choice_pt_care,
            experience_years=experience_years,
            education=edu_var,
            sex=sex_var,
            income_shock=income_shock,
            model_specs=specs,
        )

        labor_prior_none_unemp[i] = labor_unemp_care
        labor_prior_pt_unemp[i] = labor_unemp_care
        labor_prior_ft_unemp[i] = labor_unemp_care
        labor_prior_ft_pt[i] = labor_pt_care

    # Baseline non-caregiving wages (after SSC) for PT and FT work
    wage_pt_noncg = np.zeros_like(exp_levels, dtype=float)
    wage_ft_noncg = np.zeros_like(exp_levels, dtype=float)

    for i, exp in enumerate(exp_levels):
        experience_years = float(exp)
        income_shock = 0.0

        wage_pt_noncg[i] = calc_labor_income_after_ssc(
            lagged_choice=choice_pt_noncg,
            experience_years=experience_years,
            education=edu_var,
            sex=sex_var,
            income_shock=income_shock,
            model_specs=specs,
        )

        wage_ft_noncg[i] = calc_labor_income_after_ssc(
            lagged_choice=choice_ft_noncg,
            experience_years=experience_years,
            education=edu_var,
            sex=sex_var,
            income_shock=income_shock,
            model_specs=specs,
        )

    # Combined income = labor income + top-up
    total_prior_none_unemp = labor_prior_none_unemp + topup_prior_none_unemp
    total_prior_pt_unemp = labor_prior_pt_unemp + topup_prior_pt_unemp
    total_prior_ft_unemp = labor_prior_ft_unemp + topup_prior_ft_unemp
    total_prior_ft_pt = labor_prior_ft_pt + topup_prior_ft_pt

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1) Pure top-ups (solid)
    ax.plot(
        exp_levels,
        topup_prior_none_unemp,
        label="Top-up: prior none, now unemployed",
        color=JET_COLOR_MAP[0],
    )
    ax.plot(
        exp_levels,
        topup_prior_pt_unemp,
        label="Top-up: prior PT, now unemployed",
        color=JET_COLOR_MAP[1],
    )
    ax.plot(
        exp_levels,
        topup_prior_ft_unemp,
        label="Top-up: prior FT, now unemployed",
        color=JET_COLOR_MAP[2],
    )
    ax.plot(
        exp_levels,
        topup_prior_ft_pt,
        label="Top-up: prior FT, now PT",
        color=JET_COLOR_MAP[3],
    )

    # Precompute masks for specific experience years
    mask_triangles = exp_levels % 10 == EXP_MOD_5  # 5, 15, 25, ...
    mask_circles = exp_levels % 10 == 0  # 0, 10, 20, ...
    mask_baseline = exp_levels % 10 == EXP_MOD_7  # 7, 17, 27, ...

    # 2) Labor income without top-ups (dashed lines, plus triangles at odd years)
    ax.plot(
        exp_levels,
        labor_prior_none_unemp,
        label="Labor income: prior none, now unemployed",
        color=JET_COLOR_MAP[0],
        linestyle="--",
    )
    ax.plot(
        exp_levels,
        labor_prior_pt_unemp,
        label="Labor income: prior PT, now unemployed",
        color=JET_COLOR_MAP[1],
        linestyle="--",
    )
    ax.plot(
        exp_levels,
        labor_prior_ft_unemp,
        label="Labor income: prior FT, now unemployed",
        color=JET_COLOR_MAP[2],
        linestyle="--",
    )
    ax.plot(
        exp_levels,
        labor_prior_ft_pt,
        label="Labor income: prior FT, now PT",
        color=JET_COLOR_MAP[3],
        linestyle="--",
    )
    # Triangular markers at specific experience years (5, 15, 25, ...)
    ax.scatter(
        exp_levels[mask_triangles],
        labor_prior_none_unemp[mask_triangles],
        color=JET_COLOR_MAP[0],
        marker="^",
        s=15,
    )
    ax.scatter(
        exp_levels[mask_triangles],
        labor_prior_pt_unemp[mask_triangles],
        color=JET_COLOR_MAP[1],
        marker="^",
        s=15,
    )
    ax.scatter(
        exp_levels[mask_triangles],
        labor_prior_ft_unemp[mask_triangles],
        color=JET_COLOR_MAP[2],
        marker="^",
        s=15,
    )
    ax.scatter(
        exp_levels[mask_triangles],
        labor_prior_ft_pt[mask_triangles],
        color=JET_COLOR_MAP[3],
        marker="^",
        s=15,
    )

    # 3) Labor income + top-ups (solid lines, plus dots at even years)
    ax.plot(
        exp_levels,
        total_prior_none_unemp,
        label="Total: prior none, now unemployed",
        color=JET_COLOR_MAP[0],
        linestyle="-",
    )
    ax.plot(
        exp_levels,
        total_prior_pt_unemp,
        label="Total: prior PT, now unemployed",
        color=JET_COLOR_MAP[1],
        linestyle="-",
    )
    ax.plot(
        exp_levels,
        total_prior_ft_unemp,
        label="Total: prior FT, now unemployed",
        color=JET_COLOR_MAP[2],
        linestyle="-",
    )
    ax.plot(
        exp_levels,
        total_prior_ft_pt,
        label="Total: prior FT, now PT",
        color=JET_COLOR_MAP[3],
        linestyle="-",
    )
    # Dot markers at specific experience years (0, 10, 20, ...)
    ax.scatter(
        exp_levels[mask_circles],
        total_prior_none_unemp[mask_circles],
        color=JET_COLOR_MAP[0],
        marker="o",
        s=15,
    )
    ax.scatter(
        exp_levels[mask_circles],
        total_prior_pt_unemp[mask_circles],
        color=JET_COLOR_MAP[1],
        marker="o",
        s=15,
    )
    ax.scatter(
        exp_levels[mask_circles],
        total_prior_ft_unemp[mask_circles],
        color=JET_COLOR_MAP[2],
        marker="o",
        s=15,
    )
    ax.scatter(
        exp_levels[mask_circles],
        total_prior_ft_pt[mask_circles],
        color=JET_COLOR_MAP[3],
        marker="o",
        s=15,
    )

    # 4) Baseline non-caregiving PT and FT wages (dotted lines + square markers)
    ax.plot(
        exp_levels,
        wage_pt_noncg,
        label="Baseline PT wage (no caregiving)",
        color="black",
        linestyle=":",
    )
    ax.plot(
        exp_levels,
        wage_ft_noncg,
        label="Baseline FT wage (no caregiving)",
        color="gray",
        linestyle=":",
    )
    ax.scatter(
        exp_levels[mask_baseline],
        wage_pt_noncg[mask_baseline],
        color="black",
        marker="s",
        s=20,
    )
    ax.scatter(
        exp_levels[mask_baseline],
        wage_ft_noncg[mask_baseline],
        color="gray",
        marker="D",
        s=20,
    )

    ax.set_xlabel("Experience years")
    ax.set_ylabel("Annual amount (EUR)")
    ax.set_title("Caregiving leave wage top-ups and labor income by experience")

    # Custom legend: one entry per scenario and type, with matching styles
    legend_handles = [
        # Top-ups (solid)
        Line2D([0], [0], color=JET_COLOR_MAP[0], linestyle="-"),
        Line2D([0], [0], color=JET_COLOR_MAP[1], linestyle="-"),
        Line2D([0], [0], color=JET_COLOR_MAP[2], linestyle="-"),
        Line2D([0], [0], color=JET_COLOR_MAP[3], linestyle="-"),
        # Labor income (dashed + triangles)
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[0],
            linestyle="--",
            marker="^",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[1],
            linestyle="--",
            marker="^",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[2],
            linestyle="--",
            marker="^",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[3],
            linestyle="--",
            marker="^",
            markersize=5,
        ),
        # Total income (solid + dots)
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[0],
            linestyle="-",
            marker="o",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[1],
            linestyle="-",
            marker="o",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[2],
            linestyle="-",
            marker="o",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color=JET_COLOR_MAP[3],
            linestyle="-",
            marker="o",
            markersize=5,
        ),
        # Baseline PT and FT wages
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            marker="s",
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle=":",
            marker="D",
            markersize=5,
        ),
    ]
    legend_labels = [
        # Top-ups
        "Top-up: prior none, now unemployed",
        "Top-up: prior PT, now unemployed",
        "Top-up: prior FT, now unemployed",
        "Top-up: prior FT, now PT",
        # Labor income
        "Labor income: prior none, now unemployed",
        "Labor income: prior PT, now unemployed",
        "Labor income: prior FT, now unemployed",
        "Labor income: prior FT, now PT",
        # Total income
        "Total: prior none, now unemployed",
        "Total: prior PT, now unemployed",
        "Total: prior FT, now unemployed",
        "Total: prior FT, now PT",
        # Baseline wages
        "Baseline PT wage (no caregiving)",
        "Baseline FT wage (no caregiving)",
    ]
    ax.legend(legend_handles, legend_labels, loc="best")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path_to_save, dpi=300)
    plt.close(fig)


@pytask.mark.plot_incomes
def task_plot_incomes(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "income.png",
):
    """Plot before and after ssc incomes.

    Full-time, part-time, pensions and unemployment benefits.

    """

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    exp_levels = np.arange(0, 50)

    annual_unemployment = specs["annual_unemployment_benefits"]
    unemployment_benefits = np.ones_like(exp_levels) * annual_unemployment

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        # Now loop over education to generate specific net and gross wages and pensions
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axes[sex_var, edu_var]

            ax.plot(
                exp_levels,
                unemployment_benefits,
                label="Unemployment benefits",
                color=JET_COLOR_MAP[0],
            )

            # Initialize empty containers, part and full time wages and pensions
            gross_pt_wages = np.zeros_like(exp_levels, dtype=float)
            after_ssc_pt_wages = np.zeros_like(exp_levels, dtype=float)

            gross_ft_wages = np.zeros_like(exp_levels, dtype=float)
            after_ssc_ft_wages = np.zeros_like(exp_levels, dtype=float)

            net_pensions = np.zeros_like(exp_levels, dtype=float)
            gross_pensions = np.zeros_like(exp_levels, dtype=float)
            for i, exp in enumerate(exp_levels):
                gross_pt_wages[i] = calculate_gross_labor_income(
                    lagged_choice=2,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )
                after_ssc_pt_wages[i] = calc_labor_income_after_ssc(
                    lagged_choice=2,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )

                gross_ft_wages[i] = calculate_gross_labor_income(
                    lagged_choice=3,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )
                after_ssc_ft_wages[i] = calc_labor_income_after_ssc(
                    lagged_choice=3,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )

                gross_pensions[i] = np.maximum(
                    calc_gross_pension_income(
                        experience_years=exp,
                        education=edu_var,
                        sex=sex_var,
                        model_specs=specs,
                    ),
                    annual_unemployment,
                )

                net_pensions[i] = np.maximum(
                    calc_pensions_after_ssc(
                        experience_years=exp,
                        education=edu_var,
                        sex=sex_var,
                        model_specs=specs,
                    ),
                    annual_unemployment,
                )

            ax.plot(
                exp_levels,
                after_ssc_pt_wages,
                label="Average after ssc pt wage",
                color=JET_COLOR_MAP[1],
            )
            ax.plot(
                exp_levels,
                gross_pt_wages,
                label="Average gross pt wage",
                ls="--",
                color=JET_COLOR_MAP[1],
            )

            ax.plot(
                exp_levels,
                after_ssc_ft_wages,
                label="Average after ssc ft wage",
                color=JET_COLOR_MAP[2],
            )
            ax.plot(
                exp_levels,
                gross_ft_wages,
                label="Average gross ft wage",
                ls="--",
                color=JET_COLOR_MAP[2],
            )

            ax.plot(
                exp_levels,
                net_pensions,
                label="Average after ssc pension",
                color=JET_COLOR_MAP[3],
            )
            ax.plot(
                exp_levels,
                gross_pensions,
                label="Average gross pension",
                ls="--",
                color=JET_COLOR_MAP[3],
            )

            ax.legend(loc="upper left")
            ax.set_xlabel("Experience")
            ax.set_ylabel("Average income")
            ax.set_title(f"{sex_label}; {edu_label}")

    fig.suptitle("After ssc income")
    fig.tight_layout()
    fig.savefig(path_to_save, transparent=True, dpi=300)
    plt.close(fig)


def task_plot_total_household_income(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "total_income.png",
):
    """Plot total household income."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    params = {}
    exp_levels = np.arange(0, 50 + 1)  # specs["max_experience"] + 1
    marriage_labels = ["Single", "Partnered"]
    edu_labels = specs["education_labels"]

    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    sex_var = 1
    sex_label = specs["sex_labels"][sex_var]

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    for married_val, married_label in enumerate(marriage_labels):
        for edu_val, edu_label in enumerate(edu_labels):
            for choice, work_label in enumerate(specs["choice_labels"]):
                total_income = np.zeros_like(exp_levels, dtype=float)
                for i, exp in enumerate(exp_levels):
                    if work_label == "Retired":
                        period = 45
                    else:
                        period = exp
                    exp_share = exp / (exp + specs["max_exp_diffs_per_period"][period])
                    total_income[i], _budget_aux = budget_constraint(
                        period=period,
                        education=edu_val,
                        lagged_choice=choice,
                        experience=exp_share,
                        # sex=sex_var,
                        partner_state=np.array(married_val),
                        care_demand=0,
                        mother_dead=0,
                        asset_end_of_previous_period=0,
                        income_shock_previous_period=0,
                        params=params,
                        model_specs=specs,
                    )
                axs[edu_val, married_val].plot(
                    exp_levels,
                    total_income * specs["wealth_unit"],
                    label=str(work_label),
                )
            axs[edu_val, married_val].set_title(f"{edu_label} and {married_label}")
            axs[edu_val, married_val].set_xlabel("Period equals experience")
            # axs[edu_val, married_val].set_ylim([0, 80])
            axs[edu_val, married_val].legend()

    fig.suptitle(f"Total income; {sex_label}")

    fig.tight_layout()
    fig.savefig(path_to_save, transparent=True, dpi=300)
    plt.close(fig)


def task_plot_partner_wage(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_wage_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "partner_wage_ugly.png",
):
    """Plot the partner wage by age."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])

    start_age = specs["start_age"]

    wage_data = df.groupby(["sex", "education", "age"])["wage_p"].mean()
    partner_wage_est = specs["annual_partner_wage"]

    fig, axs = plt.subplots(ncols=2)

    # Only plot until 70
    max_period = 40
    max_age = start_age + max_period - 1
    periods = np.arange(40)

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_val, edu_label in enumerate(specs["education_labels"]):
            ax.plot(
                periods,
                wage_data.loc[(sex_var, edu_val, slice(start_age, max_age))] * 12,
                label=f"edu {edu_label}",
            )
            ax.plot(
                periods,
                partner_wage_est[sex_var, edu_val, :max_period],
                linestyle="--",
                label=f"edu {edu_label} est",
            )

        ax.legend()
        ax.set_title(f"Partner wage of {sex_label} by period")

    # Save the figure
    fig.suptitle("Partner wage")
    fig.tight_layout()
    fig.savefig(path_to_save, transparent=True, dpi=300)
    plt.close(fig)


def task_plot_child_benefits(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "wealth_and_budget"
    / "child_benefits.png",
):

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    max_period = 40
    start_age = specs["start_age"]
    periods = np.arange(max_period)
    ages = np.arange(start_age, start_age + max_period)

    education_labels = specs["education_labels"]
    marriage_labels = ["Single", "Partnered"]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):

        for partner_val, partner_label in enumerate(marriage_labels):
            ax = axs[sex_var, partner_val]

            for edu_val, edu_label in enumerate(education_labels):
                child_benefits = np.zeros_like(periods, dtype=float)

                for i, period in enumerate(periods):
                    child_benefits[i] = calc_child_benefits(
                        education=edu_val,
                        sex=sex_var,
                        has_partner_int=partner_val,
                        period=period,
                        model_specs=specs,
                    )
                ax.plot(ages, child_benefits, label=str(edu_label))

            ax.set_title(f"{sex_label}; {partner_label}")
            ax.set_xlabel("Age")
            ax.legend()

    fig.suptitle("Child benefits")

    fig.tight_layout()
    fig.savefig(path_to_save, transparent=True, dpi=300)
    plt.close(fig)


# def plot_wages(path_dict):
#     specs = generate_derived_and_data_derived_specs(path_dict)
#
#     exp_levels = np.arange(46)
#     # Initialize empty containers
#     gross_wages = np.zeros_like(exp_levels)
#     net_wages = np.zeros_like(exp_levels)
#
#     net_pensions = np.zeros_like(exp_levels)
#     gross_pensions = np.zeros_like(exp_levels)
#     for i, exp in enumerate(exp_levels):
#         gross_wages[i] = calculate_gross_labor_income(exp, edu, 0, specs)
#         net_wages[i] = calc_labor_income(exp, edu, 0, specs)
#
#         gross_pensions[i] = calc_gross_pension_income(exp, edu, 0, 2, specs)
#         net_pensions[i] = calc_pensions(exp, edu, 0, 2, specs)
#
#     unemployment_benefits = (
#         np.ones_like(exp_levels) * specs["monthly_unemployment_benefits"] * 12
#     )
#
#     fig, ax = plt.subplots()
#     ax.plot(exp_levels, net_wages, label="Average net wage")
#     ax.plot(exp_levels, gross_wages, label="Average gross wage")
#     ax.plot(exp_levels, unemployment_benefits, label="Unemployment benefits")
#     ax.legend(loc="upper left")
#     ax.set_xlabel("Experience")
#     ax.set_ylabel("Average wage")
