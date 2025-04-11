"""Plot income processes."""

import pickle as pkl
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.pensions import (
    calc_gross_pension_income,
    calc_pensions_after_ssc,
)
from caregiving.model.wealth_and_budget.transfers import calc_child_benefits
from caregiving.model.wealth_and_budget.wages import (
    calc_labor_income_after_ssc,
    calculate_gross_labor_income,
)


def task_plot_incomes(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
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
                    options=specs,
                )
                after_ssc_pt_wages[i] = calc_labor_income_after_ssc(
                    lagged_choice=2,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    options=specs,
                )

                gross_ft_wages[i] = calculate_gross_labor_income(
                    lagged_choice=3,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    options=specs,
                )
                after_ssc_ft_wages[i] = calc_labor_income_after_ssc(
                    lagged_choice=3,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    options=specs,
                )

                gross_pensions[i] = np.maximum(
                    calc_gross_pension_income(
                        experience_years=exp,
                        education=edu_var,
                        sex=sex_var,
                        options=specs,
                    ),
                    annual_unemployment,
                )

                net_pensions[i] = np.maximum(
                    calc_pensions_after_ssc(
                        experience_years=exp,
                        education=edu_var,
                        sex=sex_var,
                        options=specs,
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


def task_plot_total_household_income(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "total_income.png",
):
    """Plot total household income."""

    with path_to_full_specs.open("rb") as file:
        specs = pkl.load(file)

    params = {"interest_rate": 0.0}
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
                    total_income[i] = budget_constraint(
                        period=period,
                        education=edu_val,
                        lagged_choice=choice,
                        experience=exp_share,
                        # sex=sex_var,
                        partner_state=np.array(married_val),
                        savings_end_of_previous_period=0,
                        income_shock_previous_period=0,
                        params=params,
                        options=specs,
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


def task_plot_partner_wage(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_data: Path = BLD / "data" / "soep_partner_wage_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
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


def task_plot_child_benefits(
    path_to_full_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
    path_to_save: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
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
                        options=specs,
                    )
                ax.plot(ages, child_benefits, label=str(edu_label))

            ax.set_title(f"{sex_label}; {partner_label}")
            ax.set_xlabel("Age")
            ax.legend()

    fig.suptitle("Child benefits")

    fig.tight_layout()
    fig.savefig(path_to_save, transparent=True, dpi=300)


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
