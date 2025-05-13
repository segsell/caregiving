"""Estimate the parameters of the HOURLY wage equation using the SOEP panel data.

For each education level, the following equation is estimated:

    ln_wage = beta_0 + beta_1 * ln_(exp+1) + individual_FE + time_FE + epsilon

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


def task_estimate_wage_parameters(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_wage_data.csv",
    path_to_save_plot_men: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "wages_men.png",
    path_to_save_plot_women: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "wages_women.png",
    path_to_save_wage_params: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "wage_eq_params.csv",
    path_to_save_latex: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "wage_eq_params.tex",  # tables
    path_pop_avg_annual_wage: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "pop_avg_annual_wage.npy",
    path_pop_avg_working_hours: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "pop_avg_working_hours.csv",
    # paths_dict["est_results"] + "pop_avg_annual_wage",
    # paths_dict["est_results"] + "population_averages_working_hours.csv"
) -> None:
    """Estimate the wage parameters for each education group in the sample.

    Also estimate for all individuals.

    """

    specs = read_and_derive_specs(path_to_specs)
    specs["start_year"] = 2010
    specs["end_year"] = 2017

    # Specs and data
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    regressors = ["constant", "ln_exp"]
    coefficients = regressors + [param + "_ser" for param in regressors]

    wage_data = pd.read_csv(path_to_data, index_col=0)

    # Modify
    wage_data["ln_wage"] = np.log(wage_data["hourly_wage"])
    wage_data["ln_exp"] = np.log(wage_data["experience"] + 1)
    wage_data["constant"] = np.ones(len(wage_data))
    # Format & Index
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])

    # Initialize empty containers for coefficients
    index = pd.MultiIndex.from_product(
        [edu_labels, sex_labels, coefficients], names=["education", "sex", "parameter"]
    )
    index_all_types = pd.MultiIndex.from_product(
        [["all"], ["all"], coefficients], names=["education", "sex", "parameter"]
    )
    index = index.append(index_all_types)
    wage_parameters = pd.DataFrame(index=index, columns=["value"])
    year_fixed_effects = {}
    years = list(range(specs["start_year"] + 1, specs["end_year"] + 1))

    # Estimate wage equation for each type (sex x education)
    year_fixed_effects["all", "all"] = {}
    fit_panel_reg_model(
        wage_data, regressors, years, wage_parameters, year_fixed_effects, "all", "all"
    )

    for sex_val, sex_label in enumerate(sex_labels):
        fig, ax = plt.subplots()
        for edu_val, edu_label in enumerate(edu_labels):
            wage_data_type = wage_data[
                (wage_data["education"] == edu_val) & (wage_data["sex"] == sex_val)
            ].copy()
            year_fixed_effects[edu_label, sex_label] = {}
            wage_parameters, year_fixed_effects, wage_data_type = fit_panel_reg_model(
                wage_data_type,
                regressors,
                years,
                wage_parameters,
                year_fixed_effects,
                edu_label,
                sex_label,
            )

            wage_data_type = wage_data_type[
                wage_data_type["age"] < specs["max_est_age_labor"]
            ]
            # Plot
            ax.plot(
                wage_data_type.groupby("age")["ln_wage"].mean(),
                color=JET_COLOR_MAP[edu_val],
                ls="--",
                label=f"Obs. {edu_label}",
            )
            ax.plot(
                wage_data_type.groupby("age")["predicted_ln_wage"].mean(),
                color=JET_COLOR_MAP[edu_val],
                label=f"Est. {edu_label}",
            )
        # ax.set_title(sex_label)
        ax.set_xlabel("Age")
        ax.set_ylabel("Log hourly wage")
        ax.legend(loc="upper left")

        fig.savefig(
            BLD / "plots" / "stochastic_processes" / f"wages_{sex_label.lower()}.png"
        )
        plt.close(fig)

    # Save results
    wage_parameters.to_csv(path_to_save_wage_params)
    wage_parameters.T.to_latex(path_to_save_latex, float_format="%.4f")

    # After estimation print some summary statistics
    print_wage_equation(wage_parameters, edu_labels, sex_labels)
    calc_population_averages(
        wage_data,
        year_fixed_effects=year_fixed_effects,
        specs=specs,
        path_pop_avg_annual_wage=path_pop_avg_annual_wage,
        path_pop_avg_working_hours=path_pop_avg_working_hours,
    )


def fit_panel_reg_model(
    wage_data_type,
    regressors,
    years,
    wage_parameters,
    year_fixed_effects,
    edu_label,
    sex_label,
):
    # estimate parametric regression, save parameters
    model = PanelOLS(
        dependent=wage_data_type["ln_wage"],
        exog=wage_data_type[regressors + ["year"]],
        entity_effects=True,
    )
    fitted_model = model.fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    # Add prediction to data
    wage_data_type["predicted_ln_wage"] = fitted_model.predict()

    # Assign estimated parameters (column list corresponds to model params,
    # so only these are assigned)
    for param in regressors:
        wage_parameters.loc[edu_label, sex_label, param] = fitted_model.params[param]
        wage_parameters.loc[edu_label, sex_label, param + "_ser"] = (
            fitted_model.std_errors[param]
        )
    for year in years:
        year_fixed_effects[(edu_label, sex_label)][year] = fitted_model.params[
            f"year.{year}"
        ]

    # Get estimate for income shock std
    (
        wage_parameters.loc[edu_label, sex_label, "income_shock_std"],
        wage_parameters.loc[edu_label, sex_label, "income_shock_std_ser"],
    ) = est_shock_std(
        residuals=fitted_model.resids,
        n_obs=wage_data_type.shape[0],
        n_params=fitted_model.params.shape[0],
    )

    return wage_parameters, year_fixed_effects, wage_data_type


def calc_population_averages(
    df, year_fixed_effects, specs, path_pop_avg_annual_wage, path_pop_avg_working_hours
):
    """Save population average of annual wage (for pension calculation) and working
    hours by education (to compute annual wages).

    We do this here (as opposed to model specs) to avoid loading the data twice.

    """
    years = list(range(specs["start_year"] + 1, specs["end_year"] + 1))
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]

    # annual average wage (deflated by type-specific year fixed effects)
    df["ln_wage_deflated"] = df["ln_wage"].copy()
    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            for year in years:
                edu_mask = df["education"] == edu_val
                sex_mask = df["sex"] == sex_val
                year_mask = df["year"] = year
                df.loc[
                    edu_mask & sex_mask & year_mask, "ln_wage_deflated"
                ] -= year_fixed_effects[(edu_label, sex_label)][year]

    df["annual_hours"] = df["monthly_hours"] * 12
    df["annual_wage_deflated"] = np.exp(df["ln_wage_deflated"]) * df["annual_hours"]
    pop_avg_annual_wage = df["annual_wage_deflated"].mean()
    np.save(path_pop_avg_annual_wage, pop_avg_annual_wage)

    print("Population average for annual wage (deflated): " + str(pop_avg_annual_wage))

    # averageannual working hours by type
    avg_hours_by_type_choice = df.groupby(["education", "sex", "choice"])[
        "annual_hours"
    ].mean()
    avg_hours_by_type_choice.to_csv(path_pop_avg_working_hours, index=True)
    print("Population averages for working hours: \n")
    print(avg_hours_by_type_choice)


def est_shock_std(residuals, n_obs, n_params):
    """Estimate income shock std and its standard error."""
    rss = residuals @ residuals
    n_minus_k = n_obs - n_params
    income_shock_var = rss / n_minus_k
    income_shock_std = np.sqrt(income_shock_var)
    income_shock_std_ser = np.sqrt((2 * income_shock_var**2) / n_minus_k)
    return income_shock_std, income_shock_std_ser


def print_wage_equation(wage_parameters, edu_labels, sex_labels):
    # print wage equation
    for _edu_val, edu_label in enumerate(edu_labels):
        for _sex_val, sex_label in enumerate(sex_labels):
            print("Hourly wage equation: " + edu_label + " " + sex_label)
            print(
                "ln(hrly_wage) = "
                + str(wage_parameters.loc[(edu_label, sex_label, "constant"), "value"])
                + " + "
                + str(wage_parameters.loc[(edu_label, sex_label, "ln_exp"), "value"])
                + " * ln(exp+1) + epsilon"
            )
            hrly_wage_with_20_exp = np.exp(
                wage_parameters.loc[(edu_label, sex_label, "constant"), "value"]
                + wage_parameters.loc[(edu_label, sex_label, "ln_exp"), "value"]
                * np.log(20)
            )
            print(
                "Example: hourly wage with 20 years of experience: "
                + str(hrly_wage_with_20_exp)
            )
            print(
                "Income shock std: "
                + str(
                    wage_parameters.loc[
                        (edu_label, sex_label, "income_shock_std"), "value"
                    ]
                )
            )
            print("--------------------")
