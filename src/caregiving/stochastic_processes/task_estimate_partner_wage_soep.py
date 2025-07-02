"""Estimate the parameters of the hourly wage equation using SOEP panel data.

The script estimates the following equation for each education level:

    ln_partner_wage = beta_0 + beta_1 * ln(age) + individual_FE + time_FE + epsilon

where:
- ln_partner_wage: Natural logarithm of the partner's hourly wage
- ln(age): Natural logarithm of age
- individual_FE: Individual fixed effects
- time_FE: Time fixed effects
- epsilon: Error term

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel.model import PanelOLS
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


def task_estimate_partner_wage_parameters(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_partner_wage_data.csv",
    path_to_save_plot_partner_men: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "wages_partner_men.png",
    path_to_save_plot_partner_women: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "wages_partner_women.png",
    path_to_partner_wage_men: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "partner_wage_eq_params_men.csv",
    path_to_partner_wage_women: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "partner_wage_eq_params_women.csv",
) -> None:
    """Estimate the wage parameters partners by education group in the sample.

    Separately for men and women.

    """
    specs = read_and_derive_specs(path_to_specs)

    edu_labels = specs["education_labels"]
    model_params = ["constant", "period", "period_sq"]

    wage_data = pd.read_csv(path_to_data, index_col=0)

    for sex_var, sex_label in enumerate(specs["sex_labels"]):

        wage_data_sex = prepare_estimation_data(wage_data, specs, sex_var=sex_var)

        # Initialize empty container for coefficients
        wage_parameters = pd.DataFrame(
            index=pd.Index(data=edu_labels, name="education"),
            columns=model_params,
        )

        fig, ax = plt.subplots()
        for edu_val, edu_label in enumerate(edu_labels):

            # Filter df
            wage_data_edu = wage_data_sex[wage_data_sex["education"] == edu_val].copy()
            wage_data_edu = sm.add_constant(wage_data_edu)

            model = sm.OLS(
                endog=wage_data_edu["wage_p"],
                exog=sm.add_constant(
                    wage_data_edu[["constant", "period", "period_sq"]]
                ),
                missing="drop",
            )
            fitted_model = model.fit()

            # Assign prediction
            wage_data_edu["wage_pred"] = fitted_model.predict()

            # Plot wage and prediction
            ax.plot(
                wage_data_edu.groupby("age")["wage_p"].mean(),
                label=f"Obs. {edu_label}",
                linestyle="--",
                color=JET_COLOR_MAP[edu_val],
            )
            ax.plot(
                wage_data_edu.groupby("age")["wage_pred"].mean(),
                label=f"Est. {edu_label}",
                color=JET_COLOR_MAP[edu_val],
            )

            # Assign estimated parameters (column list corresponds to model params,
            # so only these are assigned)
            wage_parameters.loc[edu_label] = fitted_model.params

        ax.legend()
        # ax.set_title(f"Partner Wages of {sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Monthly Wage")

        fig.savefig(
            BLD
            / "plots"
            / "stochastic_processes"
            / f"wages_partner_{sex_label.lower()}.png"
        )

        wage_parameters.to_csv(
            BLD
            / "estimation"
            / "stochastic_processes"
            / f"partner_wage_eq_params_{sex_label.lower()}.csv"
        )

        plt.close(fig)


def prepare_estimation_data(wage_data, specs, sex_var):
    """Prepare the data for the wage estimation."""
    wage_data = wage_data[wage_data["sex"] == sex_var].copy()

    # Add period
    wage_data["period"] = wage_data["age"] - specs["start_age"]
    wage_data["period_sq"] = wage_data["period"] ** 2

    # We only want to look at working age people
    wage_data = wage_data[wage_data["age"] < specs["max_ret_age"]]

    # partner data
    # wage_data["ln_partner_hrl_wage"] = np.log(wage_data["hourly_wage_p"])

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))

    return wage_data


# def task_calculate_partner_hours(
#     path_to_specs: Path = SRC / "specs.yaml",
#     path_to_data: Path = BLD / "data" / "soep_partner_wage_data.csv",
#     path_to_partner_hours: Annotated[Path, Product] = BLD
#     / "estimation"
#     / "stochastic_processes"
#     / "partner_hours.csv",
# ) -> None:
#     """Calculates average hours worked by working partners.

#     I.e. conditional on working hours > 0.

#     """
#     specs = read_and_derive_specs(path_to_specs)

#     start_age = specs["start_age"]
#     end_age = specs["max_ret_age"]

#     # Load data, filter, and create age bins
#     df = pd.read_csv(path_to_data, index_col=0)
#     breakpoint()

#     df = df[df["age"] >= start_age]
#     df = df[df["age"] <= end_age]
#     df["age_bin"] = np.floor(df["age"] / 10) * 10
#     df.loc[df["age"] > 60, "age_bin"] = 60
#     df["period"] = df["age"] - start_age

#     # Calculate average hours worked by partner by age, sex and education
#     cov_list = ["sex", "education", "age_bin"]
#     partner_hours = df.groupby(cov_list)["working_hours_p"].mean()

#     partner_hours.to_csv(path_to_partner_hours)
