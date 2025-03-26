"""Estimate the mortality matrix for three health states."""

import itertools
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import optimagic as om
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, JET_COLOR_MAP, SRC
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.stochastic_processes.auxiliary import loglike

import statsmodels.api as sm


# @pytask.mark.skip()
def task_estimate_mortality_for_three_health_states(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_lifetable: Path = SRC
    / "data"
    / "statistical_office"
    / "mortality_table_for_pandas.csv",
    path_to_mortatility_sample: Path = BLD
    / "data"
    / "mortality_transition_estimation_sample_three_states_duplicated.pkl",
    path_to_save_mortality_params_men: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_three_states_men.csv",
    path_to_save_mortality_params_women: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_three_states_women.csv",
    path_to_save_mortality_transition_matrix: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_three_states.csv",
    path_to_save_lifetable: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "lifetable_three_states.csv",
):
    """Estimate the mortality matrix."""

    specs = read_and_derive_specs(path_to_specs)

    # Load life table data and expand/duplicate it to include all
    # possible combinations of health, education and sex
    lifetable_df = pd.read_csv(
        path_to_lifetable,
        sep=";",
    )

    mortality_df = pd.DataFrame(
        [
            {
                "age": row["age"],
                "health": combo[0],
                "education": combo[1],
                "sex": combo[2],
                "death_prob": (
                    row["death_prob_male"]
                    if combo[2] == 0
                    else row["death_prob_female"]
                ),
            }
            for _, row in lifetable_df.iterrows()
            # Now generate all combinations for health=[0,1,2], education=[0,1], sex=[0,1]
            for combo in itertools.product((0, 1, 2), (0, 1), (0, 1))
        ]
    )
    mortality_df.reset_index(drop=True, inplace=True)

    # Plain life table data
    lifetable_df = mortality_df[["age", "sex", "death_prob"]].copy()
    lifetable_df.drop_duplicates(inplace=True)

    # Estimation sample - as in Kroll Lampert 2008 / Haan Schaller et al. 2024
    df = pd.read_pickle(path_to_mortatility_sample)

    # Df initial values i.e. first observations (+ sex column)
    start_df = df[
        [col for col in df.columns if col.startswith("start")] + ["sex"]
    ].copy()
    str_cols = start_df.columns.str.replace("start ", "")
    start_df.columns = str_cols

    # Add a intercept column to the df and start_df
    df["intercept"] = 1

    start_df["intercept"] = 1

    for sex, sex_label in enumerate(specs["sex_labels"]):

        if sex_label.lower() == "men":
            path_to_save_params = path_to_save_mortality_params_men
        else:
            path_to_save_params = path_to_save_mortality_params_women

        # Filter data by sex
        filtered_df = df[df["sex"] == sex]
        filtered_start_df = start_df[start_df["sex"] == sex]

        # -----------------------------------------------------------------
        # 3a. Build dummy columns for the 3 health states × 2 edu states
        #     while dropping one baseline (good=2, low=0).
        # -----------------------------------------------------------------
        # We'll keep an explicit intercept, so we do not create a dummy for that baseline.
        # filtered_df[
        #     f"{specs['health_labels_three'][0]} {specs['education_labels'][0]}"
        # ] = ((filtered_df["health"] == 0) & (filtered_df["education"] == 0)).astype(int)
        filtered_df[
            f"{specs['health_labels_three'][0]} {specs['education_labels'][1]}"
        ] = ((filtered_df["health"] == 0) & (filtered_df["education"] == 1)).astype(int)
        filtered_df[
            f"{specs['health_labels_three'][1]} {specs['education_labels'][0]}"
        ] = ((filtered_df["health"] == 1) & (filtered_df["education"] == 0)).astype(int)
        filtered_df[
            f"{specs['health_labels_three'][1]} {specs['education_labels'][1]}"
        ] = ((filtered_df["health"] == 1) & (filtered_df["education"] == 1)).astype(int)
        filtered_df[
            f"{specs['health_labels_three'][2]} {specs['education_labels'][0]}"
        ] = ((filtered_df["health"] == 2) & (filtered_df["education"] == 0)).astype(int)
        filtered_df[
            f"{specs['health_labels_three'][2]} {specs['education_labels'][1]}"
        ] = ((filtered_df["health"] == 2) & (filtered_df["education"] == 1)).astype(int)

        # Now define your X and y
        exog_cols = [
            "intercept",
            "age",
            f"{specs['health_labels_three'][0]} {specs['education_labels'][1]}",
            f"{specs['health_labels_three'][1]} {specs['education_labels'][0]}",
            f"{specs['health_labels_three'][1]} {specs['education_labels'][1]}",
            f"{specs['health_labels_three'][2]} {specs['education_labels'][0]}",
            f"{specs['health_labels_three'][2]} {specs['education_labels'][1]}",
        ]
        endog = filtered_df["death event"]
        exog = filtered_df[exog_cols]

        # 3b. Fit logistic regression
        model = sm.Logit(endog, exog)
        result = model.fit(disp=True)  # disp=True prints iteration messages

        print(result.summary())
        # You can also get the coefficient table
        params = result.params
        conf_int = result.conf_int()

        # Save param table to CSV
        out_df = pd.DataFrame(
            {
                "param": exog_cols,
                "coef": params.values,
                "odds_ratio": np.exp(params.values),
                "ci_lower": np.exp(conf_int[0].values),
                "ci_upper": np.exp(conf_int[1].values),
            }
        )
        out_df.to_csv(path_to_save_params, index=False)

        # -----------------------------------------------------------------
        # 4. Update mortality_df["death_prob"] using the fitted logistic
        # -----------------------------------------------------------------
        # If you want to apply the *same logistic formula* to the "mortality_df"
        # to get predicted probabilities, do the same dummy expansions:
        for health, health_label in enumerate(specs["health_labels_three"]):
            for education, education_label in enumerate(specs["education_labels"]):
                # Map from the row to the correct dummy combination:
                subset_mask = (
                    (mortality_df["sex"] == sex)
                    & (mortality_df["health"] == health)
                    & (mortality_df["education"] == education)
                )

                # Build that row's X vector
                # (Remember baseline is good=2, low=0)
                X_temp = pd.DataFrame(
                    {
                        "intercept": [1.0],
                        "age": [
                            mortality_df.loc[subset_mask, "age"].mean()
                        ],  # or local row's age
                        "bad_low": [int(health == 0 and education == 0)],
                        "bad_high": [int(health == 0 and education == 1)],
                        "medium_low": [int(health == 1 and education == 0)],
                        "medium_high": [int(health == 1 and education == 1)],
                        "good_high": [int(health == 2 and education == 1)],
                    }
                )

                # Predict logit prob = 1 / (1 + exp(-X*beta))
                # statsmodels has a convenient:
                p_death = result.predict(X_temp)[0]

                # Overwrite or update the "death_prob"
                # Because logistic predictions are unconditional, you might
                # choose to replace the existing "death_prob" rather than multiply:
                mortality_df.loc[subset_mask, "death_prob"] = p_death

        # # Now health_labels has 3 states: e.g. specs["health_labels"] = ["h0", "h1", "h2"]
        # # and education_labels might still have 2 states: e.g. ["edu0", "edu1"]
        # # You need one initial-parameter entry per (health-state, education-state) combination.
        # initial_params_data = {
        #     "intercept": {
        #         "value": -13,
        #         "lower_bound": -np.inf,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": -15.0,
        #         "soft_upper_bound": 15.0,
        #     },
        #     "age": {
        #         "value": 0.1,
        #         "lower_bound": 1e-8,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": 0.0001,
        #         "soft_upper_bound": 1.0,
        #     },
        #     # -------------------------------------------
        #     # Below are the keys for combinations of
        #     # (health = 0,1,2) x (education = 0,1).
        #     # Adjust the "value" fields as you see fit.
        #     # -------------------------------------------
        #     # bad
        #     f"{specs['health_labels_three'][0]} {specs['education_labels'][1]}": {
        #         "value": 0.0,
        #         "lower_bound": -np.inf,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": -2.5,
        #         "soft_upper_bound": 2.5,
        #     },
        #     f"{specs['health_labels_three'][0]} {specs['education_labels'][0]}": {
        #         "value": 0.2,
        #         "lower_bound": -np.inf,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": -2.5,
        #         "soft_upper_bound": 2.5,
        #     },
        #     # medium
        #     f"{specs['health_labels_three'][1]} {specs['education_labels'][1]}": {
        #         "value": -0.4,
        #         "lower_bound": -np.inf,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": -2.5,
        #         "soft_upper_bound": 2.5,
        #     },
        #     f"{specs['health_labels_three'][1]} {specs['education_labels'][0]}": {
        #         "value": -0.3,
        #         "lower_bound": -np.inf,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": -2.5,
        #         "soft_upper_bound": 2.5,
        #     },
        #     # good
        #     f"{specs['health_labels_three'][2]} {specs['education_labels'][1]}": {
        #         "value": 0,
        #         "lower_bound": -np.inf,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": -2.5,
        #         "soft_upper_bound": 2.5,
        #     },
        #     f"{specs['health_labels_three'][2]} {specs['education_labels'][0]}": {
        #         "value": 0,
        #         "lower_bound": -np.inf,
        #         "upper_bound": np.inf,
        #         "soft_lower_bound": -2.5,
        #         "soft_upper_bound": 2.5,
        #     },
        # }

        # initial_params = pd.DataFrame.from_dict(initial_params_data, orient="index")

        # options = {
        #     "stopping_maxiter": 1000,
        # }

        # # Estimate parameters
        # res = om.maximize(
        #     fun=loglike,
        #     params=initial_params,
        #     algorithm="scipy_lbfgsb",
        #     algo_options=options,
        #     fun_kwargs={"data": filtered_df, "start_data": filtered_start_df},
        #     numdiff_options=om.NumdiffOptions(n_cores=4),
        #     multistart=om.MultistartOptions(n_samples=100, seed=0, n_cores=4),
        # )

        # # terminal log the results
        # print(res)
        # print(res.params)

        # # save the results
        # to_csv_summary = res.params.copy()
        # to_csv_summary["hazard_ratio"] = np.exp(to_csv_summary["value"])
        # to_csv_summary.to_csv(path_to_save_params)

        # # update mortality_df with the estimated parameters
        # for health, health_label in enumerate(
        #     specs["health_labels_three"][:-1]
        # ):  # exclude the last health label (death)
        #     for education, education_label in enumerate(specs["education_labels"]):
        #         param = f"{health_label} {education_label}"

        #         # if param not in res.params.index:
        #         #     continue

        #         mortality_df.loc[
        #             (mortality_df["sex"] == sex)
        #             & (mortality_df["health"] == health)
        #             & (mortality_df["education"] == education),
        #             "death_prob",
        #         ] *= np.exp(res.params.loc[param, "value"])

    # export the estimated mortality table and the original life table as csv
    lifetable_df = lifetable_df[
        (lifetable_df["age"] >= specs["start_age_mortality"])
        & (lifetable_df["age"] <= specs["end_age_mortality"])
    ]
    mortality_df = mortality_df[
        (mortality_df["age"] >= specs["start_age_mortality"])
        & (mortality_df["age"] <= specs["end_age_mortality"])
    ]
    mortality_df = mortality_df[["age", "sex", "health", "education", "death_prob"]]
    lifetable_df = lifetable_df[["age", "sex", "death_prob"]]
    mortality_df = mortality_df.astype(
        {
            "age": "int",
            "sex": "int",
            "health": "int",
            "education": "int",
            "death_prob": "float",
        }
    )
    lifetable_df = lifetable_df.astype(
        {
            "age": "int",
            "sex": "int",
            "death_prob": "float",
        }
    )
    mortality_df.to_csv(
        path_to_save_mortality_transition_matrix,
        sep=",",
        index=False,
    )
    lifetable_df.to_csv(
        path_to_save_lifetable,
        sep=",",
        index=False,
    )


# =====================================================================================
# Plotting
# =====================================================================================


# For plotting colors (adjust as needed)
JET_COLOR_MAP = ["blue", "red", "green", "orange", "purple"]


def task_plot_mortality_logit(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_params_men: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_three_states_men.csv",
    path_to_params_women: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_params_three_states_women.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_survival_probabilities_logit_three_states.png",
):
    """
    Plot predicted survival probabilities by age using logistic regression
    estimates (one plot for men, one plot for women).

    The logistic model uses:
      - "intercept" and "age" as continuous predictors, and
      - Five dummy variables corresponding to:
            f"{specs['health_labels_three'][0]} {specs['education_labels'][1]}"
            f"{specs['health_labels_three'][1]} {specs['education_labels'][0]}"
            f"{specs['health_labels_three'][1]} {specs['education_labels'][1]}"
            f"{specs['health_labels_three'][2]} {specs['education_labels'][0]}"
            f"{specs['health_labels_three'][2]} {specs['education_labels'][1]}"
      The dropped baseline is (health==0, education==0).
    """
    # Read specs and parameter CSV files
    specs = read_and_derive_specs(path_to_specs)
    men_params_df = pd.read_csv(path_to_params_men)
    women_params_df = pd.read_csv(path_to_params_women)

    # Create dictionaries for quick lookup: { "param_name": coef }
    men_params = dict(zip(men_params_df["param"], men_params_df["coef"]))
    women_params = dict(zip(women_params_df["param"], women_params_df["coef"]))

    # Define the age range
    start_age = specs["start_age_mortality"]
    end_age = specs["end_age_mortality"]
    ages = np.arange(start_age, end_age + 1)

    # List of health and education states
    # Here health is coded as: 0, 1, 2 and education as: 0, 1.
    health_states = [0, 1, 2]
    edu_states = [0, 1]

    # We'll accumulate rows of predicted probabilities here.
    curve_data = []

    def logistic_prob_death(age, health, edu, param_dict, specs):
        """
        Given an age, health state, and education, compute the predicted
        death probability from the logistic model.

        The linear predictor is:
            xb = intercept + age * coef_age + dummies,
        where the dummies are defined using the new parameter names.
        Note: For (health==0, edu==0) no dummy is added (baseline).
        """
        xb = param_dict.get("intercept", 0.0) + param_dict.get("age", 0.0) * age

        # Check the combination and add the corresponding dummy coefficient.
        if health == 0 and edu == 1:
            key = f"{specs['health_labels_three'][0]} {specs['education_labels'][1]}"
            xb += param_dict.get(key, 0.0)
        elif health == 1 and edu == 0:
            key = f"{specs['health_labels_three'][1]} {specs['education_labels'][0]}"
            xb += param_dict.get(key, 0.0)
        elif health == 1 and edu == 1:
            key = f"{specs['health_labels_three'][1]} {specs['education_labels'][1]}"
            xb += param_dict.get(key, 0.0)
        elif health == 2 and edu == 0:
            key = f"{specs['health_labels_three'][2]} {specs['education_labels'][0]}"
            xb += param_dict.get(key, 0.0)
        elif health == 2 and edu == 1:
            key = f"{specs['health_labels_three'][2]} {specs['education_labels'][1]}"
            xb += param_dict.get(key, 0.0)
        # For (health==0, edu==0) baseline: do nothing.
        p_death = 1.0 / (1.0 + np.exp(-xb))
        return p_death

    # Build survival curves for both sexes.
    for sex_var, param_dict in enumerate([men_params, women_params]):
        for h in health_states:
            for edu in edu_states:
                # Initialize cumulative survival probability at 1.
                cum_surv = 1.0
                for age in ages:
                    p_death = logistic_prob_death(age, h, edu, param_dict, specs)
                    # Save the survival probability *before* this year's death event.
                    curve_data.append(
                        {
                            "sex": sex_var,
                            "age": age,
                            "health": h,
                            "education": edu,
                            "p_death": p_death,
                            "p_surv": cum_surv,
                        }
                    )
                    # Update survival for next year.
                    cum_surv *= 1.0 - p_death

    curve_df = pd.DataFrame(curve_data)

    # Plotting: Create one subplot for men and one for women.
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)

    # For labeling, use the specs for health and education labels.
    health_labels = specs["health_labels_three"]
    education_labels = specs["education_labels"]

    for sex_var, ax in enumerate(axes):
        sex_label = specs["sex_labels"][sex_var]
        for edu in edu_states:
            for h in health_states:
                sub = curve_df[
                    (curve_df["sex"] == sex_var)
                    & (curve_df["health"] == h)
                    & (curve_df["education"] == edu)
                ].copy()
                # Use different linestyles to distinguish health states.
                if h == 0:
                    linestyle = "--"
                elif h == 1:
                    linestyle = "-."
                else:
                    linestyle = "-"

                # Build a label combining education and health.
                lbl = f"{education_labels[edu]}, {health_labels[h]}"
                ax.plot(
                    sub["age"],
                    sub["p_surv"],
                    linestyle=linestyle,
                    color=JET_COLOR_MAP[edu % len(JET_COLOR_MAP)],
                    label=lbl,
                )
        ax.set_title(f"Survival Probability for {sex_label}")
        ax.set_xlabel("Age")
        ax.set_xlim([start_age, end_age])
        ax.set_ylabel("Cumulative Survival Probability")
        ax.set_ylim([0, 1.05])
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path_to_save_plot)
    plt.close(fig)


@pytask.mark.skip()
def task_plot_mortality(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_mortality_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_three_states.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "estimated_survival_probabilities_three_states.png",
):
    """Plot mortality characteristics."""

    specs = read_and_derive_specs(path_to_specs)

    # Load the data
    estimated_mortality = pd.read_csv(path_to_mortality_transition_matrix)

    _n_edu_types = len(specs["education_labels"])
    _n_ages = specs["end_age_mortality"] - specs["start_age_mortality"] + 1
    _n_alive_health_states = len(specs["health_labels"]) - 1
    alive_health_states = np.where(np.array(specs["health_labels_three"]) != "Death")[0]
    _health_states = [specs["health_labels"][i] for i in alive_health_states]
    _education_states = specs["education_labels"]
    _age_range = np.arange(specs["start_age_mortality"], specs["end_age_mortality"] + 1)

    estimated_mortality["survival_prob_year"] = np.nan
    estimated_mortality["survival_prob"] = np.nan

    for sex_var, _sex_label in enumerate(specs["sex_labels"]):
        for health in alive_health_states:
            for edu_var, _edu_label in enumerate(specs["education_labels"]):
                mask = (
                    (estimated_mortality["sex"] == sex_var)
                    & (estimated_mortality["health"] == health)
                    & (estimated_mortality["education"] == edu_var)
                )

                filtered_data = estimated_mortality.loc[
                    mask,
                    ["death_prob", "age"],
                ].sort_values(by="age")

                filtered_data["survival_prob_year"] = 1 - filtered_data["death_prob"]
                filtered_data["survival_prob"] = filtered_data[
                    "survival_prob_year"
                ].cumprod()
                filtered_data["survival_prob"] = filtered_data["survival_prob"].shift(1)
                filtered_data.loc[0, "survival_prob"] = 1

                estimated_mortality.update(filtered_data)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 8))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axes[sex_var]

        for edu_var, edu_label in enumerate(specs["education_labels"]):
            for health in alive_health_states:
                mask = (
                    (estimated_mortality["sex"] == sex_var)
                    & (estimated_mortality["health"] == health)
                    & (estimated_mortality["education"] == edu_var)
                )
                health_label = specs["health_labels_three"][health]

                if health == 0:
                    linestyle = "--"
                elif health == 1:
                    linestyle = "-."
                else:
                    linestyle = "-"

                ax.plot(
                    estimated_mortality.loc[mask, "age"],
                    estimated_mortality.loc[mask, "survival_prob"],
                    color=JET_COLOR_MAP[edu_var],
                    label=f"{edu_label}; {health_label}",
                    linestyle=linestyle,
                )

        ax.set_xlabel("Age")
        ax.set_xlim(specs["start_age"], specs["end_age"] + 1)
        ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"Estimated Survival Probability for {sex_label}")

    axes[0].legend(loc="lower left")
    fig.savefig(path_to_save_plot)


def survival_function(age, health_factors, params):
    """
    Calculates the survival function: Exp(-(integral of the hazard function
    as a function of age from 0 to age)).

    Parameters:
        age (float or array-like): The age(s) at which to calculate the function.
        health_factors (dict): Keys are health-education categories
            (e.g., 'health1_edu1') and values are their indicator variables (0 or 1).
        params (DataFrame): Model parameters, with 'intercept' and coefficient names
                            in the index and their values in a column named 'value'.

    Returns:

        float or array-like: The value(s) of the survival function.
    """
    coefficients = params["value"]
    intercept = coefficients["intercept"]
    age_coef = coefficients["age"]

    lambda_ = np.exp(
        intercept
        + sum(
            coefficients[key] * value
            for key, value in health_factors.items()
            if key != "intercept"
        )
    )

    age_contrib = np.exp(age_coef * age) - 1

    return np.exp(-lambda_ / age_coef * age_contrib)
