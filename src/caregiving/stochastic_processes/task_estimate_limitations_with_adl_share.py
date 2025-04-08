"""Estimate probabilities of limitations with Activities of Daily Living.

On SHARE parent-child sample.

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import FEMALE, MALE


def task_estimate_limitations_with_adl_categories(
    path_to_parent_child_sample: Path = BLD / "data" / "parent_child_data.csv",
    path_to_health_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_good_medium_bad.csv",
    path_to_health_death_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_death_transition_matrix_good_medium_bad.csv",
    path_to_save_adl_probabilities: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_params.csv",
):

    df = pd.read_csv(path_to_parent_child_sample)

    model_men, model_women = run_multinomial_by_gender(df)
    results_dict = get_nested_params(model_men, model_women)

    df_men = pivot_model_params(model_men, "Men")
    df_women = pivot_model_params(model_women, "Women")
    df_combined = pd.concat([df_men, df_women], ignore_index=True)

    df_combined.to_csv(path_to_save_adl_probabilities, index=False)

    # plot_health_probabilities(df_combined)

    # health_transition_matrix = pd.read_csv(path_to_health_transition_matrix)
    health_death_transition_matrix = pd.read_csv(path_to_health_death_transition_matrix)
    # plot_weighted_adl_probabilities(df_combined, health_transition_matrix)

    # plot_weighted_adl_death_probabilities(df_combined, health_death_transition_matrix)
    plot_health_state_evolution(health_death_transition_matrix)

    # breakpoint()
    # Check identity
    test_params_equality(results_dict, df_combined, tol=1e-10)


def run_multinomial_by_gender(df):
    """
    Run separate multinomial logit regressions of 'adl_cat' on
    age, age^2, and health, by gender. We assume:
       - gender == 1 => men
       - gender == 2 => women
    The 'health' variable is treated as categorical (with levels 0,1,2).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
        ['adl_cat', 'age', 'health', 'gender'].

    Returns
    -------
    (model_men, model_women)
        Where each is a fitted multinomial logit model.
    """
    # Subset data for men and women
    dat_men = df[df["gender"] == MALE].copy()
    dat_women = df[df["gender"] == FEMALE].copy()

    # Define the formula
    formula = "adl_cat ~ age + I(age**2) + C(health)"

    # Fit the model for men
    model_men = smf.mnlogit(formula, data=dat_men).fit()
    print("Results for men (gender == 1):")
    print(model_men.summary())

    # Fit the model for women
    model_women = smf.mnlogit(formula, data=dat_women).fit()
    print("\nResults for women (gender == 2):")
    print(model_women.summary())

    return model_men, model_women


def get_nested_params(model_men, model_women):
    """Convert the model.params into a nested dictionary.

    {
      "men": {
        "category_1": {
           "intercept": <coef>,
           "medium_health": <coef>,
           "bad_health": <coef>,
           "age": <coef>,
           "age_squared": <coef>
        },
        "category_2": {...},
        "category_3": {...}
      },
      "women": {
        "category_1": {...},
        "category_2": {...},
        "category_3": {...}
      }
    }

    Notes:
      - We map numeric categories (0,1,2,...) to "category_{X+1}".
      - We rename row-index keys:
         "Intercept"          -> "intercept"
         "C(health)[T.1.0]"   -> "medium_health"
         "C(health)[T.2.0]"   -> "bad_health"
         "age"                -> "age"
         "I(age ** 2)"        -> "age_squared"
    """

    # Helper dictionary to rename row-index keys
    rename_map = {
        "Intercept": "const",
        "age": "age",
        "I(age ** 2)": "age_sq",
        "C(health)[T.1.0]": "medium_health",
        "C(health)[T.2.0]": "bad_health",
    }

    def process_model_params(model_params: pd.DataFrame):
        """Convert model_params (a DataFrame) to a nested dict.

        {
          "category_1": {var_name: coefficient, ...},
          "category_2": {...},
          "category_3": {...},
          ...
        }
        """
        outer_dict = {}

        # model_params.columns are the categories (often 0,1,2,...)
        for cat_col in model_params.columns:
            # Create the new category label
            cat_name = f"category_{int(cat_col) + 1}"

            cat_dict = {}
            for var_name in model_params.index:
                # Rename the variable
                new_var_name = rename_map.get(var_name, var_name)
                # Extract coefficient
                coef_value = model_params.loc[var_name, cat_col]
                cat_dict[new_var_name] = coef_value

            outer_dict[cat_name] = cat_dict

        return outer_dict

    men_nested = process_model_params(model_men.params)
    women_nested = process_model_params(model_women.params)

    return {"men": men_nested, "women": women_nested}


def pivot_model_params(model, sex_label):
    """
    Given a fitted mnlogit model (model.params) and a label for sex ("Men"/"Women"),
    return a WIDE DataFrame with columns:
       sex, adl_cat, const, age, age_sq, medium_health, bad_health
    where each row corresponds to one ADL category.
    """

    # 1) Copy the param matrix: shape [param_names x categories], e.g.:
    #    columns = [0,1,2], index = ["Intercept","age","C(health)[T.1.0]", ...]
    df = model.params.copy()

    # 2) Transpose so that categories become the row index
    #    Now shape: [categories x param_names].
    df = df.T

    # 3) Rename the columns so they match the final desired column names
    rename_dict = {
        "Intercept": "const",
        "age": "age",
        "I(age ** 2)": "age_sq",
        "C(health)[T.1.0]": "medium_health",
        "C(health)[T.2.0]": "bad_health",
    }
    df.rename(columns=rename_dict, inplace=True)

    # 4) The row index is the numeric ADL category from the model (e.g. 0,1,2,...).
    #    We want to rename 0->1, 1->2, 2->3, etc.
    df.index = df.index + 1
    df.index.name = "adl_cat"

    # 5) Turn the row index into a regular column
    df.reset_index(inplace=True)  # Now "adl_cat" is a column

    # 6) Add the sex column at the front
    df.insert(0, "sex", sex_label)

    # 7) Make sure we have the final columns in the EXACT order desired
    desired_cols = [
        "sex",
        "adl_cat",
        "const",
        "age",
        "age_sq",
        "medium_health",
        "bad_health",
    ]

    # Ensure any missing columns are added (as NaN) in case the model didn't have them
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only the desired columns, in the exact order
    df = df[desired_cols]

    return df


# =====================================================================================
# Plotting
# =====================================================================================


def plot_health_probabilities(params):
    """
    Plots predicted probabilities for three care degree outcomes (ADL categories)
    against age, separately by gender and fixed subjective health levels. The grid
    is arranged with two rows (one per gender) and three columns (for fixed health:
    Good, Medium, Bad). In each subplot, the predicted probabilities for the three
    ADL outcomes are plotted as a function of age.

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame with estimated logit parameters. Expected columns: "sex",
        "adl_cat", "const", "age", "age_sq", "medium_health", "bad_health". The
        adl_cat indicates the care degree outcome: 1: Care Degree 2, 2: Care Degree 3,
        3: Care Degree 4 or 5. The health effect is captured using dummy variables
        with Good as the reference (fixed health value 2, Medium as 1, Bad as 0).
    """

    # Define a range of ages over which to compute probabilities.
    age_vals = np.linspace(66, 101, 300)  # Change as needed.

    # Outcome labels corresponding to adl_cat values.
    outcome_labels = {1: "Care Degree 2", 2: "Care Degree 3", 3: "Care Degree 4 or 5"}

    # Fixed subjective health mapping (fixed value: label).
    fixed_health_levels = {2: "Good", 1: "Medium", 0: "Bad"}

    # Get unique genders from the parameters DataFrame.
    genders = params["sex"].unique()

    # Create a 2 x 3 grid of subplots (rows: genders, columns: fixed health).
    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(18, 10), sharex=True, sharey=True
    )

    for i, gender in enumerate(genders):
        df_gender = params[params["sex"] == gender]
        df_gender = df_gender.sort_values(by="adl_cat")
        for j, (fixed_val, health_label) in enumerate(
            sorted(fixed_health_levels.items(), reverse=True)
        ):
            utilities = []
            for _, row in df_gender.iterrows():
                if row["adl_cat"] == 1:
                    health_effect = 0.0
                elif row["adl_cat"] == 2:  # noqa: PLR2004
                    health_effect = row["medium_health"] if fixed_val == 1 else 0.0
                elif row["adl_cat"] == 3:  # noqa: PLR2004
                    health_effect = row["bad_health"] if fixed_val == 0 else 0.0
                else:
                    health_effect = 0.0
                util = (
                    row["const"]
                    + row["age"] * age_vals
                    + row["age_sq"] * (age_vals**2)
                    + health_effect
                )
                utilities.append(util)
            utilities = np.array(utilities)
            exp_util = np.exp(utilities)
            prob = exp_util / np.sum(exp_util, axis=0)
            ax = axes[i, j]
            for k, label in outcome_labels.items():
                ax.plot(age_vals, prob[k - 1, :], label=label)
            ax.set_title(f"{gender} - Health: {health_label}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Probability")
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_weighted_adl_death_probabilities(  # noqa: PLR0912, PLR0915
    params, health_death_trans_mat_df
):
    """
    Computes unconditional ADL probabilities by evolving state distributions from
    age 66 onward. At age 66 the state distribution is set by:
      Good Health   = 0.75 * 0.188007,
      Medium Health = 0.75 * 0.743285,
      Bad Health    = 0.75 * 0.068707,
      Death         = 1 - 0.75.
    For ages 67+, gender and period-specific transition probabilities update the state.
    Unconditional ADL probabilities are computed as a weighted average of conditional
    probabilities using the living state proportions.
    """

    # Define the age range (66 to 99).
    age_vals = np.arange(66, 100)

    # Outcome labels for the three ADL outcomes.
    outcome_labels = {1: "Care Degree 2", 2: "Care Degree 3", 3: "Care Degree 4 or 5"}

    # Fixed subjective health mapping.
    health_mapping = {"Good Health": 2, "Medium Health": 1, "Bad Health": 0}

    # Define the complete set of states.
    states = ["Good Health", "Medium Health", "Bad Health", "Death"]

    # --- Initialization at age 66 ---
    init_alive = 0.75
    init_good = 0.188007
    init_medium = 0.743285
    init_bad = 0.068707

    S0 = np.array(
        [
            init_alive * init_good,
            init_alive * init_medium,
            init_alive * init_bad,
            1 - init_alive,
        ]
    )

    genders = params["sex"].unique()
    fig, axes = plt.subplots(
        1, len(genders), figsize=(7 * len(genders), 5), sharey=True
    )
    if len(genders) == 1:
        axes = [axes]

    for idx, gender in enumerate(genders):
        ax = axes[idx]
        df_gender = params[params["sex"] == gender]
        df_gender = df_gender.sort_values(by="adl_cat")
        if df_gender.shape[0] != 3:  # noqa: PLR2004
            raise ValueError(
                f"Expected 3 rows for gender {gender}, got " f"{df_gender.shape[0]}"
            )
        unconditional_adl_list = []
        S = S0.copy()
        for age in age_vals:
            if age > 66:  # noqa: PLR2004
                period_val = int(round(age)) - 30
                sub_ht = health_death_trans_mat_df[
                    (health_death_trans_mat_df["sex"] == gender)
                    & (health_death_trans_mat_df["period"] == period_val)
                ]
                T = sub_ht.pivot(
                    index="health", columns="lead_health", values="transition_prob"
                )
                T = T.reindex(index=states, columns=states, fill_value=0)
                T.loc["Death"] = [0, 0, 0, 1]
                S = S.dot(T.values)
            survival_prob = S[0] + S[1] + S[2]
            if survival_prob > 0:
                w_good = S[0] / survival_prob
                w_medium = S[1] / survival_prob
                w_bad = S[2] / survival_prob
            else:
                w_good = w_medium = w_bad = 1 / 3.0
            cond_adl = {}
            for health_str, fixed_val in health_mapping.items():
                utilities = []
                for _, row in df_gender.iterrows():
                    if row["adl_cat"] == 1:
                        health_effect = 0.0
                    elif row["adl_cat"] == 2:  # noqa: PLR2004
                        health_effect = row["medium_health"] if fixed_val == 1 else 0.0
                    elif row["adl_cat"] == 3:  # noqa: PLR2004
                        health_effect = row["bad_health"] if fixed_val == 0 else 0.0
                    else:
                        health_effect = 0.0
                    u = (
                        row["const"]
                        + row["age"] * age
                        + row["age_sq"] * (age**2)
                        + health_effect
                    )
                    utilities.append(u)
                utilities = np.array(utilities)
                exp_util = np.exp(utilities)
                cond_prob = exp_util / np.sum(exp_util)
                cond_adl[health_str] = cond_prob
            uncond_adl = survival_prob * (
                w_good * cond_adl["Good Health"]
                + w_medium * cond_adl["Medium Health"]
                + w_bad * cond_adl["Bad Health"]
            )
            unconditional_adl_list.append(uncond_adl)
        unconditional_adl_arr = np.array(unconditional_adl_list).T
        for adl_cat, label in outcome_labels.items():
            ax.plot(age_vals, unconditional_adl_arr[adl_cat - 1, :], label=label)
        ax.set_title(str(gender))
        ax.set_xlabel("Age")
        ax.set_ylabel("Uncond. ADL Prob.")
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_health_state_evolution(health_death_trans_mat_df):
    """
    Plots the evolution of state distributions across ages. Starts at age 66 with:
      Good Health   = 0.75 * 0.188007,
      Medium Health = 0.75 * 0.743285,
      Bad Health    = 0.75 * 0.068707,
      Death         = 1 - 0.75.
    For ages 67+ the state is updated using gender and period-specific transitions,
    with Death as an absorbing state.
    """

    age_vals = np.arange(66, 100)
    states = ["Good Health", "Medium Health", "Bad Health", "Death"]

    init_alive = 0.75
    init_good = 0.188007
    init_medium = 0.743285
    init_bad = 0.068707

    S0 = np.array(
        [
            init_alive * init_good,
            init_alive * init_medium,
            init_alive * init_bad,
            1 - init_alive,
        ]
    )

    genders = health_death_trans_mat_df["sex"].unique()
    fig, axes = plt.subplots(
        1, len(genders), figsize=(7 * len(genders), 5), sharey=True
    )
    if len(genders) == 1:
        axes = [axes]

    for idx, gender in enumerate(genders):
        ax = axes[idx]
        state_history = []
        S = S0.copy()
        for age in age_vals:
            if age > 66:  # noqa: PLR2004
                period_val = int(round(age)) - 30
                sub_ht = health_death_trans_mat_df[
                    (health_death_trans_mat_df["sex"] == gender)
                    & (health_death_trans_mat_df["period"] == period_val)
                ]
                T = sub_ht.pivot(
                    index="health", columns="lead_health", values="transition_prob"
                )
                T = T.reindex(index=states, columns=states, fill_value=0)
                T.loc["Death"] = [0, 0, 0, 1]
                S = S.dot(T.values)
            state_history.append(S.copy())
        state_history = np.array(state_history)
        for i, state in enumerate(states):
            ax.plot(age_vals, state_history[:, i], label=state)
        ax.set_title(f"Gender: {gender}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Probability")
        ax.legend()

    plt.tight_layout()
    plt.show()


# =====================================================================================
# Testing
# =====================================================================================


def test_params_equality(results_dict, df, tol=1e-10):
    """
    Check that each row of the wide DataFrame (with columns
    ["sex","adl_cat","const","age","age_sq","medium_health","bad_health"])
    matches the corresponding dictionary entry in results_dict.

    Parameters
    ----------
    results_dict : dict
        Nested dict of the form:
        {
          "men": {
            "category_1": {"const":float, "age":float, "age_sq":float,
                "medium_health":float, "bad_health":float},
            "category_2": {...},
            ...
          },
          "women": {
            "category_1": {...},
            ...
          }
        }

    df : pandas.DataFrame
        The wide DataFrame read from CSV with columns:
        ["sex","adl_cat","const","age","age_sq","medium_health","bad_health"].

    tol : float
        Numerical tolerance for comparing floats (uses np.isclose with atol=tol).

    Raises
    ------
    AssertionError
        If any entry does not match between the DataFrame and the dictionary
        within the numerical tolerance.

    Returns
    -------
    None
    """
    required_cols = [
        "sex",
        "adl_cat",
        "const",
        "age",
        "age_sq",
        "medium_health",
        "bad_health",
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Go row by row in the DataFrame
    for i, row in df.iterrows():
        # Convert "Men" -> "men", "Women" -> "women" so we can index results_dict
        sex_key = row["sex"].lower()  # 'men' or 'women'
        cat_key = f"category_{int(row['adl_cat'])}"

        # Retrieve the dict entry for that sex and category
        dict_values = results_dict[sex_key][cat_key]

        # For each coefficient name in the dictionary, compare numeric values
        for var in ("const", "age", "age_sq", "medium_health", "bad_health"):
            val_df = row[var]
            val_dict = dict_values[var]

            # Use np.isclose for float comparisons
            if not np.isclose(val_df, val_dict, atol=tol, rtol=0):
                msg = (
                    f"Mismatch at DataFrame row {i}: "
                    f"sex={row['sex']}, adl_cat={row['adl_cat']}, variable={var}\n"
                    f"  DataFrame value = {val_df}\n"
                    f"  Dictionary value = {val_dict}\n"
                )
                raise AssertionError(msg)

    print("All checks passed!")
