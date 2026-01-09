"""Estimate probabilities of limitations with Activities of Daily Living.

On SHARE parent-child sample.

"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pytask import Product

from caregiving.config import BLD


def table(df_col):
    return pd.crosstab(df_col, columns="Count")["Count"]


def task_estimate_parental_health_transitions(
    path_to_raw_data: Path = BLD / "data" / "estimation_data.csv",
    path_to_save_parental_health_transitions: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "parental_health_params.csv",
):

    df = pd.read_csv(path_to_raw_data)

    model_mother = run_multinomial_logit(df, outcome="health", parent="mother")
    model_father = run_multinomial_logit(df, outcome="health", parent="father")

    df_mother = pivot_model_params_parent(
        model_mother, outcome="health", parent="mother"
    )
    df_father = pivot_model_params_parent(
        model_father, outcome="health", parent="father"
    )

    # Plot the results
    coefs_mother = create_nested_dict_from_params(df_mother, outcome="health")
    coefs_father = create_nested_dict_from_params(df_father, outcome="health")

    test_probabilities_sum_to_one(coefs_mother, coefs_father)

    # plot_probabilities_by_health(coefs_mother, coefs_father, ages=np.arange(60, 101))

    # Combine into one DataFrame
    df_combined = pd.concat([df_mother, df_father], ignore_index=True)

    df_combined.to_csv(path_to_save_parental_health_transitions, index=False)


# ====================================================================================
# Auxiliary functions
# ====================================================================================


def run_multinomial_logit(df, outcome, parent):
    """Estimate Markov transition probabilities for parental health."""

    formula = (
        f"{parent}_{outcome} ~ {parent}_age + I({parent}_age**2) + "
        f"C({parent}_lagged_health)"
    )

    # Fit the model
    model_parent = smf.mnlogit(formula, data=df).fit()
    print(f"Results for {parent}")
    print(model_parent.summary())

    return model_parent


def pivot_model_params_parent(model, outcome, parent):
    """
    Given a fitted mnlogit model (model.params) and a label for the parent
    (e.g., "Mother"/"Father"), return a WIDE DataFrame with columns:

       parent, health_cat, const, age, age_sq, medium_health, bad_health

    where each row corresponds to one outcome category in the multinomial model.
    """

    # 1) Copy the param matrix: shape [param_names x categories]
    params = model.params.copy()

    # 2) If it's a Series (binary logit case), convert to a single-row DataFrame
    if isinstance(params, pd.Series):
        # Binary logit -> create a 1×N DataFrame
        # shape: (1, n_params) with columns = param names
        df = pd.DataFrame([params.values], columns=params.index)
    else:
        df = params.T

    # 3) Rename the columns so they match the final desired column names.
    #    Adjust these mappings to match *your* exact variable names from the model.
    rename_dict = {
        "Intercept": "const",
        f"{parent}_age": "age",
        f"I({parent}_age ** 2)": "age_sq",
        f"C({parent}_lagged_health)[T.1.0]": "medium_health",
        f"C({parent}_lagged_health)[T.2.0]": "bad_health",
    }
    df.rename(columns=rename_dict, inplace=True)

    # 4) The row index is the numeric health category from the model (e.g., 0,1,2).
    #    We can rename 0->1, 1->2, etc. by adding +1 to the index.
    df.index = df.index + 1
    df.index.name = outcome

    # 5) Turn the row index into a regular column
    df.reset_index(inplace=True)

    # 6) Add the parent column at the front
    df.insert(0, "parent", parent.capitalize())

    # 7) Make sure we have the final columns in the EXACT order desired
    desired_cols = [
        "parent",
        outcome,
        "const",
        "age",
        "age_sq",
        "medium_health",
        "bad_health",
    ]

    # If any columns are missing (e.g., if the model omitted some terms),
    # add them as NaN
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only the desired columns, in the exact order
    df = df[desired_cols]

    return df


# ====================================================================================
# Plotting
# ====================================================================================


# def plot_probabilities_alive_by_health(coeffs_mother, coeffs_father, ages):
#     """
#     Creates a single plot showing Probability(Alive=1) vs. Age
#     for 3 health conditions (good, medium, bad) and 2 genders (Mother, Father).
#     That means 6 lines total in one figure.

#     Parameters
#     ----------
#     coeffs_mother : dict
#         Example format (binary logit => 1 category):
#             {
#               "category_1": {
#                 "intercept": ...,
#                 "age": ...,
#                 "age_squared": ...,
#                 "medium_health": ...,
#                 "bad_health": ...
#               }
#             }
#     coeffs_father : dict
#         Same structure as coeffs_mother but for father.
#     ages : array-like
#         Sequence of ages for plotting, e.g. np.arange(60, 101).
#     """

#     # Three possible health conditions
#     health_conditions = ["good_health", "medium_health", "bad_health"]

#     # Initialize a single figure and axis
#     fig, ax = plt.subplots(figsize=(10, 6))
#     fig.suptitle("Probability of Being Alive by Age\n(3 Health Conditions × Genders)")

#     # We'll store line labels and styles for clarity
#     line_styles = {
#         "Mother": "--",
#         "Father": "-",
#     }

#     for health_cond in health_conditions:
#         # -- MOTHER
#         # Only 'category_1' in binary case
#         mother_params = coeffs_mother["category_1"]
#         # If the health condition key exists, add that to the logit
#         mother_extra = mother_params.get(health_cond, 0.0)

#         # Logit for mother = intercept + age*beta + age^2*beta2 + possible extra
#         logit_mother = (
#             mother_params["intercept"]
#             + mother_params["age"] * ages
#             + mother_params["age_squared"] * (ages**2)
#             + mother_extra
#         )
#         # Probability(alive=1) = 1 / [1 + exp(-logit)]
#         mother_prob_alive = 1.0 / (1.0 + np.exp(-logit_mother))

#         ax.plot(
#             ages,
#             mother_prob_alive,
#             label=f"Mother - {health_cond}",
#             linestyle=line_styles["Mother"],
#         )

#         # -- FATHER
#         father_params = coeffs_father["category_1"]
#         father_extra = father_params.get(health_cond, 0.0)
#         logit_father = (
#             father_params["intercept"]
#             + father_params["age"] * ages
#             + father_params["age_squared"] * (ages**2)
#             + father_extra
#         )
#         father_prob_alive = 1.0 / (1.0 + np.exp(-logit_father))

#         ax.plot(
#             ages,
#             father_prob_alive,
#             label=f"Father - {health_cond}",
#             linestyle=line_styles["Father"],
#         )

#     ax.set_xlabel("Age")
#     ax.set_ylabel("Probability(Alive=1)")
#     ax.grid(True)
#     ax.legend()
#     plt.show()


def plot_probabilities_by_health(coeffs_mother, coeffs_father, ages):
    """
    Plot probability of each 'category_X' by age, for different health conditions,
    for Mother and Father side by side.

    Parameters
    ----------
    coeffs_mother : dict
        Nested dictionary from create_nested_dict_from_params(df_mother).
    coeffs_father : dict
        Nested dictionary from create_nested_dict_from_params(df_father).
    ages : array-like
        Sequence of ages for the x-axis.
    """
    # Example health conditions to cycle through
    health_conditions = ["good_health", "medium_health", "bad_health"]

    fig, axes = plt.subplots(1, len(health_conditions), figsize=(18, 6), sharey=True)
    fig.suptitle("Health by Age (Mothers and Fathers)")

    for i, health_cond in enumerate(health_conditions):
        ax = axes[i]

        # Calculate probabilities for MOTHER under this health condition
        probs_mother = calculate_probabilities_for_health(
            coeffs_mother, ages, health_cond
        )
        # Calculate probabilities for FATHER
        probs_father = calculate_probabilities_for_health(
            coeffs_father, ages, health_cond
        )

        # Plot each category for mother
        for cat_name, prob_array in probs_mother.items():
            ax.plot(ages, prob_array, label=f"{cat_name} (Mother)", linestyle="--")

        # Plot each category for father
        for cat_name, prob_array in probs_father.items():
            ax.plot(ages, prob_array, label=f"{cat_name} (Father)", linestyle="-")

        ax.set_xlabel("Age")
        ax.set_ylabel("Probability")
        ax.set_title(f'Previous period: {health_cond.replace("_", " ").capitalize()}')
        ax.legend()
        ax.grid(True)

    plt.show()


def calculate_probabilities_for_health(coeffs_dict, ages, health_condition):
    """Compute multinomial logit probabilities.


    Parameters
    ----------
    coeffs_dict : dict
        Nested dictionary { "category_1": {...}, "category_2": {...}, ... }
        Each inner dict must have keys:
            ["intercept", "age", "age_squared", "medium_health", "bad_health"].
    ages : array-like
        Sequence of ages for which to compute the probabilities.
    health_condition : str
        One of ['good_health', 'medium_health', 'bad_health'] or anything
        that might appear in the dictionary keys.

    Returns
    -------
    probabilities : dict of np.ndarray
        Keys = ["baseline", "category_1", "category_2", ...]
        Each value is an array of probabilities at each age.
    """
    # 1) Compute logits for each category
    logits = {}
    for cat_name, params in coeffs_dict.items():
        # For example, params might be:
        #   { "intercept": 1.2, "age": -0.04, "age_squared": 0.0003,
        #     "medium_health": 1.5, "bad_health": 2.7 }
        intercept = params["intercept"]
        beta_age = params["age"]
        beta_age_sq = params["age_squared"]

        # Only add param if the category matches health_condition
        # e.g., if health_condition = "bad_health", add params["bad_health"] to logit
        extra = params.get(health_condition, 0.0)

        logit_vals = intercept + beta_age * ages + beta_age_sq * (ages**2) + extra
        logits[cat_name] = logit_vals

    # 2) Exponentiate the logits
    exp_logits = {
        cat_name: np.exp(logit_vals) for cat_name, logit_vals in logits.items()
    }

    # 3) Probability denominator = 1 + sum over all exponentiated logits
    #    The "1" corresponds to the baseline category's exp(0) = 1.
    sum_exp = np.zeros_like(ages, dtype=float)

    for cat_name in exp_logits:
        sum_exp += exp_logits[cat_name]
    denominator = 1.0 + sum_exp

    # 4) Compute probabilities for each category and for the baseline
    probabilities = {"good_health": 1.0 / denominator}
    for cat_name in exp_logits:
        probabilities[cat_name] = exp_logits[cat_name] / denominator

    return probabilities


def create_nested_dict_from_params(df_parent, outcome):
    """
    Convert a pivoted parameter DataFrame (e.g., from pivot_model_params_parent)
    into a nested dictionary of coefficients, keyed by category.

    Expects df_parent to have columns:
        ["health_cat", "const", "age", "age_sq", "medium_health", "bad_health"]

    Returns
    -------
    nested_dict : dict
        {
            "category_1": {
                "intercept": float,
                "age": float,
                "age_squared": float,
                "medium_health": float,
                "bad_health": float
            },
            "category_2": {...},
            ...
        }
    """
    health_conditions = ["good_health", "medium_health", "bad_health"]

    nested_dict = {}

    # Collect all unique categories (e.g., 1, 2, 3).
    categories = sorted(df_parent[outcome].unique())

    for cat in categories:
        row = df_parent.loc[df_parent[outcome] == cat].squeeze()

        nested_dict[health_conditions[cat]] = {
            "intercept": row.get("const", 0.0),
            "age": row.get("age", 0.0),
            "age_squared": row.get("age_sq", 0.0),
            "medium_health": row.get("medium_health", 0.0),
            "bad_health": row.get("bad_health", 0.0),
        }

    return nested_dict


# ====================================================================================
# Test
# ====================================================================================


def test_probabilities_sum_to_one(coeffs_mother, coeffs_father):
    """
    Test that, for each parent (Mother/Father), across all relevant ages and
    each 'health_condition' scenario, the sum of probabilities across baseline
    + all categories is 1 (within a reasonable floating-point tolerance).
    """
    ages = np.arange(60, 101)

    # Example health conditions you might have
    health_conditions = ["good_health", "medium_health", "bad_health"]

    # We'll test both mother and father
    parent_coeffs = {"Mother": coeffs_mother, "Father": coeffs_father}

    for parent_label, coeffs in parent_coeffs.items():
        for health_cond in health_conditions:
            # Calculate probabilities for every age in 'ages'
            probabilities_dict = calculate_probabilities_for_health(
                coeffs, ages, health_cond
            )

            # We sum across all categories (including "baseline")
            total_prob = np.zeros_like(ages, dtype=float)
            for _cat_name, prob_arr in probabilities_dict.items():
                total_prob += prob_arr

            # Check if the sum is 1 (within floating-point tolerance)
            assert np.allclose(
                total_prob, 1.0, atol=1e-7
            ), f"Probabilities do not sum to 1 for {parent_label} in {health_cond}."

    print("All probabilities sum to 1 for each parent and health condition.")
