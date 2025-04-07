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

    health_transition_matrix = pd.read_csv(path_to_health_transition_matrix)
    health_death_transition_matrix = pd.read_csv(path_to_health_death_transition_matrix)
    # plot_weighted_adl_probabilities(df_combined, health_transition_matrix)
    plot_weighted_adl_death_probabilities(df_combined, health_death_transition_matrix)

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
    Plots predicted probabilities for three care degree outcomes (adl categories) against age,
    separately by gender and fixed subjective health levels. The grid is arranged with two rows (genders)
    and three columns (fixed subjective health: good, medium, bad). In each subplot, the predicted probabilities
    for the three adl categories are plotted as a function of age.

    Parameters:
    -----------
    params : pandas.DataFrame
        DataFrame containing estimated logit parameters for each outcome.
        Expected columns: 'sex', 'adl_cat', 'const', 'age', 'age_sq',
        'medium_health', 'bad_health'. Here, adl_cat indicates the care degree outcome:
            1: Care Degree 2
            2: Care Degree 3
            3: Care Degree 4 or 5
        The subjective health effect is incorporated via dummy variables where good health is the reference.
        In this function we hold the health variable fixed at:
            good = 2, medium = 1, bad = 0.
    """
    # Define a range of ages over which to compute probabilities
    age_vals = np.linspace(
        66, 101, 300
    )  # adjust the range and number of points as needed

    # Define outcome labels corresponding to adl_cat values
    outcome_labels = {1: "Care Degree 2", 2: "Care Degree 3", 3: "Care Degree 4 or 5"}

    # Define fixed subjective health levels mapping:
    # key is the fixed value to be used; value is the label.
    # Note: since the model was estimated with good as the omitted category,
    # when health is good (fixed value = 2) both dummies are 0.
    fixed_health_levels = {2: "Good", 1: "Medium", 0: "Bad"}

    # Get unique genders from the parameters dataframe
    genders = params["sex"].unique()

    # Create a 2 x 3 grid of subplots:
    # rows: genders (first row: first gender, second row: second gender)
    # columns: fixed health levels in order: Good (2), Medium (1), Bad (0)
    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(18, 10), sharex=True, sharey=True
    )

    # Loop over genders (each row) and fixed health levels (each column)
    # Sorting fixed_health_levels keys in reverse so that columns are in order: 2 (Good), 1 (Medium), 0 (Bad)
    for i, gender in enumerate(genders):
        # Subset the parameters for the current gender and sort by adl_cat
        df_gender = params[params["sex"] == gender].sort_values(by="adl_cat")

        for j, (fixed_health_value, health_label) in enumerate(
            sorted(fixed_health_levels.items(), reverse=True)
        ):
            # Initialize a list to collect the utilities for each outcome row
            utilities = []

            # Loop over the three rows (each outcome) in the gender-specific dataframe
            for _, row in df_gender.iterrows():
                # Compute the fixed health effect based on the fixed health level:
                # For the reference outcome (adl_cat==1) the effect is always 0.
                # For adl_cat==2, add the medium_health effect only if fixed health is Medium (1).
                # For adl_cat==3, add the bad_health effect only if fixed health is Bad (0).
                if row["adl_cat"] == 1:
                    health_effect = 0.0
                elif row["adl_cat"] == 2:
                    health_effect = (
                        row["medium_health"] if fixed_health_value == 1 else 0.0
                    )
                elif row["adl_cat"] == 3:
                    health_effect = (
                        row["bad_health"] if fixed_health_value == 0 else 0.0
                    )
                else:
                    health_effect = 0.0

                # Compute the linear predictor (utility) as a function of age
                util = (
                    row["const"]
                    + row["age"] * age_vals
                    + row["age_sq"] * (age_vals**2)
                    + health_effect
                )
                utilities.append(util)

            # Convert list of utilities to a NumPy array with shape (3, len(age_vals))
            utilities = np.array(utilities)

            # Apply the softmax transformation over outcomes for each age value
            exp_util = np.exp(utilities)
            prob = exp_util / np.sum(exp_util, axis=0)

            # Plot the predicted probability curves for each adl outcome on the current subplot
            ax = axes[i, j]
            for k, label in outcome_labels.items():
                # Since df_gender is sorted by adl_cat, index 0 corresponds to adl_cat 1, etc.
                ax.plot(age_vals, prob[k - 1, :], label=label)

            ax.set_title(f"{gender} - Subjective Health: {health_label}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Probability")
            ax.legend()

    plt.tight_layout()
    plt.show()
    # breakpoint()


def plot_weighted_adl_probabilities(params, health_transition_matrix):
    """
    For each gender, plots weighted predicted probabilities for the three adl (care degree) outcomes as a function of age.
    The weights are computed from the health_transition_matrix (which provides transition probabilities for subjective health),
    and the adl probabilities are computed from the estimated parameters. The idea is that the final probability for a given adl
    outcome at a given age is the mixture (weighted average) over the probabilities computed under each fixed subjective health state.

    In the params DataFrame:
      - Each row corresponds to an outcome (adl_cat) for a given gender.
      - Columns: 'sex', 'adl_cat', 'const', 'age', 'age_sq', 'medium_health', 'bad_health'.
      - The adl_cat outcomes are interpreted as:
            1: Care Degree 2
            2: Care Degree 3
            3: Care Degree 4 or 5
      - Subjective health enters as a set of dummy variables. The omitted (reference) category is Good Health.
        For the purposes of this function we map:
            Good Health   -> fixed value 2 (so no dummy is added)
            Medium Health -> fixed value 1 (so for adl_cat 2, add medium_health)
            Bad Health    -> fixed value 0 (so for adl_cat 3, add bad_health)

    The health_transition_matrix DataFrame contains columns:
      'sex', 'period', 'health', 'lead_health', 'transition_prob'
    where period is defined relative to age 30 (i.e. period = age - 30). For a given gender and period,
    we compute marginal probabilities for being in each subjective health state (using the rows for different starting states).

    The weighted adl probability is computed as:
      P(adl outcome | age) = sum_{subjective health state in {Good, Medium, Bad}}
            [ P(adl outcome | fixed health state) * P(subjective health state at age) ]

    Parameters:
    -----------
    params : pandas.DataFrame
        DataFrame with estimated parameters (see description above).
    health_transition_matrix : pandas.DataFrame
        DataFrame with health transition probabilities. Must include columns: 'sex', 'period', 'health', 'lead_health', 'transition_prob'.

    """
    # Define age range over which to compute probabilities.
    age_vals = np.arange(66, 101)  # adjust as needed

    # Outcome labels for the adl categories
    outcome_labels = {1: "Care Degree 2", 2: "Care Degree 3", 3: "Care Degree 4 or 5"}

    # Mapping from subjective health string in the transition matrix to a fixed value used in the utility calculations.
    # (Remember: Good is reference and gets fixed value 2; Medium is 1; Bad is 0)
    health_mapping = {"Good Health": 2, "Medium Health": 1, "Bad Health": 0}

    # Define a helper function that, given a row of params, computes its utility for a given fixed subjective health value.
    def compute_utility(row, fixed_health, ages):
        # For adl_cat 1: no health effect.
        # For adl_cat 2: add medium_health if fixed_health equals 1 (i.e. medium)
        # For adl_cat 3: add bad_health if fixed_health equals 0 (i.e. bad)
        if row["adl_cat"] == 1:
            health_effect = 0.0
        elif row["adl_cat"] == 2:
            health_effect = row["medium_health"] if fixed_health == 1 else 0.0
        elif row["adl_cat"] == 3:
            health_effect = row["bad_health"] if fixed_health == 0 else 0.0
        else:
            health_effect = 0.0
        return (
            row["const"] + row["age"] * ages + row["age_sq"] * (ages**2) + health_effect
        )

    # Get unique genders in params
    genders = params["sex"].unique()
    n_genders = len(genders)

    # Prepare subplots: one plot per gender.
    fig, axes = plt.subplots(1, n_genders, figsize=(7 * n_genders, 5), sharey=True)
    if n_genders == 1:
        axes = [axes]

    # Loop over genders
    for idx, gender in enumerate(genders):
        ax = axes[idx]
        # Subset the params for this gender and sort by adl_cat so that row order is: 1, 2, 3.
        df_gender = params[params["sex"] == gender].sort_values(by="adl_cat")
        # We expect exactly three rows (one per adl_cat)
        if df_gender.shape[0] != 3:
            raise ValueError(
                f"Expected 3 rows for gender {gender}, but got {df_gender.shape[0]}"
            )

        # Precompute adl probabilities for each fixed subjective health state.
        # For each fixed health state we create a 3 x len(age_vals) matrix.
        # We'll store these in a dictionary keyed by the subjective health state string.
        adl_probs_by_health = {}
        for health_str, fixed_val in health_mapping.items():
            # Compute utilities for each adl outcome (each row) as a function of age.
            utilities = []
            for _, row in df_gender.iterrows():
                util = compute_utility(row, fixed_val, age_vals)
                utilities.append(util)
            utilities = np.vstack(utilities)  # shape: (3, len(age_vals))
            # Compute softmax across the 3 outcomes for each age value:
            exp_util = np.exp(utilities)
            probs = exp_util / np.sum(exp_util, axis=0)
            adl_probs_by_health[health_str] = probs  # shape: (3, len(age_vals))

        # Now, for each age we get the subjective health (transition) weights and combine the adl probabilities.
        weighted_adl = np.zeros((3, len(age_vals)))
        for i, age in enumerate(age_vals):
            # Map age to period: assume period = age - 30 (rounded to nearest integer)
            period_val = int(round(age)) - 30
            # Filter the transition matrix for this gender and period.
            sub_htm = health_transition_matrix[
                (health_transition_matrix["sex"] == gender)
                & (health_transition_matrix["period"] == period_val)
            ]
            # We assume that sub_htm contains rows for all starting health states.
            # To get the marginal probability of being in a given subjective health state at this age,
            # we group by the "lead_health" (the health state in the next period) and average the transition_prob.
            grp = sub_htm.groupby("lead_health")["transition_prob"].mean()
            weights = grp.to_dict()
            # Ensure that weights for all three health states are present (default to 0 if not)
            w_good = weights.get("Good Health", 0)
            w_medium = weights.get("Medium Health", 0)
            w_bad = weights.get("Bad Health", 0)
            # Normalize the weights so they sum to 1.
            total = w_good + w_medium + w_bad
            if total > 0:
                w_good /= total
                w_medium /= total
                w_bad /= total
            # else:
            #     # If for some reason no data is available, assume equal weight.
            #     w_good = w_medium = w_bad = 1 / 3.0

            # Retrieve the adl probabilities for each fixed health state at this age.
            p_good = adl_probs_by_health["Good Health"][:, i]  # shape (3,)
            p_medium = adl_probs_by_health["Medium Health"][:, i]
            p_bad = adl_probs_by_health["Bad Health"][:, i]

            # Weighted probability for each adl outcome at this age.
            weighted_adl[:, i] = w_good * p_good + w_medium * p_medium + w_bad * p_bad

        # Plot the three adl outcome probability curves as a function of age.
        for adl_cat, label in outcome_labels.items():
            # Since our rows in df_gender are sorted by adl_cat (1,2,3), row index 0 corresponds to adl_cat 1, etc.
            ax.plot(age_vals, weighted_adl[adl_cat - 1, :], label=label)
        ax.set_title(f"{gender}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Weighted Probability")
        ax.legend()

    plt.tight_layout()
    plt.show()
    # breakpoint()


def plot_weighted_adl_death_probabilities(params, health_death_trans_mat_df):
    """
    Plots unconditional (weighted) predicted probabilities for the three adl (care degree)
    outcomes as a function of age, taking into account health transitions that include death.
    If a person is dead, no adl is estimated, so the overall adl probability is the survival probability
    (1 - P(Death)) times the conditional adl probability given survival.

    Parameters:
    -----------
    params : pandas.DataFrame
        DataFrame containing estimated logit parameters for each adl outcome.
        Expected columns: 'sex', 'adl_cat', 'const', 'age', 'age_sq', 'medium_health', 'bad_health'.
        The adl outcomes are interpreted as:
            1: Care Degree 2
            2: Care Degree 3
            3: Care Degree 4 or 5
        Subjective health enters via dummy variables where Good Health is the omitted (reference) category.
        For these calculations we fix subjective health as follows:
            Good Health   -> fixed value 2 (no dummy added)
            Medium Health -> fixed value 1 (adds medium_health only for adl_cat 2)
            Bad Health    -> fixed value 0 (adds bad_health only for adl_cat 3)

    health_death_trans_mat_df : pandas.DataFrame
        DataFrame with health-to-death transition probabilities.
        Expected columns: 'sex', 'period', 'health', 'lead_health', 'transition_prob', 'age'.
        Here, period is defined relative to age 30 (period = age - 30). The "lead_health" category
        includes the three living states (“Good Health”, “Medium Health”, “Bad Health”) and “Death”.
    """
    # Define the age range
    age_vals = np.arange(66, 100)  # adjust as needed

    # Labels for the adl outcomes (care degrees)
    outcome_labels = {1: "Care Degree 2", 2: "Care Degree 3", 3: "Care Degree 4 or 5"}

    # Mapping for fixed subjective health states.
    # Note: Good Health is the reference (fixed value 2), Medium Health uses 1, and Bad Health uses 0.
    health_mapping = {"Good Health": 2, "Medium Health": 1, "Bad Health": 0}

    # Helper function to compute utility for a given row in params, fixed subjective health, and ages.
    def compute_utility(row, fixed_health, ages):
        # For adl_cat==1 (Care Degree 2): always reference, so no health effect.
        # For adl_cat==2 (Care Degree 3): add medium_health if fixed_health is 1.
        # For adl_cat==3 (Care Degree 4 or 5): add bad_health if fixed_health is 0.
        if row["adl_cat"] == 1:
            health_effect = 0.0
        elif row["adl_cat"] == 2:
            health_effect = row["medium_health"] if fixed_health == 1 else 0.0
        elif row["adl_cat"] == 3:
            health_effect = row["bad_health"] if fixed_health == 0 else 0.0
        else:
            health_effect = 0.0
        return (
            row["const"] + row["age"] * ages + row["age_sq"] * (ages**2) + health_effect
        )

    # For each gender, precompute the conditional adl probabilities under each fixed subjective health state.
    # These are computed in the same way as before.
    genders = params["sex"].unique()

    # Set up one subplot per gender
    fig, axes = plt.subplots(
        1, len(genders), figsize=(7 * len(genders), 5), sharey=True
    )
    if len(genders) == 1:
        axes = [axes]

    # Loop over genders
    for idx, gender in enumerate(genders):
        ax = axes[idx]
        # Subset params for the given gender and sort by adl_cat (should be 3 rows)
        df_gender = params[params["sex"] == gender].sort_values(by="adl_cat")
        if df_gender.shape[0] != 3:
            raise ValueError(
                f"Expected 3 rows for gender {gender}, got {df_gender.shape[0]}"
            )

        # Compute conditional adl probabilities for each fixed subjective health state.
        # Store in a dictionary: keys are "Good Health", "Medium Health", "Bad Health"
        adl_probs_by_health = {}
        for health_str, fixed_val in health_mapping.items():
            utilities = []
            for _, row in df_gender.iterrows():
                util = compute_utility(row, fixed_val, age_vals)
                utilities.append(util)
            utilities = np.vstack(utilities)  # shape: (3, len(age_vals))
            # Apply softmax along the outcome dimension
            exp_util = np.exp(utilities)
            probs = exp_util / np.sum(exp_util, axis=0)
            adl_probs_by_health[health_str] = probs

        # Now, for each age, get the transition probabilities including death,
        # then compute the unconditional adl probabilities.
        weighted_adl = np.zeros((3, len(age_vals)))
        for i, age in enumerate(age_vals):
            # Determine period based on age; assume period = age - 30 (rounded to nearest integer)
            period_val = int(round(age)) - 30
            # Filter the health-death transition matrix for this gender and period
            sub_ht = health_death_trans_mat_df[
                (health_death_trans_mat_df["sex"] == gender)
                & (health_death_trans_mat_df["period"] == period_val)
            ]
            # Group by lead_health and get the average transition probability
            grp = sub_ht.groupby("lead_health")["transition_prob"].mean()
            trans_dict = grp.to_dict()
            # Extract probabilities for living states and for death
            p_death = trans_dict.get("Death", 0)
            p_good = trans_dict.get("Good Health", 0)
            p_medium = trans_dict.get("Medium Health", 0)
            p_bad = trans_dict.get("Bad Health", 0)
            # Survival probability is one minus probability of death.
            survival_prob = 1 - p_death
            # Among the living, compute normalized weights.
            sum_living = p_good + p_medium + p_bad
            if sum_living > 0:
                w_good = p_good / sum_living
                w_medium = p_medium / sum_living
                w_bad = p_bad / sum_living
            else:
                w_good = w_medium = w_bad = 1 / 3.0

            # Retrieve conditional adl probabilities for each fixed health state at age i.
            p_adl_good = adl_probs_by_health["Good Health"][:, i]  # shape (3,)
            p_adl_medium = adl_probs_by_health["Medium Health"][:, i]
            p_adl_bad = adl_probs_by_health["Bad Health"][:, i]

            # Combine the probabilities as a weighted average and multiply by survival probability.
            # That is, the unconditional probability of a given adl outcome equals:
            # survival_prob * [w_good*p_adl_good + w_medium*p_adl_medium + w_bad*p_adl_bad]
            weighted_adl[:, i] = survival_prob * (
                w_good * p_adl_good + w_medium * p_adl_medium + w_bad * p_adl_bad
            )

        # Plot the unconditional adl probabilities as a function of age.
        for adl_cat, label in outcome_labels.items():
            # df_gender is sorted by adl_cat so index 0 corresponds to adl_cat==1, etc.
            ax.plot(age_vals, weighted_adl[adl_cat - 1, :], label=label)
        ax.set_title(f"{gender}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Unconditional Probability")
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
