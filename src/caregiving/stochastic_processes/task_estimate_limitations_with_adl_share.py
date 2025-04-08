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
    # plt.show()


def plot_weighted_adl_death_probabilities(params, health_death_trans_mat_df):
    """
    Revised plotting function.

    This function computes the unconditional ADL probabilities in a two-step process:

    (1) Initialization at age 66:
         - Given initial lag health shares are provided for survivors only.
         - Since the survival (alive) share is 0.75, the initial state distribution is:
             Good Health   = 0.75 * 0.188007,
             Medium Health = 0.75 * 0.743285,
             Bad Health    = 0.75 * 0.068707,
             Death         = 1 - 0.75 = 0.25.
         This ensures the four-state distribution sums to 1.

    (2) Iteration for ages 67+:
         - For each subsequent age, we extract the gender and age-specific transition
           probabilities (for moving from the previous – or lag – health state to the current state).
         - We build a full 4x4 transition matrix (states: ["Good Health", "Medium Health", "Bad Health", "Death"]).
         - Death is imposed as absorbing.
         - The state distribution at the new age is S_new = S_old dot T.

    At each age, we then compute unconditional ADL probabilities by:
         - Converting the distribution among the living states (first three elements) into weights;
         - Computing the conditional ADL probabilities given each health state
           (using the fixed values: Good -> 2, Medium -> 1, Bad -> 0); and
         - Taking a weighted average of these conditional probabilities multiplied by the overall
           survival probability (sum of the living).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Define the age range
    age_vals = np.arange(66, 100)  # Ages 66 to 99

    # Outcome labels for the three ADL (care degree) outcomes.
    outcome_labels = {1: "Care Degree 2", 2: "Care Degree 3", 3: "Care Degree 4 or 5"}

    # Fixed mapping for subjective health used in the ADL model:
    # For "Good Health" we use fixed value 2 (no additional dummy), for "Medium Health": 1, for "Bad Health": 0.
    health_mapping = {"Good Health": 2, "Medium Health": 1, "Bad Health": 0}

    # Define the complete set of states (including death)
    states = ["Good Health", "Medium Health", "Bad Health", "Death"]

    # --- (1) Initialization at age 66 ---
    # Given lag shares (for survivors only), and the overall share alive is 0.75:
    init_alive = 0.75
    init_good = 0.188007
    init_medium = 0.743285
    init_bad = 0.068707

    # Construct state distribution at age 66:
    # Multiply survivor share by the health shares.
    S0 = np.array(
        [
            init_alive * init_good,
            init_alive * init_medium,
            init_alive * init_bad,
            1 - init_alive,
        ]
    )  # Death probability = 0.25

    # The state distribution S0 is a 4-element vector that sums to 1.
    # For example: sum(S0) == 0.75* (0.188007+0.743285+0.068707) + 0.25 = 0.75*1 + 0.25 = 1

    # Prepare to plot results for each gender.
    genders = params["sex"].unique()
    fig, axes = plt.subplots(
        1, len(genders), figsize=(7 * len(genders), 5), sharey=True
    )
    if len(genders) == 1:
        axes = [axes]

    # For each gender, iterate over age and compute:
    #   (i) the evolving state distribution via transitions,
    #  (ii) the weighted (unconditional) ADL probabilities
    for idx, gender in enumerate(genders):
        ax = axes[idx]
        # Subset parameters (logit coefficients) for the current gender.
        df_gender = params[params["sex"] == gender].sort_values(by="adl_cat")
        if df_gender.shape[0] != 3:
            raise ValueError(
                f"Expected 3 rows for gender {gender}, got {df_gender.shape[0]}"
            )

        # Initialize list to store unconditional ADL probabilities for each age.
        # Each element is a 3-element vector (one for each ADL outcome).
        unconditional_adl_list = []

        # Set the initial state distribution for age 66.
        S = S0.copy()

        # Loop over each age.
        for age in age_vals:
            if age > 66:
                # --- (2) Iterative transition for ages > 66 ---
                # Define period as age - 30 (rounded to integer).
                period_val = int(round(age)) - 30

                # Extract transitions for the given gender and period.
                # The transition data frame has columns "health" (lag state) and "lead_health" (next state).
                sub_ht = health_death_trans_mat_df[
                    (health_death_trans_mat_df["sex"] == gender)
                    & (health_death_trans_mat_df["period"] == period_val)
                ]

                # Pivot the data to create a matrix with lag states as rows and lead states as columns.
                T = sub_ht.pivot(
                    index="health", columns="lead_health", values="transition_prob"
                )

                # Ensure that all four states are present as rows and columns.
                T = T.reindex(index=states, columns=states, fill_value=0)

                # Enforce the absorbing property for death.
                T.loc["Death"] = [0, 0, 0, 1]

                # Optionally, one might also check that each row sums (approximately) to 1.
                # Update the state distribution: S_{age} = S_{previous} dot T.
                S = S.dot(T.values)
            # End of state transition

            # For the current age, compute survival probability and weights among living states.
            # Living states are the first three (Good, Medium, Bad).
            survival_prob = S[0] + S[1] + S[2]
            if survival_prob > 0:
                w_good = S[0] / survival_prob
                w_medium = S[1] / survival_prob
                w_bad = S[2] / survival_prob
            else:
                # In extreme cases when survival_prob is 0 (everyone is dead), use equal weights.
                w_good = w_medium = w_bad = 1 / 3.0

            # --- Compute conditional ADL probabilities at the current age ---
            # For each health state, we compute conditional outcome probabilities using the multinomial logit:
            cond_adl = (
                {}
            )  # will hold a 3-element probability vector for each living health state.
            for health_str, fixed_val in health_mapping.items():
                utilities = []
                # Iterate over each ADL outcome (the rows in df_gender).
                for _, row in df_gender.iterrows():
                    # For outcome 1 (Care Degree 2), no dummy effect is added.
                    if row["adl_cat"] == 1:
                        health_effect = 0.0
                    # For outcome 2 (Care Degree 3), add "medium_health" coefficient if fixed value is 1.
                    elif row["adl_cat"] == 2:
                        health_effect = row["medium_health"] if fixed_val == 1 else 0.0
                    # For outcome 3 (Care Degree 4 or 5), add "bad_health" coefficient if fixed value is 0.
                    elif row["adl_cat"] == 3:
                        health_effect = row["bad_health"] if fixed_val == 0 else 0.0
                    else:
                        health_effect = 0.0
                    # Compute the utility as a function of age.
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
                cond_adl[health_str] = cond_prob  # Vector of length 3
            # End of conditional probabilities

            # --- Compute the unconditional ADL probabilities ---
            # They are given by the survival probability (the share alive) multiplied by a weighted
            # average of the conditional ADL probabilities, where the weights come from the living state shares.
            uncond_adl = survival_prob * (
                w_good * cond_adl["Good Health"]
                + w_medium * cond_adl["Medium Health"]
                + w_bad * cond_adl["Bad Health"]
            )
            unconditional_adl_list.append(uncond_adl)
        # End loop over ages

        # Convert the collected probabilities to an array for plotting;
        # shape: (3 outcomes, len(age_vals))
        unconditional_adl_arr = np.array(unconditional_adl_list).T

        # Plot each ADL outcome.
        for adl_cat, label in outcome_labels.items():
            # df_gender rows were ordered by outcome (adl_cat 1, 2, 3),
            # so we use adl_cat-1 to index the corresponding unconditional probability.
            ax.plot(age_vals, unconditional_adl_arr[adl_cat - 1, :], label=label)
        ax.set_title(f"{gender}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Unconditional ADL Probability")
        ax.legend()

    plt.tight_layout()
    plt.show()
    # breakpoint()


def plot_health_state_evolution(health_death_trans_mat_df):
    """
    Plots the evolution of the state distribution across ages.

    This function begins at age 66 with the initial state distribution based on
    the provided shares:
         - Good Health   = 0.75 * 0.188007,
         - Medium Health = 0.75 * 0.743285,
         - Bad Health    = 0.75 * 0.068707,
         - Death         = 1 - 0.75.

    For each subsequent age (67+), the function retrieves the transition probabilities
    for the given gender and period (period = age - 30) from the health_death_trans_mat_df
    and computes the new state distribution, ensuring that Death is absorbing.

    The resulting state distributions are plotted over the age range for each gender.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Define the age range, e.g. 66 to 99.
    age_vals = np.arange(66, 100)

    # Define the four states.
    states = ["Good Health", "Medium Health", "Bad Health", "Death"]

    # --- Initialization at Age 66 ---
    # Provided shares for survivors (alive) and their health composition:
    init_alive = 0.75
    init_good = 0.188007
    init_medium = 0.743285
    init_bad = 0.068707

    # Create the initial state distribution vector S0.
    # The first three elements are the probabilities for the living states,
    # and the fourth element is the death probability.
    S0 = np.array(
        [
            init_alive * init_good,  # Good Health
            init_alive * init_medium,  # Medium Health
            init_alive * init_bad,  # Bad Health
            1 - init_alive,  # Death
        ]
    )
    # S0 now sums to 1.

    # Determine the genders present in the transition matrix.
    genders = health_death_trans_mat_df["sex"].unique()

    # Set up a subplot for each gender.
    fig, axes = plt.subplots(
        1, len(genders), figsize=(7 * len(genders), 5), sharey=True
    )
    if len(genders) == 1:
        axes = [axes]

    # Loop over each gender.
    for idx, gender in enumerate(genders):
        ax = axes[idx]
        # Create a list to store the state distribution at each age.
        state_history = []
        # Set the initial state distribution at age 66.
        S = S0.copy()

        # Iterate over each age in the range.
        for age in age_vals:
            # For age 66, we already have S0.
            if age > 66:
                # Define the period as age - 30 (using integer rounding).
                period_val = int(round(age)) - 30
                # Filter the transition matrix for the current gender and period.
                sub_ht = health_death_trans_mat_df[
                    (health_death_trans_mat_df["sex"] == gender)
                    & (health_death_trans_mat_df["period"] == period_val)
                ]
                # Pivot the data to form a 4x4 matrix with rows as lag states and columns as lead states.
                T = sub_ht.pivot(
                    index="health", columns="lead_health", values="transition_prob"
                )
                # Ensure that the pivot table covers all four states.
                T = T.reindex(index=states, columns=states, fill_value=0)
                # Enforce that death is absorbing.
                T.loc["Death"] = [0, 0, 0, 1]
                # Update the state distribution: S_new = S_previous * T.
                S = S.dot(T.values)
            # Record the current state distribution.
            state_history.append(S.copy())

        # Convert the history to a NumPy array; shape: (n_ages, 4)
        state_history = np.array(state_history)

        # Plot the probability evolution for each state.
        for i, state in enumerate(states):
            ax.plot(age_vals, state_history[:, i], label=state)
        ax.set_title(f"Gender: {gender}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Probability")
        ax.legend()

    plt.tight_layout()
    # plt.show()
    # breakpoint()


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
