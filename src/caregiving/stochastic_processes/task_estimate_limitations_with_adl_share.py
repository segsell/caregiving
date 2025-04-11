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
from caregiving.model.shared import (
    FEMALE,
    MALE,
    MIN_AGE_PARENTS,
    STATE_GOOD_HEALTH,
    STATE_MEDIUM_HEALTH,
)
from caregiving.specs.derive_specs import read_and_derive_specs


def task_estimate_adl_transitions_one_logit(
    path_to_specs: Path = SRC / "specs.yaml",
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
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_transition_matrix.csv",
):
    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_parent_child_sample)

    model_men, model_women = estimate_multinomial_logit_by_gender(df)

    df_men = pivot_model_params(model_men, "Men")
    df_women = pivot_model_params(model_women, "Women")
    df_combined = pd.concat([df_men, df_women], ignore_index=True)

    df_combined.to_csv(path_to_save_adl_probabilities, index=False)

    # health_trans_mat = pd.read_csv(path_to_health_transition_matrix)
    # health_death_trans_mat = pd.read_csv(path_to_health_death_trans_mat)
    # plot_weighted_adl_probabilities(df_combined, health_trans_mat)

    # plot_weighted_adl_death_probabilities(df_combined, health_death_trans_mat)
    # plot_health_state_evolution(health_death_trans_mat)

    # 1. Setup the index ranges
    ages = np.arange(specs["start_age_parents"], specs["end_age"] + 1)
    health_indices = specs["alive_health_vars_three"]
    health_labels = [specs["health_labels_three"][h] for h in health_indices]

    adl_labels = specs["adl_labels"]

    # 2. Create a MultiIndex
    index = pd.MultiIndex.from_product(
        [
            specs["sex_labels"],
            ages,
            health_labels,
            adl_labels,
        ],
        names=["sex", "age", "health", "adl_cat"],
    )

    # 3. Prepare an empty DataFrame
    adl_transition_matrix = pd.DataFrame(index=index, columns=["transition_prob"])

    for sex_label in specs["sex_labels"]:

        df_sex = df_combined[df_combined["sex"] == sex_label].copy()

        for h_idx, h_label in zip(health_indices, health_labels, strict=False):

            cat_params = {}
            for _, row in df_sex.iterrows():
                cat = row["adl_cat"]  # 1,2,3
                cat_params[cat] = row

            probs = np.zeros((len(ages), 4))

            for i, age in enumerate(ages):

                lp = [0.0, 0.0, 0.0, 0.0]

                for cat in (1, 2, 3):
                    if cat not in cat_params:
                        lp[cat] = 0.0
                        continue

                    row_params = cat_params[cat]
                    val = 0.0

                    if not pd.isna(row_params["const"]):
                        val += row_params["const"]

                    if "age" in row_params and not pd.isna(row_params["age"]):
                        val += row_params["age"] * age

                    if "age_sq" in row_params and not pd.isna(row_params["age_sq"]):
                        val += row_params["age_sq"] * (age**2)

                    if (
                        h_idx == STATE_MEDIUM_HEALTH
                        and "medium_health" in row_params
                        and not pd.isna(row_params["medium_health"])
                    ):
                        val += row_params["medium_health"]

                    if (
                        h_idx == STATE_GOOD_HEALTH
                        and "good_health" in row_params
                        and not pd.isna(row_params["good_health"])
                    ):
                        val += row_params["good_health"]

                    lp[cat] = val

                # Softmax to convert lps into probabilities
                exps = np.exp(lp)
                s = exps.sum()
                p = exps / s
                probs[i, :] = p

            # 5. Store the results in adl_transition_matrix
            for cat_idx, cat_label in enumerate(adl_labels):
                adl_transition_matrix.loc[
                    (sex_label, slice(None), h_label, cat_label), "transition_prob"
                ] = probs[:, cat_idx]

    adl_transition_matrix.to_csv(path_to_save)


def task_estimate_adl_transitions_via_separate_logits(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_parent_child_sample: Path = BLD / "data" / "parent_child_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_transition_matrix_separate_logits.csv",
):

    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_parent_child_sample)

    # Define ranges for age and health for prediction purposes
    ages = np.arange(specs["start_age_parents"], specs["end_age"] + 1)

    alive_health_vars = specs["alive_health_vars_three"]
    alive_health_labels = [specs["health_labels_three"][i] for i in alive_health_vars]

    index = pd.MultiIndex.from_product(
        [
            specs["sex_labels"],
            ages,
            alive_health_labels,
            specs["adl_labels"],
        ],
        names=["sex", "age", "health", "adl_cat"],
    )

    # Compute transition probabilities
    adl_transition_matrix = pd.DataFrame(
        index=index, data=None, columns=["transition_prob"]
    )
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for alive_health_var in alive_health_vars:
            alive_health_label = specs["health_labels_three"][alive_health_var]

            # Filter the data
            data = df[(df["sex"] == sex_var) & (df["health"] == alive_health_var)]

            # Fit the logit model
            y_var = "adl_cat"
            x_vars = ["age"]
            formula = y_var + " ~ " + " + ".join(x_vars)
            model = smf.mnlogit(formula=formula, data=data)
            result = model.fit()

            # Compute the transition probabilities
            transition_probabilities = result.predict(
                pd.DataFrame({"age": ages})
            ).values

            # Take the transition probabilities
            adl_0_label = specs["adl_labels"][0]
            adl_transition_matrix.loc[
                (
                    sex_label,
                    slice(None),
                    alive_health_label,
                    adl_0_label,
                ),
                "transition_prob",
            ] = transition_probabilities[:, 0]

            adl_1_label = specs["adl_labels"][1]
            adl_transition_matrix.loc[
                (
                    sex_label,
                    slice(None),
                    alive_health_label,
                    adl_1_label,
                ),
                "transition_prob",
            ] = transition_probabilities[:, 1]

            adl_2_label = specs["adl_labels"][2]
            adl_transition_matrix.loc[
                (
                    sex_label,
                    slice(None),
                    alive_health_label,
                    adl_2_label,
                ),
                "transition_prob",
            ] = transition_probabilities[:, 2]

            adl_3_label = specs["adl_labels"][3]
            adl_transition_matrix.loc[
                (
                    sex_label,
                    slice(None),
                    alive_health_label,
                    adl_3_label,
                ),
                "transition_prob",
            ] = transition_probabilities[:, 3]

    adl_transition_matrix.to_csv(path_to_save)


def estimate_multinomial_logit_by_gender(df):
    """Estimate multinomial logit model separately by gender."""

    dat_men = df[df["gender"] == MALE].copy()
    dat_women = df[df["gender"] == FEMALE].copy()

    formula = "adl_cat ~ age + I(age**2) + C(health)"

    model_men = smf.mnlogit(formula, data=dat_men).fit()
    print("Results for men (gender == 1):")
    print(model_men.summary())

    model_women = smf.mnlogit(formula, data=dat_women).fit()
    print("\nResults for women (gender == 2):")
    print(model_women.summary())

    return model_men, model_women


def pivot_model_params(model, sex_label):
    """Return wide dataframe of model parameters."""

    df = model.params.copy()

    df = df.T

    rename_dict = {
        "Intercept": "const",
        "age": "age",
        "I(age ** 2)": "age_sq",
        "C(health)[T.1.0]": "medium_health",
        "C(health)[T.2.0]": "good_health",
    }
    df.rename(columns=rename_dict, inplace=True)

    df.index = df.index + 1
    df.index.name = "adl_cat"

    df.reset_index(inplace=True)  # Now "adl_cat" is a column

    df.insert(0, "sex", sex_label)

    desired_cols = [
        "sex",
        "adl_cat",
        "const",
        "age",
        "age_sq",
        "medium_health",
        "good_health",
    ]

    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[desired_cols]

    return df


# =====================================================================================
# Plotting
# =====================================================================================


def plot_adl_transitions(
    specs: dict,
    df: pd.DataFrame,
    path_to_save_plot: str,
):
    """Plot ADL transition probs against age, by health (cols) and sex (rows)."""

    df = df.reset_index().copy()

    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]

    sex_labels = specs["sex_labels"]
    health_labels = specs["health_labels_three"]
    alive_health_states = [h for h in health_labels if h != "Death"]
    adl_labels = specs["adl_labels"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True)

    color_map_adl = {
        adl_labels[0]: "tab:blue",
        adl_labels[1]: "tab:orange",
        adl_labels[2]: "tab:green",
        adl_labels[3]: "tab:red",
    }

    for sex_idx, sex_label in enumerate(sex_labels):

        df_sex = df[df["sex"] == sex_label]

        for health_idx, prev_health in enumerate(alive_health_states):
            df_prev = df_sex[df_sex["health"] == prev_health]
            ax = axes[sex_idx, health_idx]

            for adl_cat in adl_labels:
                df_transition = df_prev[df_prev["adl_cat"] == adl_cat].sort_values(
                    "age"
                )

                ax.plot(
                    df_transition["age"],
                    df_transition["transition_prob"],
                    color=color_map_adl.get(adl_cat, "black"),  # fallback color
                    linewidth=2,
                    label=adl_cat,
                )

            ax.set_title(f"{sex_label}, {prev_health}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Transition Probability")
            ax.set_xlim(start_age, end_age)
            ax.set_ylim(0, 1)

            # Show legend only for the first column of each row (or however you prefer)
            if health_idx == 0:
                ax.legend(
                    title="ADL Transitions",
                    fontsize=9,
                    title_fontsize=10,
                    loc="upper left",
                )

    plt.tight_layout()
    fig.savefig(path_to_save_plot, dpi=300)
    plt.show()
    plt.close(fig)


def plot_weighted_adl_death_probabilities(  # noqa: PLR0912, PLR0915
    params, health_death_trans_mat_df
):
    """Computes unconditional ADL probss by evolving state distributions from age 66.

    At age 66 the state distribution is set by:
      Good Health   = 0.75 * 0.188007,
      Medium Health = 0.75 * 0.743285,
      Bad Health    = 0.75 * 0.068707,
      Death         = 1 - 0.75.

    For ages 67+, gender and period-specific transition probabilities update the state.
    Unconditional ADL probabilities are computed as a weighted average of conditional
    probabilities using the living state proportions.

    """

    age_vals = np.arange(66, 100)

    outcome_labels = {1: "Care Degree 2", 2: "Care Degree 3", 3: "Care Degree 4 or 5"}
    health_mapping = {"Good Health": 2, "Medium Health": 1, "Bad Health": 0}
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
            if age >= MIN_AGE_PARENTS:
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
    """Plot the evolution of state distributions across ages, from age 66 on.

    Starts at age 66 with:
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
