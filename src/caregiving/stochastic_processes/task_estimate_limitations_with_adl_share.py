"""Estimate probabilities of limitations with Activities of Daily Living.

On SHARE parent-child sample.

"""

from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml
from numpy.testing import assert_array_almost_equal as aaae
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.model.shared import (
    FEMALE,
    MALE,
    MIN_AGE_PARENTS,
    PARENT_DEAD,
    PARENT_GOOD_HEALTH,
    PARENT_MEDIUM_HEALTH,
)
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.utils import table


def task_estimate_adl_transitions_one_logit(  # noqa: PLR0912
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_parent_child_sample: Path = BLD / "data" / "parent_child_data.csv",
    path_to_save_adl_probabilities: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_params.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_transition_matrix.csv",
    path_to_save_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "adl_transitions.png",
):
    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_parent_child_sample)

    model_men, model_women = estimate_multinomial_logit_by_gender(df)

    df_men = pivot_model_params(model_men, "Men")
    df_women = pivot_model_params(model_women, "Women")
    df_combined = pd.concat([df_men, df_women], ignore_index=True)

    df_combined.to_csv(path_to_save_adl_probabilities, index=False)

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

                    if "age_cubed" in row_params and not pd.isna(
                        row_params["age_cubed"]
                    ):
                        val += row_params["age_cubed"] * (age**3)

                    if (
                        h_idx == PARENT_MEDIUM_HEALTH
                        and "medium_health" in row_params
                        and not pd.isna(row_params["medium_health"])
                    ):
                        val += row_params["medium_health"]

                    if (
                        h_idx == PARENT_GOOD_HEALTH
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

    plot_adl_probabilities_by_health(
        df, adl_transition_matrix, specs, path_to_save_plot=path_to_save_plot
    )


def task_plot_care_demand(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_adl_mat: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "adl_transition_matrix.csv",
    path_to_health_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_transition_matrix_good_medium_bad.csv",
    path_to_death_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "mortality_transition_matrix_logit_good_medium_bad.csv",
    path_to_health_death_transition_matrix: Path = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_death_transition_matrix_good_medium_bad.csv",
    path_to_health_death_transition_matrix_NEW: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "health_death_transition_matrix_good_medium_bad_NEW.csv",
    path_to_save_weighted_adl_transitions_by_age_plot: Annotated[Path, Product] = BLD
    / "plots"
    / "stochastic_processes"
    / "weighted_adl_transitions_by_age.png",
):

    specs = read_and_derive_specs(path_to_specs)

    health_trans_mat = pd.read_csv(path_to_health_transition_matrix)
    death_trans_mat = pd.read_csv(path_to_death_transition_matrix)
    health_death_trans_mat = pd.read_csv(
        path_to_health_death_transition_matrix, index_col=[0]
    )
    adl_transition_matrix = pd.read_csv(path_to_adl_mat, index_col=[0, 1, 2, 3])

    hdeath_df = build_health_death_transition_matrix(
        specs, health_trans_mat, death_trans_mat
    )
    hdeath_df.insert(
        1, "age", hdeath_df["period"] + specs["start_age"]
    )  # add right after 'sex'
    hdeath_df = hdeath_df.drop(columns="period")
    # save hdeath_df to csv in the same folder as the other transition matrices
    hdeath_df.to_csv(path_to_health_death_transition_matrix_NEW, index=False)

    plot_care_demand_from_hdeath_matrix(
        specs=specs,
        adl_transition_df=adl_transition_matrix,
        # health_death_df=hdeath_df,
        health_death_df=health_death_trans_mat,
        path_to_save_plot=path_to_save_weighted_adl_transitions_by_age_plot,
        start_age=66,
    )


def plot_care_demand_from_hdeath_matrix(
    specs: dict,
    adl_transition_df: pd.DataFrame,
    health_death_df: pd.DataFrame,
    path_to_save_plot: Path | str = "care_demand.png",
    *,
    start_age: int = 66,
    initial_alive_share: float = 0.75,
    initial_health_shares_alive: dict = {  # noqa: B006
        "Good Health": 0.188007,
        "Medium Health": 0.743285,
        "Bad Health": 0.068707,
    },
):
    """
    Simulate a cohort with a *combined* health-death transition matrix that is
    indexed by AGE (not period) and plot care demand by ADL category.

    Parameters
    ----------
    specs : dict
        Model spec containing 'end_age', 'sex_labels', 'adl_labels', …
    adl_transition_df : pd.DataFrame
        Multi-index (sex, age, health, adl_cat) – column 'transition_prob'.
    health_death_df : pd.DataFrame
        Columns ['sex','age','health','lead_health','transition_prob']
        Row sums (across lead_health) are 1 and include 'Death'.
    """

    # ───────────────────── labels & setup ────────────────────────────────
    health_states = ["Bad Health", "Medium Health", "Good Health", "Death"]
    alive_states = health_states[:-1]

    adl_labels = specs["adl_labels"]  # ['No ADL','Cat 1','Cat 2','Cat 3']
    care_labels = adl_labels[1:]  # Cat 1/2/3
    any_label = "Any ADL"

    colour_map = {
        "ADL 1": "tab:green",
        "ADL 2": "tab:orange",
        "ADL 3": "tab:red",
        any_label: "blue",
    }

    # ─────────────────── build {sex → {age → 4×4 P}} ─────────────────────
    P = {}
    for sex in specs["sex_labels"]:
        sex_block = health_death_df[health_death_df["sex"] == sex]
        mats = {}
        for age_key, grp in sex_block.groupby("age"):
            M = (
                grp.pivot(
                    index="health", columns="lead_health", values="transition_prob"
                )
                .reindex(index=health_states, columns=health_states)
                .fillna(0.0)
                .values
            )
            mats[age_key] = M
        P[sex] = mats

    # ───────────────── initial cohort vector  ────────────────────────────
    v0 = np.array(
        [
            initial_alive_share * initial_health_shares_alive["Bad Health"],
            initial_alive_share * initial_health_shares_alive["Medium Health"],
            initial_alive_share * initial_health_shares_alive["Good Health"],
            1.0 - initial_alive_share,
        ]
    )

    # ───────────────── simulation containers ─────────────────────────────
    ages = np.arange(start_age, specs["end_age"] + 1)
    demand = {
        s: {lab: [] for lab in (care_labels + [any_label])} for s in specs["sex_labels"]
    }

    # ───────────────── simulate Men / Women  ─────────────────────────────
    for sex in specs["sex_labels"]:
        v = v0.copy()
        for age in ages:
            # if we run beyond last available age, use the last matrix
            last_age = max(P[sex])
            A = P[sex].get(age, P[sex][last_age])
            v = v @ A

            alive_weights = dict(zip(alive_states, v[:-1], strict=False))

            for lab in care_labels:
                share = sum(
                    alive_weights[h]
                    * adl_transition_df.loc[(sex, age, h, lab), "transition_prob"]
                    for h in alive_states
                )
                demand[sex][lab].append(share)

            demand[sex][any_label].append(
                sum(demand[sex][lab][-1] for lab in care_labels)
            )

    # ─────────────────────── plotting ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, (ax, sex) in enumerate(zip(axes, specs["sex_labels"], strict=False)):
        for lab in care_labels:
            ax.plot(
                ages, demand[sex][lab], label=lab, color=colour_map[lab], linewidth=2
            )
        ax.plot(
            ages,
            demand[sex][any_label],
            label=any_label,
            color=colour_map[any_label],
            linewidth=2,
            linestyle="--",
        )

        # cosmetics
        ax.set_xlim(start_age, specs["end_age"])
        ax.set_ylim(0, 0.15)
        ax.set_xticks(np.arange(start_age, specs["end_age"] + 1, 5))

        if idx == 0:  # left axis
            ax.set_ylabel("Share of initial cohort")
            ax.tick_params(labelleft=True, labelright=False)
        else:  # right axis
            ax.set_ylabel("Share of initial cohort")
            ax.tick_params(labelleft=True, labelright=False)

        ax.set_xlabel("Age")
        ax.set_title(sex)
        ax.legend(title="Care level", fontsize=9, title_fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig(Path(path_to_save_plot), dpi=300)
    plt.close(fig)


def _plot_care_demand_from_hdeath_matrix(
    specs: dict,
    adl_transition_df: pd.DataFrame,
    health_death_df: pd.DataFrame,
    path_to_save_plot: Path | str = "care_demand.png",
    *,
    start_age: int = 66,
    initial_alive_share: float = 0.75,
    initial_health_shares_alive: dict = {  # noqa: B006
        "Good Health": 0.188007,
        "Medium Health": 0.743285,
        "Bad Health": 0.068707,
    },
):
    """
    Simulate a cohort with a *combined* health-death transition matrix and
    plot the share of the original cohort that needs Cat 1, Cat 2, Cat 3 and
    “Any ADL” care, separately for Men and Women.

    Parameters
    ----------
    specs : dict
        Usual model spec (age range, labels, etc.).
    adl_transition_df : pd.DataFrame
        Multi-index (sex, age, health, adl_cat) with column 'transition_prob'.
    health_death_df : pd.DataFrame
        Columns ['sex','period','health','lead_health','transition_prob'] where
        rows already include the 'Death' column and sum to 1.
    path_to_save_plot : str | pathlib.Path
        File to write the two-panel PNG.
    start_age : int            (default 66)
    initial_alive_share : float (default 0.75)
    initial_health_shares_alive : dict
        Distribution among Bad/Medium/Good at `start_age` (sums to 1).
    """

    # ── 1. Labels & state lists ────────────────────────────────────────────
    health_states = ["Bad Health", "Medium Health", "Good Health", "Death"]
    alive_states = health_states[:-1]

    adl_labels = specs["adl_labels"]  # ['No ADL','Cat 1','Cat 2','Cat 3']
    care_labels = adl_labels[1:]  # exclude 'No ADL'
    any_label = "Any ADL"

    colour_map = {
        "ADL 1": "tab:green",
        "ADL 2": "tab:orange",
        "ADL 3": "tab:red",
        any_label: "blue",
    }

    # ── 2. Build {sex → {period → 4×4 matrix}} from the DF ────────────────
    P = {}
    for sex in specs["sex_labels"]:
        sex_block = health_death_df[health_death_df["sex"] == sex]
        mats = {}
        for p, grp in sex_block.groupby("period"):
            M = (
                grp.pivot(
                    index="health", columns="lead_health", values="transition_prob"
                )
                .reindex(index=health_states, columns=health_states)
                .fillna(0.0)
                .values
            )
            mats[p] = M
        P[sex] = mats

    # ── 3. Initial cohort vector (Bad, Medium, Good, Death) ───────────────
    v0 = np.array(
        [
            initial_alive_share * initial_health_shares_alive["Bad Health"],
            initial_alive_share * initial_health_shares_alive["Medium Health"],
            initial_alive_share * initial_health_shares_alive["Good Health"],
            1.0 - initial_alive_share,
        ]
    )

    # ── 4. Containers for simulation results ──────────────────────────────
    ages = np.arange(start_age, specs["end_age"] + 1)
    demand = {
        s: {lab: [] for lab in (care_labels + [any_label])} for s in specs["sex_labels"]
    }

    # ── 5. Simulate for Men & Women separately ────────────────────────────
    for sex in specs["sex_labels"]:
        v = v0.copy()
        for age in ages:
            period = age - start_age
            A = P[sex].get(period, P[sex][max(P[sex])])  # last matrix if exhausted
            v = v @ A  # advance one year

            alive_weights = dict(zip(alive_states, v[:-1], strict=False))

            # unconditional share needing each category
            for lab in care_labels:
                share = sum(
                    alive_weights[h]
                    * adl_transition_df.loc[(sex, age, h, lab), "transition_prob"]
                    for h in alive_states
                )
                demand[sex][lab].append(share)

            demand[sex][any_label].append(
                sum(demand[sex][lab][-1] for lab in care_labels)
            )

    # ── 6. Plotting ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, (ax, sex) in enumerate(zip(axes, specs["sex_labels"], strict=False)):

        for lab in care_labels:
            ax.plot(
                ages, demand[sex][lab], label=lab, color=colour_map[lab], linewidth=2
            )
        ax.plot(
            ages,
            demand[sex][any_label],
            label=any_label,
            color=colour_map[any_label],
            linewidth=2,
            linestyle="--",
        )

        # cosmetics
        ax.set_xlim(start_age, specs["end_age"])
        ax.set_ylim(0, 0.15)
        ax.set_xticks(np.arange(start_age, specs["end_age"] + 1, 5))
        # ax.set_yticks(yticks)

        if idx == 0:  # left axis
            ax.set_ylabel("Share of initial cohort")
            ax.tick_params(labelleft=True, labelright=False)
        else:  # right axis
            ax.set_ylabel("Share of initial cohort")
            ax.tick_params(labelleft=True, labelright=False)

        ax.set_xlabel("Age")
        ax.set_title(sex)
        ax.legend(title="Care level", fontsize=9, title_fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig(Path(path_to_save_plot), dpi=300)
    plt.close(fig)


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

    # formula = "adl_cat ~ age + I(age**2) + C(health)"
    formula = "adl_cat ~ age + I(age**2)  + I(age**3) + C(health)"

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
        "I(age ** 3)": "age_cubed",
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
        "age_cubed",
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
    plt.close(fig)


def plot_any_adl_transitions(
    specs: dict,
    df: pd.DataFrame,
    path_to_save_plot: str,
):
    """
    Plot transition probabilities for:
      • every *limiting* ADL category separately, and
      • their aggregate (“Any ADL” = sum of all limiting categories),
    by health (columns) and sex (rows).  Probabilities for the
    *no-limitation* state are intentionally omitted.
    """

    df = df.reset_index().copy()

    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]

    sex_labels = specs["sex_labels"]
    health_labels = specs["health_labels_three"]
    alive_health_states = [h for h in health_labels if h != "Death"]
    adl_labels = specs["adl_labels"]

    # ── identify ADL groups ────────────────────────────────────────────────────
    # no_adl_label = adl_labels[0]  # assumed to be the “no-limitation” state
    limiting_adl_labels = adl_labels[1:]  # every other label is a limitation
    any_adl_label = "Any ADL"

    # colour palette for individual categories (keep your existing mapping)
    color_map_adl = {
        adl_labels[1]: "tab:orange",
        adl_labels[2]: "tab:green",
        adl_labels[3]: "tab:red",
    }
    # colour for the aggregate curve
    color_any_adl = "black"

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True)

    for sex_idx, sex_label in enumerate(sex_labels):

        df_sex = df[df["sex"] == sex_label]

        for health_idx, prev_health in enumerate(alive_health_states):
            df_prev = df_sex[df_sex["health"] == prev_health]
            ax = axes[sex_idx, health_idx]

            # ── plot each limiting ADL category separately ─────────────────────
            for adl_cat in limiting_adl_labels:
                df_trans = df_prev[df_prev["adl_cat"] == adl_cat].sort_values("age")
                ax.plot(
                    df_trans["age"],
                    df_trans["transition_prob"],
                    color=color_map_adl.get(adl_cat, "grey"),
                    linewidth=2,
                    label=adl_cat,
                )

            # ── plot the aggregate “Any ADL” curve ─────────────────────────────
            df_any_adl = (
                df_prev[df_prev["adl_cat"].isin(limiting_adl_labels)]
                .groupby("age", as_index=False)["transition_prob"]
                .sum()
                .sort_values("age")
            )
            ax.plot(
                df_any_adl["age"],
                df_any_adl["transition_prob"],
                color=color_any_adl,
                linewidth=2,
                linestyle="--",
                label=any_adl_label,
            )

            # ── cosmetics ───────────────────────────────────────────────────────
            ax.set_title(f"{sex_label}, {prev_health}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Transition Probability")
            ax.set_xlim(start_age, end_age)
            ax.set_ylim(0, 1)

            # legend only once per row (first column)
            if health_idx == 0:
                ax.legend(
                    title="ADL Transitions",
                    fontsize=9,
                    title_fontsize=10,
                    loc="upper left",
                )

    plt.tight_layout()
    fig.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)


def plot_care_demand_over_age(  # noqa: PLR0915
    health_death_transition_matrix: pd.DataFrame,
    adl_transition_matrix: pd.DataFrame,
    start_age: int = 66,
    end_age: int = 100,
    start_health_shares: dict | None = None,
    path_to_save_plots: Path = Path("care_demand_by_adl.png"),
):
    """
    Simulate a cohort from `start_age` to `end_age` and plot the share of the
    *initial* cohort that is in ADL categories 1, 2, 3 and “Any ADL”, separately
    for Men and Women.  (No ADL is also shown for context.)

    Parameters
    ----------
    health_death_transition_matrix : pd.DataFrame
        Columns: ['sex', 'period', 'health', 'lead_health', 'transition_prob'].
        Row sums are 1 and include the Death column.
    adl_transition_matrix : pd.DataFrame
        Multi-index (sex, age, health, adl_cat) with a 'transition_prob' column.
    start_age, end_age : int
        Age range of the simulation.
    start_health_shares : dict | None
        Initial distribution at age `start_age`.
        Keys: 'Good Health', 'Medium Health', 'Bad Health', 'Death'.
        If None, defaults to the example shares in the question.
    path_to_save_plots : pathlib.Path
        File name for the two-panel PNG.
    """

    # ── 1. Input frames ────────────────────────────────────────────────────
    hmat = health_death_transition_matrix.copy()
    amat = adl_transition_matrix.copy()

    # ── 2. Default starting shares if none provided ───────────────────────
    if start_health_shares is None:
        start_health_shares = {
            "Good Health": 0.75 * 0.188007,
            "Medium Health": 0.75 * 0.743285,
            "Bad Health": 0.75 * 0.068707,
            "Death": 0.25,
        }

    # ── 3. Labels and state lists ──────────────────────────────────────────
    health_states = ["Good Health", "Medium Health", "Bad Health", "Death"]
    alive_states = health_states[:-1]

    all_adl = list(amat.index.get_level_values("adl_cat").unique())
    no_lim = "No ADL"
    other_adl = [c for c in all_adl if c != no_lim]  # Cat 1,2,3
    adl_states = [no_lim] + sorted(other_adl)  # keep No ADL first
    any_label = "Any ADL"

    sexes = ["Men", "Women"]

    colour_map = {
        no_lim: "tab:blue",
        "Cat 1": "tab:orange",
        "Cat 2": "tab:green",
        "Cat 3": "tab:red",
        any_label: "black",
    }

    # ── 4. Plot setup ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    yticks = np.linspace(0, 1, 11)  # 0.0 … 1.0 in steps of 0.1
    ages = np.arange(start_age, end_age + 1)
    periods = end_age - start_age

    for idx, (ax, sex) in enumerate(zip(axes, sexes, strict=False)):
        # — 4.1 Build 4×4 transition matrices over time ————————
        h_sex = hmat[hmat["sex"] == sex]
        P_list = []
        for p in range(periods + 1):
            dfp = h_sex[h_sex["period"] == p]
            P = (
                dfp.pivot(
                    index="health", columns="lead_health", values="transition_prob"
                )
                .reindex(index=health_states, columns=health_states)
                .fillna(0.0)
                .values
            )
            P_list.append(P)

        # — 4.2 Simulate health shares ——————————————
        H = np.zeros((periods + 1, len(health_states)))
        H[0, :] = [start_health_shares[h] for h in health_states]
        for t in range(1, periods + 1):
            H[t, :] = H[t - 1] @ P_list[t - 1]

        # — 4.3 Calculate ADL shares ————————————————
        adl_shares = {cat: [] for cat in adl_states + [any_label]}
        for t, age in enumerate(ages):
            # individual categories
            for cat in adl_states:
                share = 0.0
                for h in alive_states:
                    idx_h = health_states.index(h)
                    prob_cat = amat.loc[(sex, age, h, cat), "transition_prob"]
                    share += H[t, idx_h] * prob_cat
                adl_shares[cat].append(share)

            # NEW: Any ADL = Cat1 + Cat2 + Cat3  (excludes the dead!)
            adl_shares[any_label].append(sum(adl_shares[cat][-1] for cat in other_adl))

        # — 4.4 Plot lines ——————————————————
        for cat in other_adl:  # Cat 1/2/3
            ax.plot(
                ages,
                adl_shares[cat],
                label=cat,
                color=colour_map.get(cat, None),
                linewidth=2,
            )
        ax.plot(
            ages,
            adl_shares[any_label],
            label=any_label,
            color=colour_map[any_label],
            linewidth=2,
            linestyle="--",
        )
        ax.plot(
            ages,
            adl_shares[no_lim],
            label=no_lim,
            color=colour_map[no_lim],
            linewidth=2,
        )

        # — 4.5 Cosmetics (axis labels & ticks) ——————————
        ax.set_title(sex)
        ax.set_xlabel("Age")
        ax.set_xlim(start_age, end_age)
        ax.set_ylim(0, 1)
        ax.set_yticks(yticks)

        if idx == 0:  # left subplot
            ax.set_ylabel("Share of initial cohort")
            ax.tick_params(labelleft=True, labelright=False)
        else:  # right subplot
            ax.set_ylabel("Share of initial cohort")
            ax.yaxis.set_label_position("right")  # NEW: label on right
            ax.tick_params(labelleft=False, labelright=True)

        ax.legend(title="Care level", fontsize=9, title_fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig(path_to_save_plots, dpi=300)


# =====================================================================================
#  Care-demand projection with separate health and mortality inputs
# =====================================================================================


def plot_care_demand_separate_mortality(  # noqa: PLR0915
    specs: dict,
    adl_transition_df: pd.DataFrame,
    health_trans_df: pd.DataFrame,
    mortality_df: pd.DataFrame,
    path_to_save_plot: str,
    start_age: int = 66,
    initial_alive_share: float = 0.75,
    initial_health_shares_alive: dict = {  # noqa: B006
        "Good Health": 0.188007,
        "Medium Health": 0.743285,
        "Bad Health": 0.068707,
    },
):
    """
    Combine (i) mortality probabilities and (ii) health-state transition
    probabilities (without death) to simulate cohort evolution and care demand.

    Parameters
    ----------
    specs : dict
        Usual model spec.
    adl_transition_df : pd.DataFrame (MultiIndex: sex, age, health, adl_cat)
        Conditional ADL probabilities already produced by your pipeline.
    health_trans_df : pd.DataFrame
        Long format (sex, period, health, lead_health, transition_prob)
        **Without** a 'Death' column - rows sum to 1.
    mortality_df : pd.DataFrame
        Columns: age, health, sex, death_prob
        • `sex` is coded 0 = Men, 1 = Women
        • `health` is coded 0 = Bad, 1 = Medium, 2 = Good
    """

    # ── label maps ──────────────────────────────────────────────────────────
    sex_map = {0: "Men", 1: "Women"}
    health_map = {0: "Bad Health", 1: "Medium Health", 2: "Good Health"}
    health_states = ["Bad Health", "Medium Health", "Good Health", "Death"]
    alive_states = health_states[:-1]

    # ── re-index mortality to (sex_label, age, health_label) ───────────────
    m_df = mortality_df.assign(
        sex_label=mortality_df["sex"].map(sex_map),
        health_label=mortality_df["health"].map(health_map),
    ).set_index(["sex_label", "age", "health_label"])["death_prob"]

    # ── re-index health transitions: (sex, period, health, lead_health) ────
    h_df = health_trans_df.set_index(["sex", "period", "health", "lead_health"])[
        "transition_prob"
    ]

    # ── convenience accessors ───────────────────────────────────────────────
    def death_prob(sex, age, health):
        """Return P(death | sex, age, health); use last available age if needed."""
        try:
            return m_df.loc[(sex, age, health)]
        except KeyError:
            max_age = m_df.index.get_level_values(1).max()
            return m_df.loc[(sex, max_age, health)]

    def health_prob(sex, period, health, lead_health):
        """Return unconditional P(lead_health | sex, period, health) before scaling."""
        try:
            return h_df.loc[(sex, period, health, lead_health)]
        except KeyError:
            max_p = h_df.index.get_level_values(1)[
                h_df.index.get_level_values(0) == sex
            ].max()
            return h_df.loc[(sex, max_p, health, lead_health)]

    # ── build one 4×4 matrix per (sex, period) on the fly ──────────────────
    def transition_matrix(sex, period, age):
        """Return a 4×4 unconditional transition matrix including Death."""
        A = np.zeros((4, 4))
        for i, h in enumerate(alive_states):
            d = death_prob(sex, age, h)
            scale = 1.0 - d  # mass that stays alive
            for j, lh in enumerate(alive_states):
                p_raw = health_prob(sex, period, h, lh)
                A[i, j] = scale * p_raw  # rescaled unconditional
            A[i, 3] = d  # Death column
        A[3, 3] = 1.0  # absorbing death
        return A

    # ── initial vector (Bad, Medium, Good, Death) ──────────────────────────
    v0 = np.array(
        [
            initial_alive_share * initial_health_shares_alive["Bad Health"],
            initial_alive_share * initial_health_shares_alive["Medium Health"],
            initial_alive_share * initial_health_shares_alive["Good Health"],
            1.0 - initial_alive_share,  # already dead at 66
        ]
    )

    # ── simulation container ───────────────────────────────────────────────
    end_age = specs["end_age"]
    ages = np.arange(start_age, end_age + 1)
    adl_labels = specs["adl_labels"]  # ['No ADL', 'Cat 1', 'Cat 2', 'Cat 3']
    care_labels = adl_labels[1:]
    any_label = "Any ADL"
    colour_map = {
        "ADL 1": "tab:green",
        "ADL 2": "tab:orange",
        "ADL 3": "tab:red",
        any_label: "blue",
    }
    demand = {
        s: {lab: [] for lab in (care_labels + [any_label])} for s in specs["sex_labels"]
    }

    # ── simulate Men / Women separately ────────────────────────────────────
    for sex in specs["sex_labels"]:
        v = v0.copy()
        for age in ages:
            period = age - start_age
            A = transition_matrix(sex, period, age)
            v = v @ A  # evolve one year
            alive_weights = dict(zip(alive_states, v[:-1], strict=False))

            # demand in absolute share of original cohort
            for lab in care_labels:
                prob = sum(
                    alive_weights[h]
                    * adl_transition_df.loc[(sex, age, h, lab), "transition_prob"]
                    for h in alive_states
                )
                demand[sex][lab].append(prob)
            demand[sex][any_label].append(
                sum(demand[sex][lab][-1] for lab in care_labels)
            )

    # ── plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, (ax, sex) in enumerate(zip(axes, specs["sex_labels"], strict=False)):
        for lab in care_labels:
            ax.plot(
                ages, demand[sex][lab], label=lab, color=colour_map[lab], linewidth=2
            )
        ax.plot(
            ages,
            demand[sex][any_label],
            label=any_label,
            color=colour_map[any_label],
            linewidth=2,
            linestyle="--",
        )

        ax.set_ylim(0, 0.15)
        if idx == 0:  # left subplot (Men)
            ax.set_ylabel("Share of original cohort in care")
            ax.tick_params(axis="y", labelleft=True, labelright=False)
        else:  # right subplot (Women)
            ax.set_ylabel("Share of original cohort in care")
            ax.yaxis.set_label_position("left")  # put label on the right
            ax.tick_params(axis="y", labelleft=True, labelright=False)

        ax.set_title(sex)
        ax.set_xlabel("Age")
        ax.set_ylabel("Share of original cohort")
        ax.set_xlim(start_age, end_age)

        ax.yaxis.set_tick_params(labelright=False)

        ax.legend(title="Care level", fontsize=9, title_fontsize=10, loc="upper left")

    plt.tight_layout()
    fig.savefig(path_to_save_plot, dpi=300)
    plt.close(fig)


def build_health_death_transition_matrix(
    specs: dict,
    health_trans_df: pd.DataFrame,
    mortality_df: pd.DataFrame,
    *,
    start_age: Optional[int] = None,
    max_period: Optional[int] = None,
) -> pd.DataFrame:
    """
    Combine (i) health-state transition probabilities (without death)
    and (ii) mortality probabilities to produce an unconditional
    4x4 transition matrix for every (sex, period).

    Parameters
    ----------
    health_trans_df : pd.DataFrame
        Columns: ['sex', 'period', 'health', 'lead_health', 'transition_prob'].
        Rows sum to 1 across 'lead_health' and do **not** include 'Death'.
    mortality_df : pd.DataFrame
        Columns: ['age', 'health', 'sex', 'death_prob'] with
            * sex    coded 0 = Men, 1 = Women
            * health coded 0 = Bad, 1 = Medium, 2 = Good
    start_age : int, default 66
        Age that corresponds to period 0 in `health_trans_df`.
    max_period : int or None
        Highest period to generate.  If None, uses
        health_trans_df['period'].max().

    Returns
    -------
    pd.DataFrame
        Columns ['sex','period','health','lead_health','transition_prob'],
        where 'health' and 'lead_health' now include 'Death'
        and every row sums to 1.
    """

    if start_age is None:
        start_age = 66
    else:
        start_age = specs["start_age_parents"]

    # ── label maps ──────────────────────────────────────────────────────
    sex_map = {0: "Men", 1: "Women"}
    health_map = {0: "Bad Health", 1: "Medium Health", 2: "Good Health"}
    alive_states = ["Bad Health", "Medium Health", "Good Health"]

    # ── tidy inputs for fast lookup ─────────────────────────────────────
    m_df = mortality_df.assign(
        sex_label=mortality_df["sex"].map(sex_map),
        health_label=mortality_df["health"].map(health_map),
    ).set_index(["sex_label", "age", "health_label"])["death_prob"]

    h_df = health_trans_df.set_index(["sex", "period", "health", "lead_health"])[
        "transition_prob"
    ]

    # helper: fetch death prob with graceful fallback
    def death_prob(sex: str, age: int, health: str) -> float:
        """P(Death | sex, age, health); fallback to last available age."""
        try:
            return m_df.loc[(sex, age, health)]
        except KeyError:
            last_age = m_df.index.get_level_values(1).max()
            return m_df.loc[(sex, last_age, health)]

    # helper: fetch health→health prob with fallback to last period
    def h2h_prob(sex: str, period: int, h: str, lh: str) -> float:
        try:
            return h_df.loc[(sex, period, h, lh)]
        except KeyError:
            last_p = h_df.index.get_level_values(1)[
                h_df.index.get_level_values(0) == sex
            ].max()
            return h_df.loc[(sex, last_p, h, lh)]

    if max_period is None:
        max_period = int(health_trans_df["period"].max())

    records = []

    for sex in ("Men", "Women"):
        for period in range(max_period + 1):
            age = start_age + period

            # rows for alive health states
            for h in alive_states:
                d = death_prob(sex, age, h)
                alive_scale = 1.0 - d
                for lh in alive_states:
                    p_scaled = alive_scale * h2h_prob(sex, period, h, lh)
                    records.append(
                        {
                            "sex": sex,
                            "period": period,
                            "health": h,
                            "lead_health": lh,
                            "transition_prob": p_scaled,
                        }
                    )
                # probability of dying
                records.append(
                    {
                        "sex": sex,
                        "period": period,
                        "health": h,
                        "lead_health": "Death",
                        "transition_prob": d,
                    }
                )

            # absorbing Death row
            records.append(
                {
                    "sex": sex,
                    "period": period,
                    "health": "Death",
                    "lead_health": "Death",
                    "transition_prob": 1.0,
                }
            )
            # zero probabilities from Death to any living state
            for lh in alive_states:
                records.append(
                    {
                        "sex": sex,
                        "period": period,
                        "health": "Death",
                        "lead_health": lh,
                        "transition_prob": 0.0,
                    }
                )

    out = pd.DataFrame.from_records(records)

    # optional sanity check: each (sex,period,health) row sums to 1
    row_sums = (
        out.groupby(["sex", "period", "health"])["transition_prob"].sum().round(6)
    )
    if not np.allclose(row_sums.values, 1.0):
        raise ValueError("Row sums are not 1.0; check inputs.")

    return out


def plot_adl_probabilities_by_health(
    df_sample: pd.DataFrame,
    adl_transition_matrix: pd.DataFrame,
    specs: dict,
    path_to_save_plot: Optional[str] = None,
) -> plt.Figure:
    """
    2 x 3 grid:
        ┌───────────────┬───────────────┬───────────────┐
        │ Men / Good    │ Men / Medium  │ Men / Bad     │
        ├───────────────┼───────────────┼───────────────┤
        │ Women / Good  │ Women / Medium│ Women / Bad   │
        └───────────────┴───────────────┴───────────────┘

    • Solid lines  : probabilities predicted by the multinomial-logit
    • Scatter dots : empirical probabilities in the estimation sample
                     (conditional on being alive)

    Parameters
    ----------
    df_sample
        Estimation sample with columns
        ['gender', 'age', 'health', 'adl_cat']
        *gender*: 1 = Men, 2 = Women (constants MALE/FEMALE in your code)
        *health*: 0 = Bad, 1 = Medium, 2 = Good
        *adl_cat*: 0 = No ADL, 1 = ADL 1, 2 = ADL 2, 3 = ADL 3
    adl_transition_matrix
        DataFrame produced right after the prediction loop.
        MultiIndex = (sex, age, health, adl_cat),
        value column = 'transition_prob'.
    specs
        Full specification dictionary created by `read_and_derive_specs`.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1.  String ↔ code mappings that appear in the file layout
    # ──────────────────────────────────────────────────────────────────────────
    gender_map = {1: "Men", 2: "Women"}
    health_map_num_to_str = {
        0: "Bad Health",
        1: "Medium Health",
        2: "Good Health",
    }
    adl_map_num_to_str = {
        0: "No ADL",
        1: "ADL 1",
        2: "ADL 2",
        3: "ADL 3",
    }
    adl_colors = {
        "No ADL": "blue",
        "ADL 1": "green",
        "ADL 2": "orange",
        "ADL 3": "red",
    }

    health_labels = specs["health_labels_three"][:-1]  # excludes "Death"
    adl_order = specs["adl_labels"]  # ["No ADL", "ADL 1", "ADL 2", "ADL 3"]

    # ──────────────────────────────────────────────────────────────────────────
    # 2.  Empirical (observed) probabilities
    # ──────────────────────────────────────────────────────────────────────────
    df_obs = df_sample.copy()
    df_obs = df_obs[df_obs["health"] != PARENT_DEAD]  # drop death rows if present

    df_obs["sex"] = df_obs["gender"].map(gender_map)
    df_obs["health_str"] = df_obs["health"].map(health_map_num_to_str)
    df_obs["adl_str"] = df_obs["adl_cat"].map(adl_map_num_to_str)

    # counts per (sex, age, health, adl)
    counts = (
        df_obs.groupby(["sex", "age", "health_str", "adl_str"])
        .size()
        .rename("n")
        .reset_index()
    )
    # alive totals per (sex, age, health)
    totals = (
        df_obs.groupby(["sex", "age", "health_str"])
        .size()
        .rename("total")
        .reset_index()
    )

    df_emp = counts.merge(totals, on=["sex", "age", "health_str"])
    df_emp["prob"] = df_emp["n"] / df_emp["total"]

    # ──────────────────────────────────────────────────────────────────────────
    # 3.  Predicted probabilities
    # ──────────────────────────────────────────────────────────────────────────
    df_pred = adl_transition_matrix.reset_index().rename(
        columns={"transition_prob": "prob"}
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 4.  Plot
    # ──────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    for row, sex in enumerate(specs["sex_labels"]):  # ["Men", "Women"]
        for col, hlth in enumerate(health_labels):
            ax = axes[row, col]

            # predicted ––– solid lines
            for adl in adl_order:
                dat = df_pred[
                    (df_pred["sex"] == sex)
                    & (df_pred["health"] == hlth)
                    & (df_pred["adl_cat"] == adl)
                ]
                ax.plot(dat["age"], dat["prob"], label=adl, color=adl_colors[adl])

            # observed ––– scatter points
            for adl in adl_order:
                dat = df_emp[
                    (df_emp["sex"] == sex)
                    & (df_emp["health_str"] == hlth)
                    & (df_emp["adl_str"] == adl)
                ]
                ax.scatter(
                    dat["age"], dat["prob"], s=12, alpha=1, color=adl_colors[adl]
                )

            # cosmetics
            if row == 1:
                ax.set_xlabel("Age")
            if col == 0:
                ax.set_ylabel("Probability")
            ax.set_title(f"{sex} - {hlth.replace(' Health','')}")

    # one legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(adl_order))
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if path_to_save_plot:
        fig.savefig(path_to_save_plot, dpi=300)
