"""Read family specs."""

import numpy as np
import pandas as pd
from jax import numpy as jnp

FEMALE = 1


def predict_children_by_state(params, specs):
    """Predict the number of children in the household conditional on state.

    Produces array of shape
        (n_sexes, n_education_types, n_has_partner_states, n_periods)

    """
    n_periods = specs["end_age"] - specs["start_age"] + 1

    children = np.zeros((2, specs["n_education_types"], 2, n_periods))

    for sex in (0, 1):
        for edu in range(specs["n_education_types"]):
            for has_partner in (0, 1):
                for period in range(n_periods):
                    predicted_nb_children = (
                        params.loc[(sex, edu, has_partner), "const"]
                        + params.loc[(sex, edu, has_partner), "period"] * period
                        + params.loc[(sex, edu, has_partner), "period_sq"] * period**2
                    )
                    children[sex, edu, has_partner, period] = np.maximum(
                        0, predicted_nb_children
                    )

    max_children = jnp.max(children, axis=-1)
    return jnp.asarray(children), max_children


def predict_age_of_youngest_child_by_state(params, specs):
    """Predict the age of the youngest child conditional on state.

    Produces array of shape
        (n_sexes, n_education_types, n_has_partner_states, n_periods)

    """
    n_periods = specs["end_age"] - specs["start_age"] + 1

    kidage_youngest = np.zeros((2, specs["n_education_types"], 2, n_periods))

    for sex in (0, 1):
        for edu in range(specs["n_education_types"]):
            for has_partner in (0, 1):
                for period in range(n_periods):
                    predicted_kidage_youngest = (
                        params.loc[(sex, edu, has_partner), "const"]
                        + params.loc[(sex, edu, has_partner), "period"] * period
                        + params.loc[(sex, edu, has_partner), "period_sq"] * period**2
                    )
                    kidage_youngest[sex, edu, has_partner, period] = np.maximum(
                        0, predicted_kidage_youngest
                    )

    return jnp.asarray(kidage_youngest)


def _read_in_partner_transition_specs(trans_mat, specs):
    """Read in partner transition probabilities."""

    n_periods = specs["n_periods"]
    n_partner_states = trans_mat.index.get_level_values(
        "lagged_partner_state"
    ).nunique()

    n_edu_types = len(specs["education_labels"])
    sexes = [0, 1]
    age_bins = trans_mat.index.get_level_values("age_bin").unique()

    full_series_index = pd.MultiIndex.from_product(
        [
            sexes,
            range(n_edu_types),
            age_bins,
            range(n_partner_states),
            range(n_partner_states),
        ],
        names=trans_mat.index.names,
    )
    full_series = pd.Series(index=full_series_index, data=0.0, name=trans_mat.name)
    full_series.update(trans_mat)

    # Transition probalities for partner
    female_trans_probs = np.zeros(
        (n_edu_types, n_periods, n_partner_states, n_partner_states), dtype=float
    )

    for edu in range(n_edu_types):
        for period in range(n_periods):
            for current_state in range(n_partner_states):
                for next_state in range(n_partner_states):
                    age = period + specs["start_age"]
                    # Check if age is in between 40 and 50, 50 and 60, 60 and 70
                    age_bin = np.floor(age / 10) * 10
                    female_trans_probs[edu, period, current_state, next_state] = (
                        full_series.loc[
                            (FEMALE, edu, age_bin, current_state, next_state)
                        ]
                    )
    return jnp.asarray(female_trans_probs), n_partner_states


def read_in_partner_transition_specs(est_probs_df, specs):

    n_periods = specs["n_periods"]
    n_partner_states = specs["n_partner_states"]
    n_edu_types = specs["n_education_types"]
    n_sexes = specs["n_sexes"]

    # Transition probalities for partner
    trans_probs = np.zeros(
        (n_sexes, n_edu_types, n_periods, n_partner_states, n_partner_states),
        dtype=float,
    )

    for sex_var, _sex_label in enumerate(specs["sex_labels"]):
        for edu_var, _edu_label in enumerate(specs["education_labels"]):
            for period in range(n_periods):
                for partner_state_var, _partner_state_label in enumerate(
                    specs["partner_labels"]
                ):
                    age = period + specs["start_age"]

                    for lead_partner_state_var, _lead_partner_state_label in enumerate(
                        specs["partner_labels"]
                    ):
                        # Check if age is in between 30 and 40,
                        # 40 and 50, 50 and 60, 60 and 70
                        age_bin = np.floor(age / 10) * 10

                        try:
                            trans_probs[
                                sex_var,
                                edu_var,
                                period,
                                partner_state_var,
                                lead_partner_state_var,
                            ] = est_probs_df.loc[
                                (
                                    sex_var,
                                    edu_var,
                                    age_bin,
                                    partner_state_var,
                                    lead_partner_state_var,
                                ),
                                # "proportion",
                            ]
                        except KeyError:
                            trans_probs[
                                sex_var,
                                edu_var,
                                period,
                                partner_state_var,
                                lead_partner_state_var,
                            ] = 0
                    # Assign absorbing 1 if no one in the data
                    if not np.allclose(
                        trans_probs[sex_var, edu_var, period, partner_state_var].sum(),
                        1,
                    ):
                        trans_probs[
                            sex_var,
                            edu_var,
                            period,
                            partner_state_var,
                            partner_state_var,
                        ] = 1
                        #
                        # mask = (
                        #     (est_probs_df["sex"] == sex_label)
                        #     & (est_probs_df["education"] == edu_label)
                        #     & (est_probs_df["age"] == age)
                        #     & (est_probs_df["partner_state"] == partner_state_label)
                        #     & (
                        #         est_probs_df["lead_partner_state"]
                        #         == lead_partner_state_label
                        #     )
                        # )
                        # trans_probs[
                        #     sex_var,
                        #     edu_var,
                        #     period,
                        #     partner_state_var,
                        #     lead_partner_state_var,
                        # ] = est_probs_df.loc[mask, "probability"].values[0]

                # Check whether trans_probs[sex, edu, age, :. :] sum to one
                assert np.allclose(
                    trans_probs[sex_var, edu_var, period, :, :].sum(axis=1),
                    np.array([1, 1, 1]),
                )
    return jnp.asarray(trans_probs), n_partner_states
