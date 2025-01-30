"""Read family specs."""

import numpy as np
import pandas as pd
from jax import numpy as jnp

FEMALE = 1


def read_in_partner_transition_specs(paths_dict, specs):
    """Read in partner transition probabilities."""

    trans_probs = pd.read_csv(
        paths_dict["external_estimation_results"] + "partner_transition_matrix.csv",
        index_col=[0, 1, 2, 3, 4],
    )["proportion"]

    n_periods = specs["n_periods"]
    n_partner_states = trans_probs.index.get_level_values("partner_state").nunique()
    n_edu_types = len(specs["education_labels"])
    sexes = [0, 1]
    age_bins = trans_probs.index.get_level_values("age_bin").unique()

    full_series_index = pd.MultiIndex.from_product(
        [
            sexes,
            range(n_edu_types),
            age_bins,
            range(n_partner_states),
            range(n_partner_states),
        ],
        names=trans_probs.index.names,
    )
    full_series = pd.Series(index=full_series_index, data=0.0, name=trans_probs.name)
    full_series.update(trans_probs)

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
