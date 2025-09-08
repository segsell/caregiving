"""Create transition matrices for care demand and care supply."""

import jax.numpy as jnp
import numpy as np

from caregiving.model.shared import (
    INITIAL_CONDITIONS_AGE_HIGH,
    INITIAL_CONDITIONS_AGE_LOW,
    INITIAL_CONDITIONS_COHORT_HIGH,
    INITIAL_CONDITIONS_COHORT_LOW,
    MOTHER,
)


def read_in_adl_transition_specs(adl_trans_df, specs):
    """
    Build a 4-d transition array
        [sex, period (age), health_state, adl_state]
    out of the long DataFrame *adl_transitions*.

    Parameters
    ----------
    adl_trans_df : pandas.DataFrame
        Long table with columns
        ['sex', 'age', 'health', 'adl_cat', 'transition_prob'].
    specs : dict
        Master spec-dictionary that already contains the label lists
        used elsewhere in your model code.

    """
    # unpack sizes straight from *specs*
    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]
    n_periods = end_age - start_age + 1
    n_sexes = len(specs["sex_labels"])  # 2
    n_health_states = len(specs["health_labels_three"]) - 1  # 3 (excluding 'Death')
    n_adl_states = len(specs["adl_labels"])  # 4

    adl_trans_mat = np.zeros(
        (n_sexes, n_periods, n_health_states, n_adl_states),
        dtype=float,
    )

    for sex_idx, sex_label in enumerate(specs["sex_labels"]):
        for period in range(n_periods):
            age = start_age + period  # exact age in the table
            for health_idx, health_label in enumerate(
                specs["health_labels_three"][:-1]
            ):
                for adl_idx, adl_label in enumerate(specs["adl_labels"]):
                    prob = adl_trans_df.loc[
                        (adl_trans_df["sex"] == sex_label)
                        & (adl_trans_df["age"] == age)
                        & (adl_trans_df["health"] == health_label)
                        & (adl_trans_df["adl_cat"] == adl_label),
                        "transition_prob",
                    ].values[0]

                    adl_trans_mat[sex_idx, period, health_idx, adl_idx] = prob

    # NB: if you want an absorbing ADL state for death
    #     (i.e. health == 'Death' ⇒ ADL == 'No ADL' with prob 1)
    #     uncomment the next line:
    # adl_trans_mat[:, :, -1, 0] = 1.0

    return jnp.asarray(adl_trans_mat)


def read_in_adl_transition_specs_binary(adl_trans_df, specs):
    """
    Same interface as *read_in_adl_transition_specs* but with a binary
    ADL dimension:
        adl_state 0  → 'No ADL'
        adl_state 1  → 'Any ADL'  (ADL 1 + ADL 2 + ADL 3)

    Returns
    -------
    jax.numpy.ndarray
        Shape = (n_sexes, n_periods, n_health_alive, 2)
    """

    # ──────────────────────────────────────────────────────────────────
    # sizes
    # ──────────────────────────────────────────────────────────────────
    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]
    n_periods = end_age - start_age + 1
    n_sexes = len(specs["sex_labels"])
    n_health_alive = len(specs["health_labels_three"]) - 1  # drop 'Death'
    n_adl_states = 2  # No ADL | Any ADL

    adl_trans_mat = np.zeros(
        (n_sexes, n_periods, n_health_alive, n_adl_states),
        dtype=float,
    )

    # ──────────────────────────────────────────────────────────────────
    # fill cube
    # ──────────────────────────────────────────────────────────────────
    for sex_idx, sex_lbl in enumerate(specs["sex_labels"]):
        for period in range(n_periods):
            age = start_age + period
            for h_idx, h_lbl in enumerate(specs["health_labels_three"][:-1]):  # alive
                # --- probability of *No ADL* -----------------------
                p_no = adl_trans_df.loc[
                    (adl_trans_df["sex"] == sex_lbl)
                    & (adl_trans_df["age"] == age)
                    & (adl_trans_df["health"] == h_lbl)
                    & (adl_trans_df["adl_cat"] == "No ADL"),
                    "transition_prob",
                ].values[0]

                # --- probability of *Any ADL* = ADL1+ADL2+ADL3 ----
                p_any = adl_trans_df.loc[
                    (adl_trans_df["sex"] == sex_lbl)
                    & (adl_trans_df["age"] == age)
                    & (adl_trans_df["health"] == h_lbl)
                    & (adl_trans_df["adl_cat"].isin(["ADL 1", "ADL 2", "ADL 3"])),
                    "transition_prob",
                ].sum()

                adl_trans_mat[sex_idx, period, h_idx, 0] = p_no
                adl_trans_mat[sex_idx, period, h_idx, 1] = p_any

    # ──────────────────────────────────────────────────────────────────
    # sanity check: rows must sum to 1 (within numerical tolerance)
    # ──────────────────────────────────────────────────────────────────
    row_sums = adl_trans_mat.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise ValueError(
            "ADL transition rows do not sum to 1 after collapsing to binary states."
        )

    return jnp.asarray(adl_trans_mat)


def read_in_care_supply_transition_specs(exog_care_df, specs):
    """
    Build a 4-d array of exogenous-care supply probabilities

        [sex, period(age), has_sister, education]

    Parameters
    ----------
    exog_care_df : pandas.DataFrame
        Long table with columns
        ['sex', 'age', 'has_sister', 'education', 'exog_care_prob']
        • sex          : 1 = Men, 2 = Women           (numeric)
        • has_sister   : 0 / 1
        • education    : 0 = Low, 1 = High
    specs : dict
        Master specification dictionary.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape
           (n_sexes, n_periods, 2, n_education_types)

    """

    start_age = specs["start_age"]  # max year in SOEP-IS sample
    end_age = 69  # max year in SOEP-IS sample
    n_periods = end_age - start_age + 1

    n_sister = 2
    n_edu = len(specs["education_labels"])

    care_trans_mat = np.zeros((n_periods, n_sister, n_edu), dtype=float)

    parent_sex = MOTHER
    for period in range(n_periods):
        caregiver_age = start_age + period
        for sister_idx in range(n_sister):
            for edu_idx in range(n_edu):
                prob = exog_care_df.loc[
                    (exog_care_df["sex"] == parent_sex)
                    & (exog_care_df["age"] == caregiver_age)
                    & (exog_care_df["has_sister"] == sister_idx)
                    & (exog_care_df["education"] == edu_idx),
                    "exog_care_prob",
                ].values[0]

                care_trans_mat[period, sister_idx, edu_idx] = prob

    return jnp.asarray(care_trans_mat)


def read_in_mother_age_diff_specs(sample):
    """Read in mother age difference specifications."""

    mother_cohort = sample.loc[
        (sample["gebjahr"] >= INITIAL_CONDITIONS_COHORT_LOW)
        & (sample["gebjahr"] <= INITIAL_CONDITIONS_COHORT_HIGH)
        & (sample["age"] >= INITIAL_CONDITIONS_AGE_LOW)
        & (sample["age"] <= INITIAL_CONDITIONS_AGE_HIGH)
    ]
    mother_age_diff_means = (
        mother_cohort.groupby(["has_sister", "education"])["mother_age_diff"]
        .mean()  # mean within each (education, has_sister) cell
        .unstack()  # 2×2 matrix: rows → education, cols → has_sister
        # .rename(columns={0: "no_sister", 1: "has_sister"})
        # .sort_idex()  # optional: sort education levels
        .reindex(index=[0, 1], columns=[0, 1])
    )
    # _mother_cohort = sample.loc[(sample["syear"] == 2010)]
    # _mother_age_diff_means = (
    #     _mother_cohort.groupby(["has_sister", "education"])["mother_age_diff"]
    #     .mean()  # mean within each (education, has_sister) cell
    #     .unstack()  # 2×2 matrix: rows → education, cols → has_sister
    #     # .rename(columns={0: "no_sister", 1: "has_sister"})
    #     # .sort_idex()  # optional: sort education levels
    #     .reindex(index=[0, 1], columns=[0, 1])
    # )
    # b = jnp.asarray(_mother_age_diff_means.values).astype(jnp.uint8)

    a = jnp.asarray(mother_age_diff_means.values).astype(jnp.uint8)

    return a
