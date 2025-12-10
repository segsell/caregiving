"""Create transition matrices for care demand and care supply."""

from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd

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


def read_in_adl_state_transition_specs(
    adl_state_trans_df, specs, path_to_save: Optional[Path] = None
):
    """
    Build a 4-d transition array
        [sex, period (age), adl_lag, adl_next]
    out of the long DataFrame *adl_state_transitions*.

    This is a simpler ADL transition matrix that doesn't condition on health,
    only on sex, age, and lagged ADL state.

    Parameters
    ----------
    adl_state_trans_df : pandas.DataFrame
        Long table with columns
        ['sex', 'age', 'adl_lag', 'adl_next', 'transition_prob'].
        • sex: 'Men' or 'Women'
        • age: integer age
        • adl_lag: 'No ADL', 'ADL 1', 'ADL 2', 'ADL 3'
        • adl_next: 'No ADL', 'ADL 1', 'ADL 2', 'ADL 3'
        • transition_prob: probability of transitioning from adl_lag to adl_next
    specs : dict
        Master spec-dictionary that already contains the label lists
        used elsewhere in your model code.
    path_to_save : Optional[Path]
        Optional path to save the transition matrix as a CSV file.
        If provided, saves in long format with columns:
        ['sex', 'age', 'adl_lag', 'adl_next', 'transition_prob'].

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_sexes, n_periods, n_adl_states, n_adl_states)
    """
    # unpack sizes straight from *specs*
    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]
    n_periods = end_age - start_age + 1
    n_sexes = len(specs["sex_labels"])  # 2
    n_adl_states = len(specs["adl_labels"])  # 4

    adl_state_trans_mat = np.zeros(
        (n_sexes, n_periods, n_adl_states, n_adl_states),
        dtype=float,
    )

    for sex_idx, sex_label in enumerate(specs["sex_labels"]):
        for period in range(n_periods):
            age = start_age + period  # exact age in the table
            for adl_lag_idx, adl_lag_label in enumerate(specs["adl_labels"]):
                for adl_next_idx, adl_next_label in enumerate(specs["adl_labels"]):
                    prob = adl_state_trans_df.loc[
                        (adl_state_trans_df["sex"] == sex_label)
                        & (adl_state_trans_df["age"] == age)
                        & (adl_state_trans_df["adl_lag"] == adl_lag_label)
                        & (adl_state_trans_df["adl_next"] == adl_next_label),
                        "transition_prob",
                    ].values[0]

                    adl_state_trans_mat[sex_idx, period, adl_lag_idx, adl_next_idx] = (
                        prob
                    )

    # ──────────────────────────────────────────────────────────────────
    # sanity check: rows must sum to 1 (within numerical tolerance)
    # ──────────────────────────────────────────────────────────────────
    row_sums = adl_state_trans_mat.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise ValueError(
            "ADL state transition rows do not sum to 1. "
            "Check transition probabilities."
        )

    # Save to CSV if path provided
    if path_to_save is not None:
        start_age = specs["start_age_parents"]
        end_age = specs["end_age"]
        ages = np.arange(start_age, end_age + 1)
        sex_labels = specs["sex_labels"]
        adl_labels = specs["adl_labels"]

        rows = []
        for sex_idx, sex_label in enumerate(sex_labels):
            for period, age in enumerate(ages):
                for adl_lag_idx, adl_lag_label in enumerate(adl_labels):
                    for adl_next_idx, adl_next_label in enumerate(adl_labels):
                        prob = adl_state_trans_mat[
                            sex_idx, period, adl_lag_idx, adl_next_idx
                        ]
                        rows.append(
                            {
                                "sex": sex_label,
                                "age": age,
                                "adl_lag": adl_lag_label,
                                "adl_next": adl_next_label,
                                "transition_prob": prob,
                            }
                        )
        adl_state_trans_df_save = pd.DataFrame(rows)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        adl_state_trans_df_save.to_csv(path_to_save, index=False)

    return jnp.asarray(adl_state_trans_mat)


def read_in_adl_state_transition_specs_light_intensive(
    adl_state_trans_df, specs, path_to_save: Optional[Path] = None
):
    """
    Build a 4-d transition array with collapsed ADL categories
        [sex, period (age), adl_lag, adl_next]
    where ADL categories are:
        - 0: 'No ADL'
        - 1: 'ADL 1'
        - 2: 'ADL 2' or 'ADL 3' (collapsed)

    This is similar to read_in_adl_state_transition_specs but collapses
    ADL 2 and ADL 3 into a single "intensive" category.

    Parameters
    ----------
    adl_state_trans_df : pandas.DataFrame
        Long table with columns
        ['sex', 'age', 'adl_lag', 'adl_next', 'transition_prob'].
        • sex: 'Men' or 'Women'
        • age: integer age
        • adl_lag: 'No ADL', 'ADL 1', 'ADL 2', 'ADL 3'
        • adl_next: 'No ADL', 'ADL 1', 'ADL 2', 'ADL 3'
        • transition_prob: probability of transitioning from adl_lag to adl_next
    specs : dict
        Master spec-dictionary that already contains the label lists
        used elsewhere in your model code.
    path_to_save : Optional[Path]
        Optional path to save the transition matrix as a CSV file.
        If provided, saves in long format with columns:
        ['sex', 'age', 'adl_lag', 'adl_next', 'transition_prob'].

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_sexes, n_periods, 3, 3)
        where the last two dimensions are:
        - 0: No ADL
        - 1: ADL 1
        - 2: ADL 2 or ADL 3 (intensive)
    """
    # unpack sizes straight from *specs*
    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]
    n_periods = end_age - start_age + 1
    n_sexes = len(specs["sex_labels"])  # 2
    n_adl_states_collapsed = 3  # No ADL, ADL 1, ADL 2+3

    adl_state_trans_mat = np.zeros(
        (n_sexes, n_periods, n_adl_states_collapsed, n_adl_states_collapsed),
        dtype=float,
    )

    # Mapping from original ADL labels to collapsed indices
    # 0: 'No ADL' -> 0
    # 1: 'ADL 1' -> 1
    # 2: 'ADL 2' -> 2
    # 3: 'ADL 3' -> 2
    adl_label_to_collapsed_idx = {
        "No ADL": 0,
        "ADL 1": 1,
        "ADL 2": 2,
        "ADL 3": 2,
    }

    for sex_idx, sex_label in enumerate(specs["sex_labels"]):
        for period in range(n_periods):
            age = start_age + period  # exact age in the table
            for _adl_lag_idx, adl_lag_label in enumerate(specs["adl_labels"]):
                # Map lag ADL to collapsed index
                lag_collapsed_idx = adl_label_to_collapsed_idx[adl_lag_label]

                for _adl_next_idx, adl_next_label in enumerate(specs["adl_labels"]):
                    # Map next ADL to collapsed index
                    next_collapsed_idx = adl_label_to_collapsed_idx[adl_next_label]

                    # Get probability from original DataFrame
                    prob = adl_state_trans_df.loc[
                        (adl_state_trans_df["sex"] == sex_label)
                        & (adl_state_trans_df["age"] == age)
                        & (adl_state_trans_df["adl_lag"] == adl_lag_label)
                        & (adl_state_trans_df["adl_next"] == adl_next_label),
                        "transition_prob",
                    ].values[0]

                    # Aggregate probabilities into collapsed categories by addition:
                    # - FROM collapsed state: add transitions from ADL 2 and 3
                    # - TO collapsed state: add transitions to ADL 2 and 3
                    # Example: P(ADL_1 -> ADL_2 or ADL_3) =
                    # P(ADL_1 -> ADL_2) + P(ADL_1 -> ADL_3)
                    adl_state_trans_mat[
                        sex_idx, period, lag_collapsed_idx, next_collapsed_idx
                    ] += prob

    # ──────────────────────────────────────────────────────────────────
    # Normalize rows to ensure they sum to 1
    # Rows from non-collapsed states (No ADL=0, ADL 1=1) should already sum to 1
    # Rows from collapsed state (ADL 2 or 3=2) will sum to 2 (since we summed
    # two original rows that each sum to 1), so we need to normalize them
    # ──────────────────────────────────────────────────────────────────
    row_sums = adl_state_trans_mat.sum(axis=-1, keepdims=True)
    # Avoid division by zero (shouldn't happen, but be safe)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    adl_state_trans_mat = adl_state_trans_mat / row_sums

    # ──────────────────────────────────────────────────────────────────
    # sanity check: rows must sum to 1 (within numerical tolerance)
    # ──────────────────────────────────────────────────────────────────
    row_sums_check = adl_state_trans_mat.sum(axis=-1)
    if not np.allclose(row_sums_check, 1.0, atol=1e-10):
        raise ValueError(
            "ADL state transition rows do not sum to 1 after collapsing and "
            "normalization. Check transition probabilities."
        )

    # Save to CSV if path provided
    if path_to_save is not None:
        start_age = specs["start_age_parents"]
        end_age = specs["end_age"]
        ages = np.arange(start_age, end_age + 1)
        sex_labels = specs["sex_labels"]
        adl_labels_light_intensive = ["No ADL", "ADL 1", "ADL 2 or ADL 3"]

        rows = []
        for sex_idx, sex_label in enumerate(sex_labels):
            for period, age in enumerate(ages):
                for (
                    adl_lag_idx,
                    adl_lag_label,
                ) in enumerate(adl_labels_light_intensive):
                    for (
                        adl_next_idx,
                        adl_next_label,
                    ) in enumerate(adl_labels_light_intensive):
                        prob = adl_state_trans_mat[
                            sex_idx, period, adl_lag_idx, adl_next_idx
                        ]
                        rows.append(
                            {
                                "sex": sex_label,
                                "age": age,
                                "adl_lag": adl_lag_label,
                                "adl_next": adl_next_label,
                                "transition_prob": prob,
                            }
                        )
        adl_state_trans_df_save = pd.DataFrame(rows)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        adl_state_trans_df_save.to_csv(path_to_save, index=False)

    return jnp.asarray(adl_state_trans_mat)


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


def read_in_survival_by_age_specs(survival_df, specs):
    """
    Build a 2-d array of survival probabilities
        [sex, age_index]

    Parameters
    ----------
    survival_df : pandas.DataFrame
        Long table with columns
        ['sex', 'age', 'cum_survival_prob'].
        • sex: 0 = Men, 1 = Women (numeric)
        • age: integer age
        • cum_survival_prob: cumulative survival probability
    specs : dict
        Master spec-dictionary that already contains the label lists
        used elsewhere in your model code.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_sexes, n_ages)
        where first dimension is sex (0=Men, 1=Women) and
        second dimension is age_index (0=min_age, 1=min_age+1, ...)
    """
    # Get unique ages and sort them
    ages = sorted(survival_df["age"].unique())
    min_age = min(ages)
    n_ages = len(ages)
    n_sexes = len(specs["sex_labels"])  # 2

    # Create array: (n_sexes, n_ages) where [sex_idx, age_idx]
    # age_idx = 0 corresponds to min_age, age_idx = 1 to min_age+1, etc.
    survival_mat = np.zeros((n_sexes, n_ages), dtype=float)

    # Create age to index mapping (age_index = age - min_age)
    age_to_idx = {age: age - min_age for age in ages}

    # Map sex: 0 = Men, 1 = Women
    for sex_num in (0, 1):
        sex_data = survival_df[survival_df["sex"] == sex_num]
        for _, row in sex_data.iterrows():
            age = row["age"]
            prob = row["cum_survival_prob"]
            age_idx = age_to_idx[age]
            survival_mat[sex_num, age_idx] = prob

    return jnp.asarray(survival_mat)


def weight_adl_transitions_by_survival(specs):
    """
    Weight ADL state transition probabilities by survival probabilities.

    Dead individuals are treated as ADL category 0. The weighted transitions:
    - For transitions TO ADL 0: (1 - survival_prob) + survival_prob * original_prob
      (includes both dead individuals and alive individuals with no ADL)
    - For transitions TO ADL 1, 2, 3: survival_prob * original_prob
      (only applies to alive population)

    Parameters
    ----------
    specs : dict
        Master spec-dictionary containing:
        - adl_state_transition_mat: unweighted ADL state transition matrix
        - survival_by_age_mat: survival matrix of shape [sex, age_index]
        - survival_min_age: minimum age in survival matrix (age_index 0)
        - start_age_parents, end_age: age range

    Returns
    -------
    jax.numpy.ndarray
        Weighted ADL state transition matrix of same shape as input
    """

    # Extract data from specs
    adl_state_trans_mat = specs["adl_state_transition_mat"]
    survival_mat = specs["survival_by_age_mat"]

    start_age = specs["start_age_parents"]
    end_age = specs["end_age"]

    n_periods = end_age - start_age + 1
    n_sexes = adl_state_trans_mat.shape[0]
    n_adl_states = adl_state_trans_mat.shape[2]

    # Convert to numpy for easier indexing
    adl_mat_np = np.array(adl_state_trans_mat)
    survival_mat_np = np.array(survival_mat)
    survival_min_age = specs["survival_min_age"]
    survival_max_age = survival_min_age + survival_mat_np.shape[1] - 1

    weighted_mat = np.zeros_like(adl_mat_np)

    for sex_idx in range(n_sexes):
        for period in range(n_periods):
            age = start_age + period
            # Get cumulative survival probability at this age
            # (matching the logic in the right panel plotting function)
            if survival_min_age <= age <= survival_max_age:
                age_idx = age - survival_min_age
                survival_prob = survival_mat_np[sex_idx, age_idx]
            else:
                # If age not found, use 0.0 (all dead)
                survival_prob = 0.0

            for adl_lag_idx in range(n_adl_states):
                for adl_next_idx in range(n_adl_states):
                    original_prob = adl_mat_np[
                        sex_idx, period, adl_lag_idx, adl_next_idx
                    ]

                    if adl_next_idx == 0:  # Transition to ADL 0 (No ADL)
                        # Include dead (1 - survival_prob) + alive with no ADL
                        # ADL 0 contains both alive with no ADL and dead
                        weighted_mat[sex_idx, period, adl_lag_idx, adl_next_idx] = (
                            1 - survival_prob
                        ) + survival_prob * original_prob
                    else:  # Transition to ADL 1, 2, 3
                        # Weight by survival prob (only for alive population)
                        weighted_mat[sex_idx, period, adl_lag_idx, adl_next_idx] = (
                            survival_prob * original_prob
                        )

    return jnp.asarray(weighted_mat)


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
