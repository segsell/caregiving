"""Inheritance probability precomputation functions."""

from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd


def read_in_inheritance_prob_specs(specs, path_to_save: Optional[Path] = None):
    """Precompute inheritance probability matrix by age, education, and care type.

    Builds a matrix with columns for no care, light care, intensive care, and formal care.
    Since the probability model uses any_care (binary), light and intensive care
    will have the same probability value (any_care = 1).

    Parameters
    ----------
    specs : dict
        Master spec-dictionary containing:
        - inheritance_prob_spec5_params: DataFrame with logit parameters
        - start_age: Starting age
        - sex_labels: Sex labels for parameter lookup
        - education_labels: Education labels
    path_to_save : Optional[Path]
        Optional path to save the matrix as a CSV file.
        If provided, saves in long format with columns:
        ['sex', 'age', 'education', 'care_type', 'prob_positive_inheritance'].

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_sexes, n_periods, n_education, 4)
        where the last dimension is: [no_care, formal_care, light_care, intensive_care]
    """
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    n_periods = end_age - start_age + 1
    n_sexes = len(specs["sex_labels"])  # 2
    n_education = len(specs["education_labels"])  # 2
    n_care_types = 4  # no_care, formal_care, light_care, intensive_care

    inheritance_prob_mat = np.zeros(
        (n_sexes, n_periods, n_education, n_care_types),
        dtype=float,
    )

    # Get inheritance probability parameters
    # Filter out rows with _se and _rsq suffixes (standard errors and R-squared)
    inheritance_prob_params_raw = specs["inheritance_prob_params"]
    inheritance_prob_params = inheritance_prob_params_raw[
        ~(
            inheritance_prob_params_raw.index.str.endswith("_se")
            | inheritance_prob_params_raw.index.str.endswith("_rsq")
        )
    ].copy()

    for sex_idx, sex_label in enumerate(specs["sex_labels"]):
        for period in range(n_periods):
            age = start_age + period
            age_sq = age**2

            for edu_idx in range(n_education):
                # Get parameters for this sex
                params = inheritance_prob_params.loc[sex_label]

                # Get formal care coefficient (default to 0 if not present for backward compatibility)
                if "formal_care_costs_dummy" in params.index:
                    formal_care_coef = params["formal_care_costs_dummy"]
                elif "formal_care_costs_dummy_recent" in params.index:
                    formal_care_coef = params["formal_care_costs_dummy_recent"]
                else:
                    formal_care_coef = 0

                # Get any_care coefficient (could be any_care or any_care_recent)
                if "any_care" in params.index:
                    any_care_coef = params["any_care"]
                elif "any_care_recent" in params.index:
                    any_care_coef = params["any_care_recent"]
                else:
                    any_care_coef = 0

                # Compute logit linear predictor for no care
                # (any_care = 0, formal_care = 0)
                logit_linear_no_care = (
                    params["age"] * age
                    + params["age_sq"] * age_sq
                    + any_care_coef * 0
                    + formal_care_coef * 0
                    + params["education"] * edu_idx
                    + params["const"]
                )
                prob_no_care = 1.0 / (1.0 + np.exp(-logit_linear_no_care))

                # Compute logit linear predictor for care (any_care = 1, formal_care = 0)
                # Note: light and intensive care use the same value since
                # the model uses any_care (binary), not separate light/intensive
                logit_linear_care = (
                    params["age"] * age
                    + params["age_sq"] * age_sq
                    + any_care_coef * 1
                    + formal_care_coef * 0
                    + params["education"] * edu_idx
                    + params["const"]
                )
                prob_care = 1.0 / (1.0 + np.exp(-logit_linear_care))

                # Compute logit linear predictor for formal care (any_care = 0, formal_care = 1)
                logit_linear_formal_care = (
                    params["age"] * age
                    + params["age_sq"] * age_sq
                    + any_care_coef * 0
                    + formal_care_coef * 1
                    + params["education"] * edu_idx
                    + params["const"]
                )
                prob_formal_care = 1.0 / (1.0 + np.exp(-logit_linear_formal_care))

                # Store probabilities
                # Index 0: no_care, Index 1: formal_care, Index 2: light_care, Index 3: intensive_care
                inheritance_prob_mat[sex_idx, period, edu_idx, 0] = prob_no_care
                inheritance_prob_mat[sex_idx, period, edu_idx, 1] = prob_formal_care
                inheritance_prob_mat[sex_idx, period, edu_idx, 2] = prob_care  # light
                inheritance_prob_mat[sex_idx, period, edu_idx, 3] = (
                    prob_care  # intensive
                )

    # Save to CSV if path provided
    if path_to_save is not None:
        ages = np.arange(start_age, end_age + 1)
        sex_labels = specs["sex_labels"]
        education_labels = specs["education_labels"]
        care_type_labels = ["no_care", "formal_care", "light_care", "intensive_care"]

        rows = []
        for sex_idx, sex_label in enumerate(sex_labels):
            for period, age in enumerate(ages):
                for edu_idx, edu_label in enumerate(education_labels):
                    for care_idx, care_label in enumerate(care_type_labels):
                        prob = inheritance_prob_mat[sex_idx, period, edu_idx, care_idx]
                        rows.append(
                            {
                                "sex": sex_label,
                                "age": age,
                                "education": edu_label,
                                "care_type": care_label,
                                "prob_positive_inheritance": prob,
                            }
                        )
        inheritance_prob_df = pd.DataFrame(rows)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        inheritance_prob_df.to_csv(path_to_save, index=False)

    return jnp.asarray(inheritance_prob_mat)


def read_in_inheritance_amount_specs(specs, path_to_save: Optional[Path] = None):
    """Precompute inheritance amount matrix by age, education, and care type.

    Builds a matrix with columns for no care, light care, intensive care, and formal care.
    The amount model distinguishes between light_care_recent and intensive_care_recent,
    so the care types will have different values.

    Parameters
    ----------
    specs : dict
        Master spec-dictionary containing:
        - inheritance_amount_spec5_params: DataFrame with OLS parameters
        - start_age: Starting age
        - sex_labels: Sex labels for parameter lookup
        - education_labels: Education labels
    path_to_save : Optional[Path]
        Optional path to save the matrix as a CSV file.
        If provided, saves in long format with columns:
        ['sex', 'age', 'education', 'care_type', 'inheritance_amount'].

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_sexes, n_periods, n_education, 4)
        where the last dimension is: [no_care, formal_care, light_care, intensive_care]
    """
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    n_periods = end_age - start_age + 1
    n_sexes = len(specs["sex_labels"])  # 2
    n_education = len(specs["education_labels"])  # 2
    n_care_types = 4  # no_care, formal_care, light_care, intensive_care

    inheritance_amount_mat = np.zeros(
        (n_sexes, n_periods, n_education, n_care_types),
        dtype=float,
    )

    # Get inheritance amount parameters
    # Filter out rows with _se and _rsq suffixes (standard errors and R-squared)
    inheritance_amount_params_raw = specs["inheritance_amount_params"]
    inheritance_amount_params = inheritance_amount_params_raw[
        ~(
            inheritance_amount_params_raw.index.str.endswith("_se")
            | inheritance_amount_params_raw.index.str.endswith("_rsq")
        )
    ].copy()

    for sex_idx, sex_label in enumerate(specs["sex_labels"]):
        for period in range(n_periods):
            age = start_age + period
            age_sq = age**2

            for edu_idx in range(n_education):
                # Get parameters for this sex
                params = inheritance_amount_params.loc[sex_label]

                # Get formal care coefficient (default to 0 if not present for backward compatibility)
                if "formal_care_costs_dummy" in params.index:
                    formal_care_coef = params["formal_care_costs_dummy"]
                elif "formal_care_costs_dummy_recent" in params.index:
                    formal_care_coef = params["formal_care_costs_dummy_recent"]
                else:
                    formal_care_coef = 0

                # Get care coefficients (could be light_care_recent/intensive_care_recent or variants)
                if "light_care_recent" in params.index:
                    light_care_coef = params["light_care_recent"]
                elif "light_care" in params.index:
                    light_care_coef = params["light_care"]
                else:
                    light_care_coef = 0

                if "intensive_care_recent" in params.index:
                    intensive_care_coef = params["intensive_care_recent"]
                elif "intensive_care" in params.index:
                    intensive_care_coef = params["intensive_care"]
                else:
                    intensive_care_coef = 0

                # Compute OLS linear predictor for no care
                # (light_care_recent = 0, intensive_care_recent = 0, formal_care = 0)
                ln_inheritance_amount_no_care = (
                    params["age"] * age
                    + params["age_sq"] * age_sq
                    + light_care_coef * 0
                    + intensive_care_coef * 0
                    + formal_care_coef * 0
                    + params["education"] * edu_idx
                    + params["const"]
                )
                amount_no_care = np.exp(ln_inheritance_amount_no_care)

                # Compute OLS linear predictor for light care
                # (light_care_recent = 1, intensive_care_recent = 0, formal_care = 0)
                ln_inheritance_amount_light = (
                    params["age"] * age
                    + params["age_sq"] * age_sq
                    + light_care_coef * 1
                    + intensive_care_coef * 0
                    + formal_care_coef * 0
                    + params["education"] * edu_idx
                    + params["const"]
                )
                amount_light = np.exp(ln_inheritance_amount_light)

                # Compute OLS linear predictor for intensive care
                # (light_care_recent = 0, intensive_care_recent = 1, formal_care = 0)
                ln_inheritance_amount_intensive = (
                    params["age"] * age
                    + params["age_sq"] * age_sq
                    + light_care_coef * 0
                    + intensive_care_coef * 1
                    + formal_care_coef * 0
                    + params["education"] * edu_idx
                    + params["const"]
                )
                amount_intensive = np.exp(ln_inheritance_amount_intensive)

                # Compute OLS linear predictor for formal care
                # (light_care_recent = 0, intensive_care_recent = 0, formal_care = 1)
                ln_inheritance_amount_formal_care = (
                    params["age"] * age
                    + params["age_sq"] * age_sq
                    + light_care_coef * 0
                    + intensive_care_coef * 0
                    + formal_care_coef * 1
                    + params["education"] * edu_idx
                    + params["const"]
                )
                amount_formal_care = np.exp(ln_inheritance_amount_formal_care)

                # Store amounts
                # Index 0: no_care, Index 1: formal_care, Index 2: light_care, Index 3: intensive_care
                inheritance_amount_mat[sex_idx, period, edu_idx, 0] = amount_no_care
                inheritance_amount_mat[sex_idx, period, edu_idx, 1] = amount_formal_care
                inheritance_amount_mat[sex_idx, period, edu_idx, 2] = amount_light
                inheritance_amount_mat[sex_idx, period, edu_idx, 3] = amount_intensive

    # Save to CSV if path provided
    if path_to_save is not None:
        ages = np.arange(start_age, end_age + 1)
        sex_labels = specs["sex_labels"]
        education_labels = specs["education_labels"]
        care_type_labels = ["no_care", "formal_care", "light_care", "intensive_care"]

        rows = []
        for sex_idx, sex_label in enumerate(sex_labels):
            for period, age in enumerate(ages):
                for edu_idx, edu_label in enumerate(education_labels):
                    for care_idx, care_label in enumerate(care_type_labels):
                        amount = inheritance_amount_mat[
                            sex_idx, period, edu_idx, care_idx
                        ]
                        rows.append(
                            {
                                "sex": sex_label,
                                "age": age,
                                "education": edu_label,
                                "care_type": care_label,
                                "inheritance_amount": amount,
                            }
                        )
        inheritance_amount_df = pd.DataFrame(rows)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        inheritance_amount_df.to_csv(path_to_save, index=False)

    return jnp.asarray(inheritance_amount_mat)
