"""Estimate the parameters of the partner transition matrix.

For each sex, education level, and age bin (10-year intervals), the function estimates
the conditional probability P(partner_state | lagged_partner_state) non-parametrically.

The partner state is a binary variable indicating whether the partner is present or not.
The lagged partner state is a binary variable indicating whether the partner was present
in the previous year.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.specs.derive_specs import read_and_derive_specs


def task_estimate_partner_transitions(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "partner_transition_matrix.csv",
) -> None:
    """Estimate the partner state transition matrix non-parametrically."""

    # Read specs and data; restrict to ages below end_age
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])
    df = df.loc[df["age"] <= specs["end_age"]].copy()

    # Create age bins (10-year intervals)
    df["age_bin"] = np.floor(df["age"] / 10) * 10

    # Define covariates for grouping (including dummy partner state columns)
    cov_list = ["sex", "education", "age_bin", "partner_state_1.0", "partner_state_2.0"]

    # Create dummy variables for 'partner_state'
    df = pd.get_dummies(df, columns=["partner_state"])

    # Group and compute the conditional probabilities, renaming the resulting Series
    trans_mat_series = (
        df.groupby(cov_list)["lead_partner_state"]
        .value_counts(normalize=True)
        .rename("proportion")
    )

    # Convert the Series with a MultiIndex to a DataFrame
    df = trans_mat_series.reset_index()

    # Vectorized mapping to compute lagged_partner_state:
    # We assume that the dummy columns contain booleans.
    # Mapping: (False, False) -> 0, (True, False) -> 1, (False, True) -> 2.
    # (If both are True, this will produce 3; if that situation should never occur,
    # it is fine.)
    df["lagged_partner_state"] = df["partner_state_1.0"].astype(int) + 2 * df[
        "partner_state_2.0"
    ].astype(int)

    # Clean up
    df.drop(columns=["partner_state_1.0", "partner_state_2.0"], inplace=True)

    # Set the new multi-index in the desired order:
    # sex, education, age_bin, lagged_partner_state, lead_partner_state.
    df = df.set_index(
        ["sex", "education", "age_bin", "lagged_partner_state", "lead_partner_state"]
    )
    df = df.rename(columns={"lead_partner_state": "partner_state"})

    # Optionally, sort the index
    df.sort_index(inplace=True)

    # Save the resulting DataFrame to CSV
    df.to_csv(path_to_save)


def task_estimate_number_of_children_in_household(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_data: Path = BLD / "data" / "soep_partner_transition_data.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "estimation"
    / "stochastic_processes"
    / "nb_children_estimates.csv",
) -> None:
    """Estimate the number of children in the household vio OLS.

    Conditiononal on sex, education, and age bin."""

    # Read specs and data; restrict to ages below end_age
    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_data, index_col=["pid", "syear"])

    start_age = specs["start_age"]
    end_age = specs["end_age_children_in_household"]

    df = df.loc[(df["age"] >= start_age) & (df["age"] <= end_age)]

    df["period"] = df["age"] - start_age
    df["period_sq"] = df["period"] ** 2
    df["has_partner"] = (df["partner_state"] > 0).astype(int)
    # estimate OLS for each combination of sex, education and has_partner

    edu_states = list(range(specs["n_education_types"]))
    sexes = [0, 1]
    partner_states = [0, 1]

    sub_group_names = ["sex", "education", "has_partner"]

    multiindex = pd.MultiIndex.from_product(
        [sexes, edu_states, partner_states],
        names=sub_group_names,
    )

    columns = ["const", "period", "period_sq"]
    estimates = pd.DataFrame(index=multiindex, columns=columns)
    for sex in sexes:
        for education in edu_states:
            for has_partner in partner_states:
                df_reduced = df[
                    (df["sex"] == sex)
                    & (df["education"] == education)
                    & (df["has_partner"] == has_partner)
                ]
                X = df_reduced[columns[1:]]
                X = sm.add_constant(X)
                Y = df_reduced["children"]
                model = sm.OLS(Y, X).fit()
                estimates.loc[(sex, education, has_partner), columns] = model.params

    estimates.to_csv(path_to_save)
