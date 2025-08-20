"""Create sample for partner transition."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    filter_below_age,
    filter_years,
    recode_sex,
    span_dataframe,
)
from caregiving.data_management.soep.variables import (
    create_education_type,
    create_kidage_youngest,
    create_partner_state,
)
from caregiving.specs.derive_specs import read_and_derive_specs
from caregiving.utils import table


def task_create_partner_transition_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_partner_transition_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_partner_transition_data.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_raw, index_col=["pid", "syear"])

    df = create_education_type(df)

    # Filter estimation years
    df = filter_years(df, specs["start_year"], specs["end_year"])

    # In this function also merging is called
    df = _create_partner_and_lagged_state(df, specs)
    df = create_kidage_youngest(df)

    # Filter age and sex
    df = filter_below_age(df, specs["start_age"])
    df = recode_sex(df)

    df = df[
        [
            "age",
            "sex",
            "education",
            "partner_state",
            "lead_partner_state",
            "children",
            "kidage_youngest",
        ]
    ]

    print(
        str(len(df))
        + " observations in the final partner transition sample.  \n ----------------"
    )

    df.to_csv(path_to_save)


def _create_partner_and_lagged_state(df, specs):
    # The following code is dependent on span dataframe being called first.
    # In particular the lagged partner state must be after span dataframe and
    # create partner state.
    # We should rewrite this
    df = span_dataframe(df, specs["start_year"], specs["end_year"])

    df = create_partner_state(df)
    df["lead_partner_state"] = df.groupby(["pid"])["partner_state"].shift(-1)
    df = df[df["lead_partner_state"].notna()]
    df = df[df["partner_state"].notna()]
    print(
        str(len(df))
        + " obs. after dropping people with a partner whose choice is not observed."
    )
    return df
