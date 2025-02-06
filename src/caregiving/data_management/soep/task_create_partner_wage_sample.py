"""Create sample for estimation of partner's wage."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import filter_data
from caregiving.data_management.soep.variables import (
    create_education_type,
    create_partner_state,
)
from caregiving.model.shared import N_MONTHS, N_WEEKS_IN_YEAR, PART_TIME, WORK
from caregiving.specs.derive_specs import read_and_derive_specs


def task_create_partner_wage_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_partner_wage_data_raw.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_partner_wage_data.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_raw, index_col=["pid", "syear"])

    df = filter_data(df, specs)
    df = create_wages(df.copy())

    # Drop singles
    df = df[df["parid"] >= 0]
    print(str(len(df)) + " observations after dropping singles.")

    df = create_education_type(df)

    # Create partner state and drop if partner is absent or in non-working age
    df = create_partner_state(df)
    df = df[df["partner_state"] == 1]

    df.reset_index(inplace=True)
    df.to_csv(path_to_save)


def create_wages(df):
    df.rename(columns={"pglabgro": "wage"}, inplace=True)
    df.loc[df["wage"] < 0, "wage"] = 0
    print(str(len(df)) + " observations after dropping invalid wage values.")
    return df
