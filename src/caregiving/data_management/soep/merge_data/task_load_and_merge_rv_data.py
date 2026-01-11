"""Merge SOEP-RV modules."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC


def task_load_and_merge_rv_data(
    rv_fix: Path = SRC / "data" / "soep_rv_vskt" / "2022.fix.1-0.dta",
    rv_var: Path = SRC / "data" / "soep_rv_vskt" / "2022.var.1-0.dta",
    path_to_save: Annotated[Path, Product] = BLD / "data" / "rv_raw.csv",
) -> None:
    """Merge SOEP-RV modules."""

    fix_data = pd.read_stata(
        rv_fix,
        columns=[
            "rv_id",
            "GEVS",  # Geschlecht: 1 männlich, 2 weiblich
            "GBJAVS",  # Geburtsjahr
        ],
        convert_categoricals=False,
    )
    var_data = pd.read_stata(rv_var, convert_categoricals=False)

    rv_data = fix_data.merge(
        var_data,
        on="rv_id",
        how="left",
        indicator="_merge_vskt_var",
        validate="one_to_many",
    )

    # Keep only observations that have "both" in the merge indicator
    # Check if the merge indicator has only "both" values
    if not rv_data["_merge_vskt_var"].isin(["both"]).all():
        raise ValueError(
            "Merge indicator '_merge_vskt_var' contains values other than 'both'."
        )
    # df = df[df["_merge_vskt_var"] == "both"]

    # # Set index
    # rv_data.set_index(["pid", "syear"], inplace=True)
    # print(str(len(rv_data)) + " observations in RV VSKT.")

    rv_data.to_csv(path_to_save)
