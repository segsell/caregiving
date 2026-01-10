"""Create SHARE moments and variances for MSM estimation."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.share.task_create_parent_child_data_set import (
    AGE_BINS_PARENTS,
    AGE_LABELS_PARENTS,
    CARE_COLS,
    weighted_shares_and_counts,
)

DEGREES_OF_FREEDOM = 1


@pytask.mark.cip
def task_add_share_moments(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_share_parent_child_sample: Path = BLD
    / "data"
    / "share_parent_child_data.csv",
    path_to_soep_moments: Path = BLD / "moments" / "soep_moments_new.csv",
    path_to_soep_variances: Path = BLD / "moments" / "soep_variances_new.csv",
    path_to_save_full_moments: Annotated[Path, Product] = BLD
    / "moments"
    / "moments_full.csv",
    path_to_save_full_variances: Annotated[Path, Product] = BLD
    / "moments"
    / "variances_full.csv",
) -> None:
    """Create moments for MSM estimation."""

    soep_moments = pd.read_csv(path_to_soep_moments, index_col=[0])
    soep_variances = pd.read_csv(path_to_soep_variances, index_col=[0])

    share_moments = {}
    share_variances = {}

    dat = pd.read_csv(path_to_share_parent_child_sample, index_col=[0])
    dat = dat[dat["sex"] == 1].copy()  # Mothers only

    dat["age_group"] = pd.cut(
        dat["age"], bins=AGE_BINS_PARENTS, labels=AGE_LABELS_PARENTS, right=False
    )

    shares_weighted_hh = weighted_shares_and_counts(
        dat,
        care_cols=CARE_COLS,
        weight_col="hh_weight",
        group_cols=["age_group"],
    )
    variances_weighted_hh = weighted_shares_and_counts(
        dat,
        care_cols=CARE_COLS,
        weight_col="hh_weight",
        group_cols=["age_group"],
        show_variances=True,
    )

    # shares_w_ind = weighted_shares_and_counts(
    #     dat,
    #     CARE_COLS,
    #     weight_col="ind_weight",
    #     group_cols=["sex", "age_group"],
    # )
    # shares_w_design = weighted_shares_and_counts(
    #     dat, CARE_COLS, weight_col="design_weight", group_cols=["sex", "age_group"]
    # )

    # shares_child_w_hh = weighted_shares_and_counts(
    #     dat,
    #     care_cols=CHILD_CARE_COLS,
    #     weight_col="hh_weight",
    #     group_cols=["sex", "age_group"],
    #     show_counts=True,
    # )
    # shares_child_w_ind = weighted_shares_and_counts(
    #     dat, CHILD_CARE_COLS, weight_col="ind_weight", group_cols=["sex", "age_group"]
    # )
    # shares_child_w_design = weighted_shares_and_counts(
    #     dat,
    #     CHILD_CARE_COLS,
    #     weight_col="design_weight",
    #     group_cols=["sex", "age_group"],
    # )

    # ==================================================================================
    _strip_suffix = "_general"
    for col in shares_weighted_hh.columns:
        for age_group, val in shares_weighted_hh[col].items():
            col_stripped = col.replace(_strip_suffix, "")
            share_moments[f"{col_stripped}_age_bin_{age_group}"] = val

    for col in variances_weighted_hh.columns:
        for age_group, val in variances_weighted_hh[col].items():
            col_stripped = col.replace(_strip_suffix, "")
            share_variances[f"{col_stripped}_age_bin_{age_group}"] = val

    # moments = pd.concat(
    #     [soep_moments, pd.Series(share_moments, name="value").to_frame()]
    # )
    # variances = pd.concat(
    #     [soep_variances, pd.Series(share_variances, name="value").to_frame()]
    # )
    # ==================================================================================

    # Save
    soep_moments.to_csv(path_to_save_full_moments)
    soep_variances.to_csv(path_to_save_full_variances)
    # moments.to_csv(path_to_save_full_moments)
    # variances.to_csv(path_to_save_full_variances)
