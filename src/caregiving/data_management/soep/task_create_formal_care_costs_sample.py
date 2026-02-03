"""Health transition sample."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import statsmodels.api as sm
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    filter_above_age,
    filter_below_age,
    filter_years,
    recode_sex,
    span_dataframe,
)
from caregiving.data_management.soep.task_create_event_study_sample import (
    create_caregiving,
)
from caregiving.data_management.soep.variables import (
    clean_health_create_states,
    create_education_type,
    create_health_var_good_bad,
    create_health_var_good_medium_bad,
    create_nursing_home,
    deflate_formal_care_costs,
)
from caregiving.model.shared import MAX_AGE_SIM
from caregiving.specs.task_write_specs import read_and_derive_specs

# Filter formal care costs to reasonable range (in euros)
MIN_FORMAL_CARE_COSTS = 100
MAX_FORMAL_CARE_COSTS = 4_000


@pytask.mark.formal_care_costs
def task_create_formal_care_costs_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw_data: Path = BLD / "data" / "soep_formal_care_costs_data_raw.csv",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "formal_care_costs_sample.pkl",
):
    """Create formal care costs sample."""

    specs = read_and_derive_specs(path_to_specs)
    df = pd.read_csv(path_to_raw_data, index_col=["pid", "syear"])
    cpi_data = pd.read_csv(path_to_cpi, index_col=0)

    # Pre-Filter age and sex
    df = filter_below_age(df, specs["start_age"] - specs["health_smoothing_bandwidth"])
    df = filter_above_age(df, specs["end_age"] + specs["health_smoothing_bandwidth"])
    df = recode_sex(df)

    # Create education type
    df = create_education_type(df)

    # create health states
    df = create_health_var_good_bad(df)

    # df = span_dataframe(df, 2001, 2023)
    # df = clean_health_create_states(df)

    # Drop rows where hle0016 < 0 and create new variable formal_care_costs_raw
    df = df[df["hle0016"] >= 0].copy()
    df["formal_care_costs_raw"] = df["hle0016"]

    # CPI adjustment: deflate formal_care_costs
    df = deflate_formal_care_costs(
        df, cpi_data=cpi_data, specs=specs, var_name="formal_care_costs_raw"
    )

    df = df[
        (df["formal_care_costs"] >= MIN_FORMAL_CARE_COSTS)
        & (df["formal_care_costs"] <= MAX_FORMAL_CARE_COSTS)
    ].copy()

    # # Keep only data within 10th and 90th percentile
    # p10 = df["formal_care_costs"].quantile(0.02)
    # p90 = df["formal_care_costs"].quantile(0.98)
    # df = df[(df["formal_care_costs"] >= p10)&(df["formal_care_costs"] <= p90)].copy()

    df.to_pickle(path_to_save)
