"""Create inheritance sample."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytask
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    create_lagged_and_lead_variables,
    enforce_model_choice_restriction,
    filter_below_age,
    filter_data,
    filter_years,
    recode_sex,
)
from caregiving.data_management.soep.soep_variables.experience import (
    create_experience_variable_with_cap,
)
from caregiving.data_management.soep.task_create_event_study_sample import (
    create_caregiving,
    create_parent_info,
    create_sibling_info,
)
from caregiving.data_management.soep.task_create_structural_estimation_sample import (
    add_wealth_data,
    create_alreay_retired_variable,
)
from caregiving.data_management.soep.task_create_wealth_sample import deflate_wealth
from caregiving.data_management.soep.variables import (  # create_experience_variable,
    create_choice_variable,
    create_education_type,
    create_health_var_good_bad,
    create_hh_has_moved,
    create_inheritance,
    create_kidage_youngest,
    create_nursing_home,
    create_partner_state,
    create_policy_state,
    deflate_formal_care_costs,
    determine_observed_job_offers,
    generate_job_separation_var,
)
from caregiving.model.shared import (
    PART_TIME_CHOICES,
    RETIREMENT_CHOICES,
    SOEP_DOES_NOT_APPLY,
    WORK_CHOICES,
)
from caregiving.specs.task_write_specs import read_and_derive_specs


# @pytask.mark.inheritance
def task_create_inheritance_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw: Path = BLD / "data" / "soep_estimation_data_raw.csv",
    path_to_wealth: Path = BLD / "data" / "soep_wealth_data.csv",
    path_to_cpi: Path = SRC / "data" / "statistical_office" / "cpi_germany.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_inheritance_sample.csv",
) -> None:
    """Create inheritance sample.

    Loads raw SOEP data and performs basic preparation for inheritance estimation.
    """
    # specs = read_and_derive_specs(path_to_specs)
    # cpi = pd.read_csv(path_to_cpi, index_col=0)

    # # Set specs start year to 2001 and end year to 2023
    # specs["start_year"] = 2001
    # specs["end_year"] = 2019

    # # Load raw data
    # df = pd.read_csv(path_to_raw, index_col=[0, 1])

    # # Pre-filter age and sex
    # start_age_adj = specs["start_age"] - specs["health_smoothing_bandwidth"]
    # df = filter_below_age(df, start_age_adj)
    # df = filter_above_age(df, specs["end_age"] + specs["health_smoothing_bandwidth"])
    # df = recode_sex(df)

    # # Create partner state
    # df = create_partner_state(df, filter_missing=True)

    # # Create parent info
    # df = create_parent_info(df, filter_missing=False)

    # # Create caregiving
    # df = create_caregiving(df, filter_missing=False)

    # # Create dummy variable: > 0 -> 1, == -2 -> 0, else -> NaN
    # df["formal_care_costs_dummy"] = np.select(
    #     [df["hle0016"] > 0, df["hle0016"] == -2],
    #     [1, 0],
    #     default=np.nan,
    # )
    # df["formal_care_costs_raw"] = df["hle0016"].copy()

    # # Create education type
    # df = create_education_type(df)

    # # Create inheritance
    # df = create_inheritance(df, cpi_data=cpi, specs=specs)

    # # Save to CSV
    # df.to_csv(path_to_save)

    specs = read_and_derive_specs(path_to_specs)
    cpi = pd.read_csv(path_to_cpi, index_col=0)
    # wealth = pd.read_csv(path_to_wealth, index_col=[0])

    specs["start_year"] = 2001
    specs["end_year"] = 2019

    # merged_data = pd.read_csv(path_to_raw, index_col=[0, 1])
    df = pd.read_csv(path_to_raw, index_col=[0, 1])

    df = create_partner_state(df, filter_missing=True)
    df = create_kidage_youngest(df)

    df = create_parent_info(df, filter_missing=False)
    df = create_sibling_info(df, filter_missing=False)

    df = create_choice_variable(df)
    df = create_caregiving(df, filter_missing=False)

    # df = add_wealth_data(df, wealth, cpi=cpi, specs=specs, drop_missing=False)

    # filter data. Leave additional years in for lagging and leading.
    df = filter_data(df, specs)

    # Create dummy variable: > 0 -> 1, == -2 -> 0, else -> NaN
    # -2 indicates "no formal care costs" in SOEP data

    df["formal_care_costs_dummy"] = np.select(
        [df["hle0016"] > 0, df["hle0016"] == SOEP_DOES_NOT_APPLY],
        # [df["hle0016"] > 0, df["hle0016"] < 0],
        [1, 0],
        default=np.nan,
    )
    df["formal_care_costs_raw"] = df["hle0016"].copy()
    # df["any_formal_care_costs"] = (df["formal_care_costs_raw"] > 0).astype(int)
    # df = deflate_formal_care_costs(
    #     df, cpi_data=cpi, specs=specs, var_name="formal_care_costs_raw"
    # )

    # df = generate_job_separation_var(df)
    # df = create_lagged_and_lead_variables(
    #     df, specs, lead_job_sep=True, drop_missing_lagged_choice=True
    # )

    df = generate_job_separation_var(df)
    df = create_lagged_and_lead_variables(
        df, specs, lead_job_sep=True, drop_missing_lagged_choice=True
    )
    # Create lagged_any_care variable from any_care
    # pid is in the index (level 0)
    df["lagged_any_care"] = df.groupby(level=0)["any_care"].shift(1)

    # df = create_alreay_retired_variable(df)
    # df = df.reset_index()
    # df = df.sort_values(["pid", "syear"])

    # retired_values = np.asarray(RETIREMENT).ravel().tolist()
    # df["retire_flag"] = (
    #     df["lagged_choice"].isin(retired_values) & df["choice"].isin(retired_values)
    # ).astype(int)
    # df["already_retired"] = df.groupby("pid")["retire_flag"].cummax()

    # df.drop(columns=["retire_flag"], inplace=True)
    # df.set_index(["pid", "syear"], inplace=True)

    df["period"] = df["age"] - specs["start_age"]

    df = create_policy_state(df, specs)

    df = create_experience_variable_with_cap(df, exp_cap=specs["start_age"] - 14)
    # df = create_experience_and_working_years(df, filter_missings=True)

    df = create_education_type(df)
    df = create_inheritance(df, cpi_data=cpi, specs=specs)

    # health variable not yet available for 2023
    # df = create_health_var_good_bad(df, drop_missing=False)
    df = create_health_var_good_bad(df, drop_missing=True)
    df = create_nursing_home(df)

    df = enforce_model_choice_restriction(df, specs)

    # Construct job offer state
    was_fired_last_period = df["job_sep_this_year"] == 1
    df = determine_observed_job_offers(
        df, working_choices=WORK_CHOICES, was_fired_last_period=was_fired_last_period
    )

    # Filter out part-time men
    part_time_values = np.asarray(PART_TIME_CHOICES).ravel().tolist()
    mask = df["sex"] == 0
    df = df.loc[~(mask & df["choice"].isin(part_time_values))]
    df = df.loc[~(mask & df["lagged_choice"].isin(part_time_values))]

    df["has_sister"] = (df["n_sisters"] > 0).astype(int)
    df["mother_age_diff"] = df["mother_age"] - df["age"]
    df["father_age_diff"] = df["father_age"] - df["age"]

    _obs_per_pid = df.groupby("pid").size().rename("n_obs")

    # Keep relevant columns (i.e. state variables) and set their minimal datatype
    type_dict = {
        "pid": "int32",
        "syear": "int16",
        "gebjahr": "int16",
        "age": "int8",
        "period": "int8",
        "choice": "int8",
        "lagged_choice": "float32",  # can be NA
        "policy_state": "int8",
        "policy_state_value": "int8",
        # "already_retired": "int8",
        "partner_state": "int8",
        "job_offer": "int8",
        "experience": "int8",
        # "wealth": "float32",
        # "lagged_wealth": "float32",
        "education": "int8",
        "health": "float16",
        "nursing_home": "float16",
        "sex": "int8",
        "children": "int8",
        "kidage_youngest": "int8",
        # caregiving, contains nans
        "any_care": "float32",
        "lagged_any_care": "float32",
        "light_care": "float32",
        "intensive_care": "float32",
        "has_sister": "float32",
        "n_siblings": "float32",
        "mother_age_diff": "float32",
        "father_age_diff": "float32",
        "mother_alive": "float32",
        "father_alive": "float32",
        "mother_died_this_year": "float32",
        "father_died_this_year": "float32",
        "inheritance_this_year": "float32",
        "year_inheritance": "float32",
        "inheritance_amount": "float32",
        "formal_care_costs_dummy": "float32",
        "formal_care_costs_raw": "float32",
    }
    # df = df.reset_index(level="syear")
    # Reset both levels of the index to make pid and syear regular columns
    df = df.reset_index()
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    # print_data_description(df)

    # Anonymize and save data
    # df.reset_index(drop=True, inplace=True)
    # df.to_csv(path_to_save)

    # Save without index (pid and syear are now columns)
    df.to_csv(path_to_save, index=True)
