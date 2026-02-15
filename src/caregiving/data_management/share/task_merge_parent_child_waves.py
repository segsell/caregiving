"""Merge parent information."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC

GERMANY = 12
MISSING_VALUE = -9
WAVE_1 = 1
WAVE_2 = 2
WAVE_3 = 3
WAVE_4 = 4
WAVE_5 = 5
WAVE_6 = 6
WAVE_7 = 7
WAVE_8 = 8

CV_R = [
    "int_year",
    "int_month",
    "gender",
    "mobirth",
    "yrbirth",
    "age_int",
    "hhsize",
]

CHILDREN = [
    "ch005_",  # gender
    "ch006_",  # year of birth
    "ch007_",  # where does child live
    "ch012_",  # marital status
    "ch014_",  # frequency contact with child
    "ch016_",  # employment status
    "ch017_",  # highest education
]

PHYSICAL_HEALTH = ["ph003_"]

HEALTH_CARE = [
    "hc029_",  # in nursing home during last 12 months
    # 1 = yes, temporarily
    # 3 = yes, permanently
    # 5 = no
    "hc031_",  # Weeks stayed in a nursing home or residential care facility
    "hc032d1",  # nursing or personal care
    "hc032d2",  # domestic tasks
    "hc032d3",  # meals on wheels
    "hc032dno",  # none of these
    "hc033_",  # How many weeks did you receive paid help for personal care
    "hc034_",  # How many hours per week did you receive such professional help?
    "hc035_",  # How many weeks did you receive professional help for domestic tasks
    "hc036_",  # How many hours per week did you receive such professional help?
    "hc037_",  # How many weeks did you receive meals-on-wheel
    # since wave 5
    "hc696_",  # Paid anything yourself stay in nursing home
    "hc127d1",  # help with personal care
    "hc127d2",  # help with domestic tasks in own home
    "hc127d3",  # meals on wheels
    "hc127d4",  # helpt with other activities
    "hc127dno",  # none of these
]


SOCIAL_SUPPORT = [
    "sp020_",  # someone in this household helped you regularly with personal care
    "sp021d1",  # R received help with personal care from: spouse/partner (inside HH)
    "sp021d10",  # from own child
    "sp021d11",  # from stepchild
    "sp021d20",  # from son-in-law
    "sp021d21",  # from daughter-in-law
    # only from wave 7 on :9
    # "sp033_1",  # from which child in household
    # "sp033_2",  # from which child in household
    # "sp033_3",  # from which child in household
    # "sp033_4",  # from which child in household
    # "sp033_5",  # from which child in household
    # "sp033_6",  # from which child in household
    # "sp033_7",  # from which child in household
    "sp002_",  # help from outsided the household
    "sp003_1",  # from whom, person 1: 10, 11, 20 ,21
    "sp003_2",  # from whom, person 2: 10, 11, 20 ,21
    "sp003_3",  # from whom, person 3: 10, 11, 20 ,21
    # only from wave 7 on :9
    # "sp027_1",  # from which child
    # "sp027_2",  # from which child
    # "sp027_3",  # from which child
    "sp004d1_1",  # help with personal care
    "sp004d1_2",  # help with personal care
    "sp004d1_3",  # help with personal care
    "sp004d2_1",  # practical household help
    "sp004d2_2",  # practical household help
    "sp004d2_3",  # practical household help
    "sp005_1",  # how often per week from this person
    "sp005_2",  # how often per week from this person
    "sp005_3",  # how often per week from this person
]


WAVE_4_SN_PERSONS_CARE = {
    # Social network person: relationship (only waves 4, 6, 8, 9)
    "sn": [
        "sn005_1",
        "sn005_2",
        "sn005_3",
        "sn005_4",
        "sn005_5",
        "sn005_6",
        "sn005_7",
    ],
    "sp": [
        "sp021d1sn",  # received help with personal care from social network person 1
        "sp021d2sn",  # received help with personal care from social network person 2
        "sp021d3sn",  # received help with personal care from social network person 3
        "sp021d4sn",  # received help with personal care from social network person 4
        "sp021d5sn",  # received help with personal care from social network person 5
        "sp021d6sn",  # received help with personal care from social network person 6
        "sp021d7sn",  # received help with personal care from social network person 7
    ],
}

GV_CHILDREN = [
    "ch_gender_",
    "ch_yrbirth_",
    "ch_yrbirth_youngest_child_",
    "ch_occupation_",
    "ch_proximity_",
    "ch_marital_status_",
    "ch_school_education_",
    "ch_outhh_receive_care_",
    "ch_hh_receive_care_",
]

GV_VARS = [
    "gender",
    "age",  # Age of respondent (based on interview year)
    "age_p",  # Age of partner (based on interview year)
    "mstat",  # Marital status
    "single",
    "couple",
    "partner",
    "nursinghome",
    # "perho",
    "ydip",
    "yind",
    "ypen1",  # Annual old age, early retirement pensions, survivor and war pension
    "ypen2",  # Annual private occupational pensions
    "ypen5",  # Annual payment from social assistance
    "yreg1",  # Other regular payments from private pensions
    "yreg2",  # Other regular payment from private transfer
    "thinc",  # Total household net income - version A
    "thinc2",  # Total household net income - version B
    "hnetw",  # Household net worth (hnfass + hrass)
    "yedu",
    "yedu_p",
    "isced",
    "sphus",  # Self-perceived health - US scale
    "nchild",  # Number of children
    "gali",  # Limitation with activities: PH005
    "chronic",  # Number of chronic diseases: PH006
    "adl",  # Limitations with activities of daily living: PH049_1
    "adl2",
    "iadl",  # Limitations with instrumental activities of daily living: PH049_2
    "iadl2",
    "eurod",  # EURO depression scale: MH002-MH017 (MH031)
    "cjs",  # Current job situation: EP005
    "pwork",  # Did any paid work: EP002
    "empstat",  # Employee or self-employed: EP009; 2- 8
    "rhfo",  # Received help from others (how many): SP002, SP005, SP007
    "ghto",  # Given help to others (how many): SP008, SP011, SP013
    "ghih",  # Given help in the household (how many): SP0181 2 4 5 6 7 (R) 8
    "rhih",  # Received help in the household (how many): SP0201 2 4 5 6 7 (R) 8
    "otrf",  # Owner, tenant or rent free: HO0021 2 4 5 6 7 (R) 8
]


# =============================================================================


def task_merge_parent_child_waves_and_modules(
    path_parent_child: Annotated[Path, Product] = BLD
    / "data"
    / "share_data_parent_child_merged.csv",
) -> None:
    """Merge raw parent information."""
    child_suffixes = [str(i) for i in range(1, 21)]
    vars_children = [var + child for var in CHILDREN for child in child_suffixes]

    all_variables = {
        "cv_r": CV_R,
        "ph": PHYSICAL_HEALTH,
        "ch": vars_children,
        "sp": SOCIAL_SUPPORT,
        "hc": HEALTH_CARE,
    }

    variables_wave1 = all_variables | {"gv_weights": ["dw_w1", "cchw_w1", "cciw_w1"]}
    variables_wave2 = all_variables | {"gv_weights": ["dw_w2", "cchw_w2", "cciw_w2"]}
    variables_wave4 = (
        all_variables
        | WAVE_4_SN_PERSONS_CARE
        | {"gv_weights": ["dw_w4", "cchw_w4", "cciw_w4"]}
    )
    variables_wave5 = all_variables | {"gv_weights": ["dw_w5", "cchw_w5", "cciw_w5"]}
    variables_wave6 = all_variables | {"gv_weights": ["dw_w6", "cchw_w6", "cciw_w6"]}
    variables_wave7 = all_variables | {"gv_weights": ["dw_w7", "cchw_w7", "cciw_w7"]}
    variables_wave8 = all_variables | {
        "gv_weights": ["dw_w8", "cchw_w8_main", "cciw_w8_main"],
    }
    variables_wave7 = all_variables | {"gv_weights": ["dw_w7", "cchw_w7", "cciw_w7"]}
    variables_wave9 = all_variables | {"gv_weights": ["dw_w9", "cchw_w9", "cciw_w9"]}

    wave1 = process_wave(wave=1, data_modules=variables_wave1)
    wave2 = process_wave(wave=2, data_modules=variables_wave2)
    wave4 = process_wave(wave=4, data_modules=variables_wave4)
    wave5 = process_wave(wave=5, data_modules=variables_wave5)
    wave6 = process_wave(wave=6, data_modules=variables_wave6)
    wave7 = process_wave(wave=7, data_modules=variables_wave7)
    wave8 = process_wave(wave=8, data_modules=variables_wave8)
    wave9 = process_wave(wave=9, data_modules=variables_wave9)

    waves_list = [wave1, wave2, wave4, wave5, wave6, wave7, wave8, wave9]
    data = merge_wave_datasets(waves_list)

    vars_gv_children = [var + child for var in GV_CHILDREN for child in child_suffixes]
    gv_6 = process_gv_children(wave=6, args=vars_gv_children)
    gv_7 = process_gv_children(wave=7, args=vars_gv_children)
    gv_8 = process_gv_children(wave=8, args=vars_gv_children)
    gv_9 = process_gv_children(wave=9, args=vars_gv_children)

    gv_children_datasets = [gv_6, gv_7, gv_8, gv_9]
    gv_data = pd.concat(gv_children_datasets, axis=0, ignore_index=True)
    gv_data = gv_data.sort_values(by=["mergeid", "wave"])

    wave_data = data.merge(gv_data, on=["mergeid", "wave"], how="left")

    gv_wave1 = process_gv_imputations(wave=1, args=GV_VARS)
    gv_wave2 = process_gv_imputations(wave=2, args=GV_VARS)
    gv_wave4 = process_gv_imputations(wave=4, args=GV_VARS)
    gv_wave5 = process_gv_imputations(wave=5, args=GV_VARS)
    gv_wave6 = process_gv_imputations(wave=6, args=GV_VARS)
    gv_wave7 = process_gv_imputations(wave=7, args=GV_VARS)
    gv_wave8 = process_gv_imputations(wave=8, args=GV_VARS)
    gv_wave9 = process_gv_imputations(wave=9, args=GV_VARS)

    gv_list = [
        gv_wave1,
        gv_wave2,
        gv_wave4,
        gv_wave5,
        gv_wave6,
        gv_wave7,
        gv_wave8,
        gv_wave9,
    ]

    stacked_gv_data = pd.concat(gv_list, axis=0, ignore_index=True)
    stacked_gv_data = stacked_gv_data.sort_values(by=["mergeid", "wave"])
    stacked_gv_data = stacked_gv_data.reset_index(drop=True)
    stacked_gv_data = stacked_gv_data.drop("gender", axis=1)

    data_merged_gv = wave_data.merge(
        stacked_gv_data,
        on=["mergeid", "wave"],
        how="left",
    )

    columns_to_drop = [
        col
        for col in data.columns
        if any(col.startswith(prefix) for prefix in ("dw_w", "cciw_w", "cchw_w"))
    ]
    data_merged_gv = data_merged_gv.drop(columns=columns_to_drop)

    data_merged_gv.to_csv(path_parent_child, index=False)


def process_wave(wave, data_modules):
    """Process wave of the standard SHARE modules."""
    wave_data = {}

    for module in data_modules:
        _wave_module = process_module(module, wave, data_modules[module])
        wave_data[module] = _wave_module

    merged_data = wave_data["cv_r"]
    merged_data = merged_data.rename(
        columns={f"mergeidp{wave}": "mergeidp", f"coupleid{wave}": "coupleid"},
    )

    for module_key in ("ph", "ch", "sp", "hc", "gv_weights"):
        merged_data = merged_data.merge(
            wave_data[module_key],
            on="mergeid",
            how="outer",
        )

    merged_data = merged_data.copy()
    merged_data["wave"] = wave

    # Add survey weights
    if wave == WAVE_8:
        merged_data["design_weight"] = merged_data["dw_w8"]
        merged_data["hh_weight"] = merged_data["cchw_w8_main"]
        merged_data["ind_weight"] = merged_data["cciw_w8_main"]
    else:
        merged_data["design_weight"] = merged_data[f"dw_w{wave}"]
        merged_data["hh_weight"] = merged_data[f"cchw_w{wave}"]
        merged_data["ind_weight"] = merged_data[f"cciw_w{wave}"]

    return merged_data


def process_module(module, wave, args):
    """Process a single wave module."""
    # module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel9-0-0_{module}.dta"
    module_file = SRC / (
        f"data/sharew{wave}_rel9-0-0_ALL_datasets_stata/"
        f"sharew{wave}_rel9-0-0_{module}.dta"
    )

    data = pd.read_stata(module_file, convert_categoricals=False)

    data.columns = [col.removesuffix("sp") for col in data.columns]

    # Filter the data based on the "country" column
    data = data[data["country"] == GERMANY]

    selected_columns = ["mergeid"] + [col for col in args if col in data.columns]

    if module == "cv_r":
        selected_columns += [f"mergeidp{wave}", f"coupleid{wave}"]

    columns = ["mergeid", *args]

    if wave == WAVE_7:
        not_included = [
            "ch007_1",
            "ch007_2",
            "ch007_3",
            "ch007_4",
            "ch014_1",
            "ch014_2",
            "ch014_3",
            "ch014_4",
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave in (WAVE_1, WAVE_2):
        not_included = [
            "hc696_",  # Paid anything yourself stay in nursing home
            "hc127d1",  # help with personal care
            "hc127d2",  # help with domestic tasks in own home
            "hc127d3",  # meals on wheels
            "hc127d4",  # helpt with other activities
            "hc127dno",
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave == WAVE_4:
        not_included = [
            "sp021d10",
            "sp021d11",
            "sp004d1_1",
            "sp004d1_2",
            "sp004d1_3",
            "sp004d2_1",
            "sp004d2_2",
            "sp004d2_3",
            # "hc032_",
            # "hc033_",
            # "hc034_",
            "hc035_",  # How many weeks did you receive paid help for domestic tasks
            "hc036_",  # How many hours per week did you receive such professional help?
            "hc037_",  # How many weeks did you receive meals-on-wheel
            "hc696_",  # Paid anything yourself stay in nursing home
            "hc696_",  # Paid anything yourself stay in nursing home
            "hc127d1",  # help with personal care
            "hc127d2",  # help with domestic tasks in own home
            "hc127d3",  # meals on wheels
            "hc127d4",  # helpt with other activities
            "hc127dno",
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave == WAVE_5:
        not_included = [
            "sp004d1_1",
            "sp004d1_2",
            "sp004d1_3",
            "sp004d2_1",
            "sp004d2_2",
            "sp004d2_3",
            "hc035_",  # How many weeks did you receive paid help for domestic tasks
            "hc036_",  # How many hours per week did you receive such professional help?
            "hc037_",  # How many weeks did you receive meals-on-wheel
            "hc696_",  # Paid anything yourself stay in nursing home
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    elif wave == WAVE_6:
        not_included = [
            "hc035_",  # How many weeks did you receive paid help for domestic tasks
            "hc036_",  # How many hours per week did you receive such professional help?
            "hc037_",  # How many weeks did you receive meals-on-wheel
        ]

        filtered_args = [arg for arg in args if arg not in not_included]
        columns = ["mergeid", *filtered_args]
    else:
        columns = ["mergeid", *args]

    if module == "cv_r":
        columns += [f"mergeidp{wave}", f"coupleid{wave}"]

    # Replace negative values with NaN using NumPy
    # Create missing columns and fill with NaN
    data = data.copy()
    for col in args:
        if col not in selected_columns:
            data[col] = np.nan

    return data[columns]


def process_gv_children(wave, args):
    """Process single wave of the gv_children module."""
    module = "gv_children"

    # module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel9-0-0_{module}.dta"
    module_file = SRC / (
        f"data/sharew{wave}_rel9-0-0_ALL_datasets_stata/"
        f"sharew{wave}_rel9-0-0_{module}.dta"
    )

    data = pd.read_stata(module_file, convert_categoricals=False)

    # Filter the data based on the "country" column
    data = data[data["country"] == GERMANY]

    # Select columns 'mergeid' and the specified args (create missing columns with NaN)
    selected_columns = ["mergeid"] + [col for col in args if col in data.columns]
    columns = ["mergeid", *args]

    # Replace negative values with NaN using NumPy
    # Create missing columns and fill with NaN
    data = data.copy()
    for col in args:
        if col not in selected_columns:
            data[col] = np.nan

    data = data[columns]
    data["wave"] = wave

    return data


def merge_wave_datasets(wave_datasets):
    """Combine data frames in the wave_datasets into one data frame."""
    combined_data = pd.concat(wave_datasets, axis=0, ignore_index=True)

    # Filter out rows where the 'int_year' column is not equal to -9
    combined_data = combined_data[combined_data["int_year"] != MISSING_VALUE]

    return combined_data.sort_values(by=["mergeid", "int_year"])


def process_gv_imputations(wave, args):
    module = "gv_imputations"

    # module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel9-0-0_{module}.dta"
    module_file = SRC / (
        f"data/sharew{wave}_rel9-0-0_ALL_datasets_stata/"
        f"sharew{wave}_rel9-0-0_{module}.dta"
    )

    data = pd.read_stata(module_file, convert_categoricals=False)

    # Filter the data based on the "country" column
    data = data[data["country"] == GERMANY]

    # Select columns 'mergeid' and the specified args (create missing columns with NaN)
    selected_columns = ["mergeid"] + [col for col in args if col in data.columns]
    columns = ["mergeid", *args]

    # Create missing columns and fill with NaN
    for col in args:
        if col not in selected_columns:
            data[col] = np.nan

    data = data[columns]

    # Replace negative values with NaN using NumPy

    # Group the data by 'mergeid'
    grouped_data = data.groupby("mergeid")

    # Create a dictionary to store the aggregation method for each column
    aggregation_methods = {}
    for column in args:
        dtype = data[column].dtype
        if pd.api.types.is_integer_dtype(dtype):
            aggregation_methods[column] = "median"
        elif pd.api.types.is_float_dtype(dtype):
            aggregation_methods[column] = "mean"

    # Replace negative values with NaN using NumPy
    # this should not change the meaning except for cases where
    # all 5 entries are missing
    # check ?!
    data[args] = np.where(data[args] >= 0, data[args], np.nan)

    # Apply aggregation methods and store the results in a new DataFrame
    aggregated_data = grouped_data.agg(aggregation_methods).reset_index()

    # if "age_p" in args:
    #    # note that single people also have partner_alive = 0

    aggregated_data["wave"] = wave

    return aggregated_data
