"""Merge all SHARE waves and modules."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC

GERMANY = 12
MISSING_VALUE = -9


WAVE_7 = 7
WAVE_8 = 8

ALL_VARIABLES = {
    "cv_r": [
        "int_year",
        "int_month",
        "gender",
        "mobirth",
        "yrbirth",
        "age_int",
        "hhsize",
    ],
    "dn": [
        "dn002_",
        "dn003_",
        "dn019_",  # widowed since when
        "dn010_",
        "dn041_",
        "dn009_",
        "dn014_",
        "dn015_",
        "dn016_",
        "dn026_1",
        "dn026_2",
        "dn033_1",
        "dn033_2",
        "dn027_1",
        "dn027_2",
        "dn028_1",
        "dn028_2",
        "dn030_1",
        "dn030_2",
        "dn034_",  # any siblings
        "dn036_",  # how many brothers alive
        "dn037_",  # how many sisters alive
        "dn127_1",
        "dn127_2",
        "dn032_1",
        "dn032_2",
        "dn012d1",
        "dn012d2",
        "dn012d3",
        "dn012d4",
        "dn012d5",
        "dn012d6",
        "dn012d7",
        "dn012d8",
        "dn012d9",
        "dn012d10",
        "dn012d11",
        "dn012d12",
        "dn012d13",
        "dn012d14",
        "dn012d15",
        "dn012d16",
        "dn012d17",
        "dn012d18",
        "dn012d19",
        "dn012d20",
        # "dn012d95",
        "dn012dno",
        "dn012dot",
    ],
    "ep": [
        "ep002_",
        "ep005_",
        # "ep009_",
        "ep013_",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
        "ep328_",
        "ep329_",
        "ep213_1",
        # "ep213_2",
        # "ep213_3",
        # "ep213_4",
        # "ep213_5",
        # "ep213_6",
        # "ep213_7",
        # "ep213_8",
        # "ep213_9",
        # "ep213_10",
        # "ep213_11",
        # "ep213_12",
        # "ep213_13",
        # "ep213_14",
        # "ep213_15",
        # "ep213_16",
    ],
    "sp": [
        # outside household
        "sp008_",  # given help outside
        "sp009_1",  # to whom given help outside 1
        "sp009_2",  # to whom given help outside 2
        "sp009_3",  # to whom given help outside 3
        "sp010d1_1",  # help given person 1: personal care
        "sp010d1_2",  # help given person 2: personal care
        "sp010d1_3",  # help given person 3: personal care
        # only wave 1 and 2
        # sp012_1, # number of hours practical help
        #
        "sp011_1",  # how often given help to person 1
        "sp011_2",  # how often given help to person 1
        "sp011_3",  # how often given help to person 1
        "sp013_1",  # GiveHelpToOth
        "sp013_2",  # GiveHelpToOth
        # "sp013_3",  # GiveHelpToOth
        # within household
        "sp018_",  # given help within
        "sp019d1",  # provided help with personal care to: spouse/partner
        "sp019d2",  # provided help with personal care to: mother
        "sp019d3",  # provided help with personal care to: father
        "sp019d4",
        "sp019d5",
        "sp019d6",
        "sp019d7",
        "sp019d8",
        "sp019d9",
        "sp019d10",
        "sp019d11",
        # "sp019d12",
        # "sp019d13",
        # "sp019d14",
        # "sp019d15",
        # "sp019d16",
        # "sp019d17",
        # "sp019d18",
        # "sp019d19",
        # "sp019d20",
        # received personal care in household
        "sp020_",  # someone in this household helped you regularly with personal care
        "sp021d1",  # R received help with personal care from: spouse/partner
        "sp021d10",  # child 1
        "sp021d11",  # child 2
        "sp021d12",  # child 3
        "sp021d13",  # child 4
        "sp021d14",  # child 5
        "sp021d15",  # child 6
        "sp021d16",  # child 7
        "sp021d17",  # child 8
        "sp021d18",  # child 9
        "sp021d19",  # child other
        "sp021d20",  # son in law
        "sp021d21",  # daughter in law
    ],
    "gv_isced": ["isced1997_r"],
    # "gv_imputations": [
    #    "hnetw"
    # ],
    #  household net worth =
    # total gross financial assets + total real assets - total libailities
    # children
    "ch": [
        "ch001_",  # number of children
    ],
}

KEYS_TO_REMOVE_WAVE1 = {
    "dn": [
        "dn041_",
        "dn127_1",
        "dn127_2",
        "dn012d15",
        "dn012d16",
        "dn012d17",
        "dn012d18",
        "dn012d19",
        "dn012d20",
    ],
    "ep": [
        "ep013_",
        "ep328_",
        "ep329_",
        "ep213_12",
        "ep213_13",
        "ep213_14",
        "ep213_15",
        "ep213_16",
    ],
    # "hc": [
    #     "hc696_",  # Paid anything yourself stay in nursing home
    #     "hc127d1",  # help with personal care
    #     "hc127d2",  # help with domestic tasks in own home
    #     "hc127d3",  # meals on wheels
    #     "hc127d4",  # helpt with other activities
    #     "hc127dno",  # none of these
    # ],
}

KEYS_TO_REMOVE_WAVE2 = {
    "dn": [
        "dn127_1",
        "dn127_2",
        "dn012d15",
        "dn012d16",
        "dn012d17",
        "dn012d18",
        "dn012d19",
        "dn012d20",
    ],
    # "hc": [
    #     "hc696_",  # Paid anything yourself stay in nursing home
    #     "hc127d1",  # help with personal care
    #     "hc127d2",  # help with domestic tasks in own home
    #     "hc127d3",  # meals on wheels
    #     "hc127d4",  # helpt with other activities
    #     "hc127dno",  # none of these
    # ],
}


KEYS_TO_REMOVE_WAVE4 = {
    "dn": [
        "dn127_1",
        "dn127_2",
        "dn012d14",
        "dn012d15",
        "dn012d16",
        "dn012d17",
        "dn012d18",
        "dn012d19",
        "dn012d20",
    ],
    # type of help not answered, assume help includes personal care
    "sp": [
        "sp010d1_1",  # help given person 1: personal care
        "sp010d1_2",  # help given person 2: personal care
        "sp010d1_3",  # help given person 3: personal care
        # within household personal care to brother etc.
        # "sp019d8",
        # "sp019d9",
        "sp019d10",
        "sp019d11",
        # provided help with personal care to child 3 - 9
        "sp019d12",
        "sp019d13",
        "sp019d14",
        "sp019d15",
        "sp019d16",
        "sp019d17",
        "sp019d18",
        "sp019d19",
        "sp019d20",
        #
        # received help with personal care from child 3 - 9
        "sp021d10",
        "sp021d11",
        "sp021d12",
        "sp021d13",
        "sp021d14",
        "sp021d15",
        "sp021d16",
        "sp021d17",
        "sp021d18",
        "sp021d19",
        "sp021d20",
        "sp021d21",
    ],
    # "hc": [
    #     "hc035_",  # How many weeks did you receive paid help for domestic tasks
    #     "hc036_",  # How many hours per week did you receive such professional help?
    #     "hc037_",  # How many weeks did you receive meals-on-wheel
    #     "hc696_",  # Paid anything yourself stay in nursing home
    #     "hc127d1",  # help with personal care
    #     "hc127d2",  # help with domestic tasks in own home
    #     "hc127d3",  # meals on wheels
    #     "hc127d4",  # helpt with other activities
    #     "hc127dno",  # none of these
    # ],
}


KEYS_TO_REMOVE_WAVE5 = {
    "dn": [
        "dn127_1",  # year of death mother
        "dn127_2",  # year of death father
        "dn012d20",  # further educ category 20
        "dn012dno",  # further educ none
    ],
    # type of help not answered, assume help includes personal care
    "sp": [
        "sp010d1_1",  # help given person 1: personal care
        "sp010d1_2",  # help given person 2: personal care
        "sp010d1_3",  # help given person 3: personal care
    ],
    # "hc": [
    #     "hc035_",  # How many weeks did you receive paid help for domestic tasks
    #     "hc036_",  # How many hours per week did you receive such professional help?
    #     "hc037_",  # How many weeks did you receive meals-on-wheel
    #     "hc696_",  # Paid anything yourself stay in nursing home
    # ],
}

KEYS_TO_REMOVE_WAVE6 = {
    "dn": [
        "dn012dno",
    ],
    "ep": [
        "ep213_14",
        "ep213_15",
        "ep213_16",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
    ],
    # provided help with personal care to child 3 - 9
    # "sp": [
    #    "sp019d12",
    #    "sp019d13",
    #    "sp019d14",
    #    "sp019d15",
    #    "sp019d16",
    #    "sp019d17",
    #    "sp019d18",
    #    "sp019d19",
    # ],
    # received help with personal care from child 3 - 9
    "sp": [
        "sp021d12",
        "sp021d13",
        "sp021d14",
        "sp021d15",
        "sp021d16",
        "sp021d17",
        "sp021d18",
        "sp021d19",
    ],
    # "hc": [
    #     "hc035_",  # How many weeks did you receive paid help for domestic tasks
    #     "hc036_",  # How many hours per week did you receive such professional help?
    #     "hc037_",  # How many weeks did you receive meals-on-wheel
    # ],
}

KEYS_TO_REMOVE_WAVE7 = {
    "dn": [
        "dn012dno",
    ],
    "ep": [
        "ep213_14",
        "ep213_15",
        "ep213_16",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
    ],
    "re": [
        "sl_re018_1",
        "sl_re018_2",
        "sl_re018_3",
        "sl_re018_4",
        "sl_re018_5",
        "sl_re018_6",
        "sl_re018_7",
        "sl_re018_8",
        "sl_re018_9",
        "sl_re018_10",
        "sl_re018_11",
        "sl_re018_12",
        "sl_re018_13",
        "sl_re018_14",
        "sl_re018_15",
        "sl_re018_16",
        "sl_re018_17",
        "sl_re020_1",
        "sl_re020_2",
        "sl_re020_3",
        "sl_re020_4",
        "sl_re020_5",
        "sl_re020_6",
        "sl_re020_7",
        "sl_re020_8",
        "sl_re020_9",
        "sl_re020_10",
        "sl_re020_11",
        "sl_re020_12",
        "sl_re020_13",
        "sl_re020_14",
        "sl_re020_15",
        "sl_re020_16",
        "sl_re020_17",
    ],
    "rp": [
        "sl_rp003_18",
        "sl_rp008_6",
    ],  # year started living with partner, year married
    # provided help with personal care to child 3 - 9
    # "sp": [
    #    "sp019d12",
    #    "sp019d13",
    #    "sp019d14",
    #    "sp019d15",
    #    "sp019d16",
    #    "sp019d17",
    #    "sp019d18",
    #    "sp019d19",
    # ],
    # received help with personal care from child 3 - 9
    "sp": [
        "sp021d12",
        "sp021d13",
        "sp021d14",
        "sp021d15",
        "sp021d16",
        "sp021d17",
        "sp021d18",
        "sp021d19",
    ],
}

KEYS_TO_REMOVE_WAVE8 = {
    "dn": [
        "dn012dno",
    ],
    "ep": [
        "ep213_14",
        "ep213_15",
        "ep213_16",
        "ep071d1",  # income sources, pension payments
        "ep071d2",  # income sources, pension payments
        "ep071d3",  # income sources, pension payments
        "ep071d4",  # income sources, pension payments
        "ep071d5",  # income sources, pension payments
        "ep071d6",  # income sources, pension payments
        "ep071d7",  # income sources, pension payments
        "ep071d8",  # income sources, pension payments
        "ep071d9",  # income sources, pension payments
        "ep071d10",  # income sources, pension payments
    ],
    # provided help with personal care to child 3 - 9
    # "sp": [
    #    "sp019d12",
    #    "sp019d13",
    #    "sp019d14",
    #    "sp019d15",
    #    "sp019d16",
    #    "sp019d17",
    #    "sp019d18",
    #    "sp019d19",
    # ],
    # received help with personal care from child 3 - 9
    "sp": [
        "sp021d12",
        "sp021d13",
        "sp021d14",
        "sp021d15",
        "sp021d16",
        "sp021d17",
        "sp021d18",
        "sp021d19",
    ],
}

GV_VARS = [
    "gender",
    "age",  # Age of respondent (based on interview year)
    "age_p",  # Age of partner (based on interview year)
    "mstat",  # Marital status
    "single",
    "couple",
    "partner",
    "nursinghome",  # Living in nursing home: MN024_
    # "perho", # Percentage of house owned
    "ydip",  # Earnings from employment: EP205
    "yind",  # Earnings from self-employment: EP207
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
    "iadl",  # Limitations with instrumental activities of daily living: PH049_2
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


def task_merge_waves_and_modules(
    path: Annotated[Path, Product] = BLD / "data" / "data_merged.csv",
) -> None:
    # Retrospective waves
    re_vars = (
        [f"sl_re011_{i}" for i in range(1, 21)]
        + [f"sl_re016_{i}" for i in range(1, 21)]
        + [f"sl_re026_{i}" for i in range(1, 21)]
        + [f"sl_re018_{i}" for i in range(1, 17)]
        + [f"sl_re020_{i}" for i in range(1, 17)]
    )
    rp_vars = (
        ["sl_rp002_", "sl_rp002d_", "sl_rp002e_"]
        + [f"sl_rp003_{i}" for i in range(11, 19)]  # year started living with partner
        + [
            f"sl_rp004b_{i}" for i in range(1, 6)
        ]  # year started living with married partner
        # year started living with partner
        + [f"sl_rp004c_{i}" for i in (1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17)]
        + [f"sl_rp008_{i}" for i in range(1, 7)]  # year married
        + [f"sl_rp013_{i}" for i in range(1, 5)]  #  divorced partner
        + [f"sl_rp014_{i}" for i in range(1, 5)]  # year divorced partner
    )

    rp_vars_wave3 = [
        f"sl_rp011_{i}" for i in (1, 2, 3, 11, 12, 13, 14, 15)
    ]  # 16 year death of partner
    rp_vars_wave7 = [f"sl_rp011_{i}" for i in (1, 2, 3, 4, 11, 12, 13, 14, 15)]

    # Data modules for wave 3
    variables_wave3 = {
        "cv_r": [
            "int_year",
            "int_month",
            "gender",
            "mobirth",
            "yrbirth",
            "age_int",
            "hhsize",
        ],
        "re": re_vars,
        "rp": rp_vars + rp_vars_wave3,
        "gv_weights": ["dw_w3", "cchw_w3", "cciw_w3"],
    }

    # Separate modules for partly retrospective wave 7
    _retrospective_wave7 = {"re": re_vars, "rp": rp_vars + rp_vars_wave7}
    _survey_weights_wave7 = {"gv_weights": ["dw_w7", "cchw_w7", "cciw_w7"]}
    variables_wave7 = filter_nested_dict(
        ALL_VARIABLES | _retrospective_wave7 | _survey_weights_wave7,
        KEYS_TO_REMOVE_WAVE7,
    )

    _weights_w1 = {"gv_weights": ["dw_w1", "cchw_w1", "cciw_w1"]}
    _weights_w2 = {"gv_weights": ["dw_w2", "cchw_w2", "cciw_w2"]}
    _weights_w4 = {"gv_weights": ["dw_w4", "cchw_w4", "cciw_w4"]}
    _weights_w5 = {"gv_weights": ["dw_w5", "cchw_w5", "cciw_w5"]}
    _weights_w6 = {"gv_weights": ["dw_w6", "cchw_w6", "cciw_w6"]}
    _weights_w8 = {"gv_weights": ["dw_w8", "cchw_w8_main", "cciw_w8_main"]}

    variables_wave1 = filter_nested_dict(
        ALL_VARIABLES | _weights_w1,
        KEYS_TO_REMOVE_WAVE1,
    )
    variables_wave2 = filter_nested_dict(
        ALL_VARIABLES | _weights_w2,
        KEYS_TO_REMOVE_WAVE2,
    )
    variables_wave4 = filter_nested_dict(
        ALL_VARIABLES | _weights_w4,
        KEYS_TO_REMOVE_WAVE4,
    )
    variables_wave5 = filter_nested_dict(
        ALL_VARIABLES | _weights_w5,
        KEYS_TO_REMOVE_WAVE5,
    )
    variables_wave6 = filter_nested_dict(
        ALL_VARIABLES | _weights_w6,
        KEYS_TO_REMOVE_WAVE6,
    )
    variables_wave8 = filter_nested_dict(
        ALL_VARIABLES | _weights_w8,
        KEYS_TO_REMOVE_WAVE8,
    )

    wave1 = process_wave(wave_number=1, data_modules=variables_wave1)
    wave2 = process_wave(wave_number=2, data_modules=variables_wave2)
    wave3 = process_wave(wave_number=3, data_modules=variables_wave3)
    wave4 = process_wave(wave_number=4, data_modules=variables_wave4)
    wave5 = process_wave(wave_number=5, data_modules=variables_wave5)
    wave6 = process_wave(wave_number=6, data_modules=variables_wave6)
    wave7 = process_wave(wave_number=7, data_modules=variables_wave7)
    wave8 = process_wave(wave_number=8, data_modules=variables_wave8)

    waves_list = [wave1, wave2, wave3, wave4, wave5, wave6, wave7, wave8]

    # Drop all nan rows
    for i, df in enumerate(waves_list):
        waves_list[i] = df.dropna(how="all", axis=0, inplace=False)

    data = merge_wave_datasets(waves_list)

    # GV_IMPUTATIONS
    gv_wave1 = process_gv_imputations(wave=1, args=GV_VARS)
    gv_wave2 = process_gv_imputations(wave=2, args=GV_VARS)
    gv_wave4 = process_gv_imputations(wave=4, args=GV_VARS)
    gv_wave5 = process_gv_imputations(wave=5, args=GV_VARS)
    gv_wave6 = process_gv_imputations(wave=6, args=GV_VARS)
    gv_wave7 = process_gv_imputations(wave=7, args=GV_VARS)
    gv_wave8 = process_gv_imputations(wave=8, args=GV_VARS)

    gv_wave_list = [
        gv_wave1,
        gv_wave2,
        gv_wave4,
        gv_wave5,
        gv_wave6,
        gv_wave7,
        gv_wave8,
    ]

    # Concatenate the DataFrames vertically
    stacked_gv_data = pd.concat(gv_wave_list, axis=0, ignore_index=True)

    # Sort the DataFrame by 'mergeid' and 'wave'
    stacked_gv_data = stacked_gv_data.sort_values(by=["mergeid", "wave"])
    stacked_gv_data = stacked_gv_data.reset_index(drop=True)
    stacked_gv_data = stacked_gv_data.drop("gender", axis=1)

    # Merge 'data' and 'stacked_gv_data' on 'mergeid' and 'wave' with a left join
    data_merged = data.merge(stacked_gv_data, on=["mergeid", "wave"], how="left")

    # Drop per-wave weights
    columns_to_drop = [
        col
        for col in data.columns
        if any(col.startswith(prefix) for prefix in ("dw_w", "cciw_w", "cchw_w"))
    ]
    data_merged = data_merged.drop(columns=columns_to_drop)

    # save data
    data_merged.to_csv(path, index=False)


# =============================================================================


def merge_wave_datasets(wave_datasets):
    """Merge the wave datasets into one data frame."""
    # Combine the data frames in wave_datasets into one data frame
    combined_data = pd.concat(wave_datasets, axis=0, ignore_index=True)

    # Filter out rows where the 'int_year' column is not equal to -9
    combined_data = combined_data[combined_data["int_year"] != MISSING_VALUE]

    # Sort the data frame by 'mergeid' and 'int_year'
    return combined_data.sort_values(by=["mergeid", "int_year"])


def process_wave(wave_number, data_modules):
    """Process a single wave of the SHARE data."""
    wave_data = {}

    for module in data_modules:
        module_file = (
            SRC / f"data/sharew{wave_number}/sharew{wave_number}_rel8-0-0_{module}.dta"
        )

        # Read and filter
        if module in ("re", "rp") and wave_number == WAVE_7:
            _wave_module = pd.read_stata(module_file, convert_categoricals=False)
            _wave_module = _wave_module[_wave_module["country"] == GERMANY]

            lookup = {
                str(var[3:]): str(var)
                for var in data_modules[module]
                if var.startswith("sl")
            }

        else:
            _wave_module = pd.read_stata(module_file, convert_categoricals=False)
            _wave_module = _wave_module[_wave_module["country"] == GERMANY]

            lookup = {
                "sp009_1sp": "sp009_1",
                "sp009_2sp": "sp009_2",
                "sp009_3sp": "sp009_3",
                "sp019d1sp": "sp019d1",
                "sp019d2sp": "sp019d2",
                "sp019d3sp": "sp019d3",
                "sp019d4sp": "sp019d4",
                "sp019d5sp": "sp019d5",
                "sp019d6sp": "sp019d6",
                "sp019d7sp": "sp019d7",
                "sp019d8sp": "sp019d8",
                "sp019d9sp": "sp019d9",
                "sp019d10sp": "sp019d10",
                "sp019d11sp": "sp019d11",
                "sp019d12sp": "sp019d12",
                "sp019d13sp": "sp019d13",
                "sp019d14sp": "sp019d14",
                "sp019d15sp": "sp019d15",
                "sp019d16sp": "sp019d16",
                "sp019d17sp": "sp019d17",
                "sp019d18sp": "sp019d18",
                "sp019d19sp": "sp019d19",
                "sp019d20sp": "sp019d20",
                "sp019d21sp": "sp019d21",
                # received personal care within household
                "sp021d1sp": "sp021d1",
                "sp021d2sp": "sp021d2",
                "sp021d3sp": "sp021d3",
                "sp021d4sp": "sp021d4",
                "sp021d5sp": "sp021d5",
                "sp021d6sp": "sp021d6",
                "sp021d7sp": "sp021d7",
                "sp021d8sp": "sp021d8",
                "sp021d9sp": "sp021d9",
                "sp021d10sp": "sp021d10",
                "sp021d11sp": "sp021d11",
                "sp021d12sp": "sp021d12",
                "sp021d13sp": "sp021d13",
                "sp021d14sp": "sp021d14",
                "sp021d15sp": "sp021d15",
                "sp021d16sp": "sp021d16",
                "sp021d17sp": "sp021d17",
                "sp021d18sp": "sp021d18",
                "sp021d19sp": "sp021d19",
                "sp021d20sp": "sp021d20",
                "sp021d21sp": "sp021d21",
            }

        # Rename columns using the dictionary
        _wave_module = _wave_module.rename(columns=lookup)

        module_vars = ["mergeid"] + data_modules[module]

        # Select columns
        _wave_module = _wave_module[module_vars]

        wave_data[module] = _wave_module

    add_wealth_data = "gv_imputations" in data_modules
    merged_data = wave_data["cv_r"]

    data_modules.pop("cv_r")
    data_modules.pop("gv_imputations", None)

    for module_key in data_modules:
        merged_data = merged_data.merge(
            wave_data[module_key],
            on="mergeid",
            how="outer",
        )

    if add_wealth_data:
        merged_data = merged_data.merge(
            wave_data["gv_imputations"],
            on="mergeid",
            how="left",
        )

    merged_data["wave"] = wave_number

    # Add survey weights
    if wave_number == WAVE_8:
        merged_data["design_weight"] = merged_data["dw_w8"]
        merged_data["hh_weight"] = merged_data["cchw_w8_main"]
        merged_data["ind_weight"] = merged_data["cciw_w8_main"]
    else:
        merged_data["design_weight"] = merged_data[f"dw_w{wave_number}"]
        merged_data["hh_weight"] = merged_data[f"cchw_w{wave_number}"]
        merged_data["ind_weight"] = merged_data[f"cciw_w{wave_number}"]

    return merged_data


def process_gv_imputations(wave, args):
    module = "gv_imputations"
    module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel8-0-0_{module}.dta"
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


def filter_nested_dict(original_dict, keys_to_remove):
    return {
        key: (
            [value for value in values if value not in keys_to_remove.get(key, [])]
            if key in keys_to_remove
            else values
        )
        for key, values in original_dict.items()
    }


def load_and_rename_wave_data(wave):
    module = "sp"
    module_file = SRC / f"data/sharew{wave}/sharew{wave}_rel8-0-0_{module}.dta"

    data = pd.read_stata(module_file, convert_categoricals=False)
    data.columns = [col.removesuffix("sp") for col in data.columns]

    return data
