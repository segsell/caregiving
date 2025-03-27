"""Survival transition sample."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC
from caregiving.data_management.soep.auxiliary import (
    filter_above_age,
    filter_below_age,
    filter_years,
    recode_sex,
    span_dataframe,
)
from caregiving.data_management.soep.variables import (
    DECEASED,
    clean_health_create_states,
    create_education_type,
    create_health_var_good_bad,
    create_health_var_good_medium_bad,
)
from caregiving.specs.task_write_specs import read_and_derive_specs

# =====================================================================================
# Create mortality estimation sample
# =====================================================================================


def task_create_survival_sample(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw_data: Path = BLD / "data" / "soep_survival_sample_raw.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "data" / "mortality_transition_estimation_sample.pkl"
    ),
    path_to_save_duplicate: Annotated[Path, Product] = (
        BLD / "data" / "mortality_transition_estimation_sample_duplicated.pkl"
    ),
):
    """Create the survival transition sample for the mortality estimation."""

    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_raw_data, index_col=["pid", "syear"])

    # Filter age and estimation years
    df = filter_below_age(df, specs["start_age_mortality"])
    df = filter_above_age(df, specs["end_age_mortality"])
    df = filter_years(df, specs["start_year_mortality"], specs["end_year_mortality"])

    # Create columns for the start age and health state
    df = create_start_age_and_health(df)

    df = df[
        [
            "age",
            "start age",
            "death event",
            "education",
            "sex",
            "health",
            "start health",
        ]
    ]

    # Set the dtype of the columns to float
    df = df.astype(float)

    # Sum the death events for the entire sample
    print(
        "Death events in the sample: ",
        f"{len(df[df['death event'] == 1])} (total) = "
        f"{len(df[(df['death event'] == 1) & (df['health'] == 1)])} (health 1) + "
        f"{len(df[(df['death event'] == 1) & (df['health'] == 0)])} (health 0)",
    )

    print(
        f"Years: {df.index.get_level_values('syear').min()}-"
        f"{df.index.get_level_values('syear').max()}, "
        f"Min age: {df['age'].min()}, Max age: {df['age'].max()}, "
        f"Avg age: {round(df['age'].mean(), 2)}, "
        f"Unique individuals: {df.index.get_level_values('pid').nunique()}, "
        f"Avg time in sample: {round(df.groupby('pid').size().mean(), 2)}"
    )

    print(
        str(len(df))
        + " observations in the final survival transition sample.\n"
        + " ----------------"
    )

    df.to_pickle(path_to_save)

    # Create original and duplicated samples
    df1 = df.copy().reset_index()
    df2 = df.copy().reset_index()

    # Modify df2 with unknown values
    df2["education"] = np.nan
    df2["health"] = np.nan
    df2["start_health"] = np.nan

    # Add true_sample indicators
    df1["true_sample"] = 1
    df2["true_sample"] = 0

    # Duplicate the sample - Kroll and Lampert (2009)
    df_dup = pd.concat([df1, df2])
    df_dup.set_index(["pid", "syear", "true_sample"], inplace=True)
    df_dup.sort_index(inplace=True)

    # # Create interaction indicators for health and education
    # df_dup = create_interaction_columns(df_dup, ("health", "education"), specs)

    # # Create interaction indicators for start health and education
    # df_dup = create_interaction_columns(df_dup, ("start health", "education"), specs)

    df_dup = create_health_columns(df_dup, "health", specs)
    df_dup = create_health_columns(df_dup, "start health", specs)

    # Convert DataFrame to floats for computation
    df_dup = df_dup.astype(float)

    # Save the duplicated sample
    df_dup.to_pickle(path_to_save_duplicate)


def task_create_survival_sample_good_medium_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    path_to_raw_data: Path = BLD
    / "data"
    / "soep_survival_sample_good_medium_bad_raw.csv",
    path_to_save: Annotated[Path, Product] = (
        BLD / "data" / "mortality_transition_estimation_sample_three_states.pkl"
    ),
    path_to_save_duplicate: Annotated[Path, Product] = (
        BLD
        / "data"
        / "mortality_transition_estimation_sample_three_states_duplicated.pkl"
    ),
):
    """Create the survival transition sample for the mortality estimation."""

    specs = read_and_derive_specs(path_to_specs)

    df = pd.read_csv(path_to_raw_data, index_col=["pid", "syear"])

    # Filter age and estimation years
    df = filter_below_age(df, specs["start_age_mortality"])
    df = filter_above_age(df, specs["end_age_mortality"])
    df = filter_years(df, specs["start_year_mortality"], specs["end_year_mortality"])

    # Create columns for the start age and health state
    df = create_start_age_and_health(df)

    df = df[
        [
            "age",
            "start age",
            "death event",
            "education",
            "sex",
            "health",
            "start health",
        ]
    ]

    # Set the dtype of the columns to float
    df = df.astype(float)

    # Sum the death events for the entire sample
    print(
        "Death events in the sample: ",
        f"{len(df[df['death event'] == 1])} (total) = "
        f"{len(df[(df['death event'] == 1) & (df['health'] == 0)])} (health=0) + "
        f"{len(df[(df['death event'] == 1) & (df['health'] == 1)])} (health=1) + "
        f"{len(df[(df['death event'] == 1) & (df['health'] == 2)])} (health=2)",
    )

    print(
        f"Years: {df.index.get_level_values('syear').min()}-"
        f"{df.index.get_level_values('syear').max()}, "
        f"Min age: {df['age'].min()}, Max age: {df['age'].max()}, "
        f"Avg age: {round(df['age'].mean(), 2)}, "
        f"Unique individuals: {df.index.get_level_values('pid').nunique()}, "
        f"Avg time in sample: {round(df.groupby('pid').size().mean(), 2)}"
    )

    print(
        str(len(df))
        + " observations in the final survival transition sample.\n"
        + " ----------------"
    )

    df.to_pickle(path_to_save)

    # Create original and duplicated samples
    df1 = df.copy().reset_index()
    df2 = df.copy().reset_index()

    # Modify df2 with unknown values
    df2["education"] = np.nan
    df2["health"] = np.nan
    df2["start_health"] = np.nan

    # Add true_sample indicators
    df1["true_sample"] = 1
    df2["true_sample"] = 0

    # Duplicate the sample - Kroll and Lampert (2009)
    df_dup = pd.concat([df1, df2])
    df_dup.set_index(["pid", "syear", "true_sample"], inplace=True)
    df_dup.sort_index(inplace=True)

    # Create interaction indicators for health and education
    df_dup = create_health_columns_three_health_states(df_dup, "health", specs)

    # Create interaction indicators for start health and education
    df_dup = create_health_columns_three_health_states(df_dup, "start health", specs)

    # Convert DataFrame to floats for computation
    df_dup = df_dup.astype(float)

    # Save the duplicated sample
    df_dup.to_pickle(path_to_save_duplicate)


# =====================================================================================
# Merge and process raw data
# =====================================================================================


def task_merge_survival_sample_good_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    soep_c38_lifespell: Path = SRC / "data" / "soep" / "lifespell.dta",
    soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_survival_sample_raw.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)

    # Time-invariant data
    pgen_data = pd.read_stata(
        soep_c38_pgen,
        columns=[
            "syear",
            "pid",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        soep_c38_ppathl,
        columns=["pid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )
    time_invariant_data = load_and_process_soep_yearly_survey_data(
        pgen_data, ppathl_data
    )

    # Life spell data
    lifespell_data = pd.read_stata(
        soep_c38_lifespell,
        convert_categoricals=False,
    )
    life_spell_data = load_and_process_life_spell_data(lifespell_data)

    # health data
    pequiv_data = pd.read_stata(
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        soep_c38_pequiv,
        columns=["pid", "syear", "m11126", "m11124"],
        convert_categoricals=False,
    )
    pequiv_data.set_index(["pid", "syear"], inplace=True)
    pequiv_data.sort_index(inplace=True)

    health_data = load_and_process_soep_health(pequiv_data, specs)

    # Merge time invariant data onto the life spell data
    df = pd.merge(life_spell_data, time_invariant_data, on="pid", how="left")
    # Combine life spell data with health data (keep intersection)
    df = pd.merge(df, health_data, on=["pid", "syear"], how="inner")

    df["age"] = df["syear"] - df["gebjahr"]
    df = df.set_index(["pid", "syear"])

    # Save the resulting DataFrame to CSV
    df.to_csv(path_to_save)


def task_merge_survival_sample_good_medium_bad(
    path_to_specs: Path = SRC / "specs.yaml",
    soep_c38_pgen: Path = SRC / "data" / "soep" / "pgen.dta",
    soep_c38_ppathl: Path = SRC / "data" / "soep" / "ppathl.dta",
    soep_c38_lifespell: Path = SRC / "data" / "soep" / "lifespell.dta",
    soep_c38_pequiv: Path = SRC / "data" / "soep" / "pequiv.dta",
    path_to_save: Annotated[Path, Product] = BLD
    / "data"
    / "soep_survival_sample_good_medium_bad_raw.csv",
) -> None:

    specs = read_and_derive_specs(path_to_specs)

    # Time-invariant data
    pgen_data = pd.read_stata(
        soep_c38_pgen,
        columns=[
            "syear",
            "pid",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        soep_c38_ppathl,
        columns=["pid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )
    time_invariant_data = load_and_process_soep_yearly_survey_data(
        pgen_data, ppathl_data
    )

    # Life spell data
    lifespell_data = pd.read_stata(
        soep_c38_lifespell,
        convert_categoricals=False,
    )
    life_spell_data = load_and_process_life_spell_data(lifespell_data)

    # health data
    pequiv_data = pd.read_stata(
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        soep_c38_pequiv,
        columns=["pid", "syear", "m11126", "m11124"],
        convert_categoricals=False,
    )
    pequiv_data.set_index(["pid", "syear"], inplace=True)
    pequiv_data.sort_index(inplace=True)

    health_data = load_and_process_soep_health_good_medium_bad(pequiv_data, specs)

    # Merge time invariant data onto the life spell data
    df = pd.merge(life_spell_data, time_invariant_data, on="pid", how="left")
    # Combine life spell data with health data (keep intersection)
    df = pd.merge(df, health_data, on=["pid", "syear"], how="inner")

    df["age"] = df["syear"] - df["gebjahr"]
    df = df.set_index(["pid", "syear"])

    # Save the resulting DataFrame to CSV
    df.to_csv(path_to_save)


# =====================================================================================
# Load and process raw data
# =====================================================================================


def load_and_process_soep_yearly_survey_data(pgen_data, ppathl_data):
    """Process annual data from SOEP C38"""
    merged_data = pd.merge(pgen_data, ppathl_data, on=["pid", "syear"], how="inner")
    merged_data.set_index(["pid", "syear"], inplace=True)

    # Keep male and female obs. and transform to 0/1 = male/female
    df = recode_sex(merged_data)
    # Create education type variable
    df = create_education_type(df)

    # Keep only the one observation for each individual
    # with the highest education level + invariant variables
    df = df.groupby("pid")[["sex", "education", "gebjahr"]].max()

    return df


def load_and_process_life_spell_data(lifespell_data):
    """Load and process life spell data."""

    # --- Generate spell duration and expand dataset --- lifespell data
    lifespell_data["spellduration"] = (
        lifespell_data["end"] - lifespell_data["begin"]
    ) + 1
    lifespell_data_long = lifespell_data.loc[
        lifespell_data.index.repeat(lifespell_data["spellduration"])
    ].reset_index(drop=True)

    # --- Generate syear --- lifespell data
    lifespell_data_long["n"] = (
        lifespell_data_long.groupby(["pid", "spellnr"]).cumcount() + 1
    )  # +1 since cumcount starts at 0
    lifespell_data_long["syear"] = (
        lifespell_data_long["begin"] + lifespell_data_long["n"] - 1
    )

    # --- Keep only relevant columns --- lifespell data
    columns_to_keep = ["pid", "syear", "spelltyp", "spellnr"]
    lifespell_data_long = lifespell_data_long[columns_to_keep]

    # --- Generate death event variable --- lifespell data
    # https://paneldata.org/soep-core/datasets/lifespell/spelltyp
    lifespell_data_long["death event"] = (
        lifespell_data_long["spelltyp"] == DECEASED
    ).astype("int")

    # Split into dataframes of death and not death
    not_death_idx = lifespell_data_long[lifespell_data_long["death event"] == 0].index
    first_death_idx = (
        lifespell_data_long[lifespell_data_long["death event"] == 1]
        .groupby("pid")["syear"]
        .idxmin()
    )

    # Final index
    final_index = not_death_idx.union(first_death_idx)
    lifespell_data_long = lifespell_data_long.loc[final_index]

    return lifespell_data_long


def load_and_process_soep_health(pequiv_data, specs):
    """Load and process health data."""

    # Create health state variable and span the dataframe
    pequiv_data = create_health_var_good_bad(pequiv_data)
    pequiv_data = span_dataframe(
        pequiv_data, specs["start_year_mortality"], specs["end_year_mortality"]
    )
    pequiv_data = clean_health_create_states(pequiv_data)

    # Fill health gaps
    pequiv_data = fill_health_gaps_vectorized(pequiv_data)

    # Forward fill health state for every individual
    pequiv_data["health"] = pequiv_data.groupby("pid")["health"].ffill()
    # TO DO: this makes the fill gaps function obsolete and is a
    # very strong assumption

    # # for deaths, set the health state to the last known health state
    # pequiv_data["last_known_health"] = pequiv_data.groupby("pid")["health"].transform(
    #     "last"
    # )
    # pequiv_data.loc[
    #     (pequiv_data["death event"] == 1) & (pequiv_data["health"].isna()), "health"
    # ] = pequiv_data.loc[
    #     (pequiv_data["death event"] == 1) & (pequiv_data["health"].isna()),
    #     "last_known_health",
    # ]

    # Drop individuals without any health state information
    pequiv_data = pequiv_data[(pequiv_data["health"].notna())]

    return pequiv_data


def load_and_process_soep_health_good_medium_bad(pequiv_data, specs):
    """Load and process health data."""

    # Create health state variable and span the dataframe
    pequiv_data = create_health_var_good_medium_bad(pequiv_data)
    pequiv_data = span_dataframe(
        pequiv_data, specs["start_year_mortality"], specs["end_year_mortality"]
    )
    pequiv_data = clean_health_create_states(pequiv_data)

    # Fill health gaps
    pequiv_data = fill_health_gaps_vectorized(pequiv_data)

    # Forward fill health state for every individual
    pequiv_data["health"] = pequiv_data.groupby("pid")["health"].ffill()
    # TO DO: this makes the fill gaps function obsolete and is a
    # very strong assumption

    # # for deaths, set the health state to the last known health state
    # pequiv_data["last_known_health"] = pequiv_data.groupby("pid")["health"].transform(
    #     "last"
    # )
    # pequiv_data.loc[
    #     (pequiv_data["death event"] == 1) & (pequiv_data["health"].isna()), "health"
    # ] = pequiv_data.loc[
    #     (pequiv_data["death event"] == 1) & (pequiv_data["health"].isna()),
    #     "last_known_health",
    # ]

    # Drop individuals without any health state information
    pequiv_data = pequiv_data[(pequiv_data["health"].notna())]

    return pequiv_data


# =====================================================================================
# Auxiliary
# =====================================================================================


def fill_health_gaps_vectorized(df):
    """Fill gaps where the first and last known health state are identical.

    Parameters:
        df (DataFrame): The DataFrame containing the "health" column.

    Returns:
        DataFrame: The modified DataFrame with filled health state gaps.

    """
    ffilled = df.groupby("pid")["health"].ffill()
    bfilled = df.groupby("pid")["health"].bfill()

    agreeing_mask = ffilled == bfilled
    df["health"] = np.where(df["health"].isna() & agreeing_mask, ffilled, df["health"])

    return df


def create_start_age_and_health(df):
    """Determine the starting age and health state for each "pid".

    Parameters:
        df (DataFrame): The DataFrame containing "pid", "age", and "syear" columns.

    Returns:
        DataFrame: The modified DataFrame with "start_age" and "start_health" columns.

    """
    df = df.reset_index()
    df["start age"] = df.groupby("pid")["age"].transform("min")
    idx = df.groupby("pid")["syear"].idxmin()

    df["start health"] = np.nan
    df.loc[idx, "start health"] = df.loc[idx, "health"]
    df["start health"] = df.groupby("pid")["start health"].transform("max")
    df = df.set_index(["pid", "syear"])

    return df


def create_interaction_columns(df, columns, specs):
    """Create interaction indicator columns based on two 0-1-columns in the DataFrame.
    Adds start_ prefix if necessary.

    Parameters:
        df (DataFrame): The DataFrame to modify.
        columns (tuple): A tuple containing the two column names.
        specs (dict): The specifications dictionary.

    Returns:
        DataFrame: The modified DataFrame with new interaction columns.

    """
    column1, column2 = columns

    # Get the labels for the columns (start variables get the same labels
    # as the original variables)
    col1_labels = specs[f"{column1}_labels".replace("start ", "")]
    col2_labels = specs[f"{column2}_labels".replace("start ", "")]

    for val1 in (0, 1):
        for val2 in (0, 1):
            col_name = f"{col1_labels[val1]} {col2_labels[val2]}"
            if "start" in column1 or "start" in column2:
                col_name = f"start {col_name}"
            df[col_name] = (df[column1] == val1) & (df[column2] == val2)

    return df


def create_interaction_columns_three_health_states(df, columns, specs):
    """Create interaction indicator columns based on two 0-1-columns in the DataFrame.
    Adds start_ prefix if necessary.

    Parameters:
        df (DataFrame): The DataFrame to modify.
        columns (tuple): A tuple containing the two column names.
        specs (dict): The specifications dictionary.

    Returns:
        DataFrame: The modified DataFrame with new interaction columns.

    """
    column1, column2 = columns

    # Grab the label dictionaries from specs (e.g. {0: 'bad', 1: 'medium', 2: 'good'})
    col1_labels = specs[f"{column1}_labels_three".replace("start ", "")]
    col2_labels = specs[f"{column2}_labels".replace("start ", "")]

    # Decide which states to iterate over based on which column is "health"
    # (0,1,2) or is something else like "education" (0,1).
    if "health" in column1:
        val1_states = [0, 1, 2]
    else:
        val1_states = [0, 1]

    val2_states = [0, 1]

    # Now create columns for every combination
    for val1 in val1_states:
        for val2 in val2_states:
            # Label name
            col_name = f"{col1_labels[val1]} {col2_labels[val2]}"

            # If it's a 'start' variable, you can prepend 'start '
            if "start " in column1 or "start " in column2:
                col_name = f"start {col_name}"

            # Create boolean indicator
            df[col_name] = (df[column1] == val1) & (df[column2] == val2)

    return df


def create_health_columns(df, column1, specs):
    """Create interaction indicator columns based on two 0-1-columns in the DataFrame.
    Adds start_ prefix if necessary.

    Parameters:
        df (DataFrame): The DataFrame to modify.
        columns (tuple): A tuple containing the two column names.
        specs (dict): The specifications dictionary.

    Returns:
        DataFrame: The modified DataFrame with new interaction columns.

    """
    # Grab the label dictionaries from specs (e.g. {0: 'bad', 1: 'medium', 2: 'good'})
    col1_labels = specs[f"{column1}_labels".replace("start ", "")]

    # Decide which states to iterate over based on which column is "health"
    # (0,1,2) or is something else like "education" (0,1).
    val1_states = [0, 1]

    # Now create columns for every combination
    for val1 in val1_states:
        # Label name
        col_name = f"{col1_labels[val1]}"

        # If it's a 'start' variable, you can prepend 'start '
        if "start " in column1:
            col_name = f"start {col_name}"

        # Create boolean indicator
        df[col_name] = df[column1] == val1

    return df


def create_health_columns_three_health_states(df, column1, specs):
    """Create interaction indicator columns based on two 0-1-columns in the DataFrame.
    Adds start_ prefix if necessary.

    Parameters:
        df (DataFrame): The DataFrame to modify.
        columns (tuple): A tuple containing the two column names.
        specs (dict): The specifications dictionary.

    Returns:
        DataFrame: The modified DataFrame with new interaction columns.

    """
    # Grab the label dictionaries from specs (e.g. {0: 'bad', 1: 'medium', 2: 'good'})
    col1_labels = specs[f"{column1}_labels_three".replace("start ", "")]

    # Decide which states to iterate over based on which column is "health"
    # (0,1,2) or is something else like "education" (0,1).
    if "health" in column1:
        val1_states = [0, 1, 2]
    else:
        val1_states = [0, 1]

    # Now create columns for every combination
    for val1 in val1_states:
        # Label name
        col_name = f"{col1_labels[val1]}"

        # If it's a 'start' variable, you can prepend 'start '
        if "start " in column1:
            col_name = f"start {col_name}"

        # Create boolean indicator
        df[col_name] = df[column1] == val1

    return df
