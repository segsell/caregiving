"""Care mix moments from the German statistical office."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product

from caregiving.config import BLD, SRC

FEMALE = 2
MALE = 1
FINAL_AGE_GROUP = 99


TWO_ROWS = 2

# Prepare for future behavior of pandas with respect to downcasting
pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003


def task_create_care_mix_moments(
    path_to_data: Path = SRC / "data/statistical_office" / "22421-0001_$F_modified.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "moments"
    / "statistical_office_care_mix.csv",
):
    """Create care mix moments from statistical office data.

    out = {
        f"share_nursing_home_{sex_string}_age_60_to_65": _nursing_home[12],
        f"share_formal_home_care_{sex_string}_age_60_to_65": _only_home_care[12],
        f"share_combination_care_{sex_string}_age_60_to_65": _combination_care[12],
        f"share_pure_informal_care_{sex_string}_age_60_to_65": _only_informal[12],
        #
        f"share_nursing_home_{sex_string}_age_65_to_70": _nursing_home[13],
        f"share_formal_home_care_{sex_string}_age_65_to_70": _only_home_care[13],
        f"share_combination_care_{sex_string}_age_65_to_70": _combination_care[13],
        f"share_pure_informal_care_{sex_string}_age_65_to_70": _only_informal[13],
        #
    }

    Args:
        path_to_data (str): Path to the dataset file.
        path_to_save (str): Path to save the moments.

    Returns:
        dict: A dictionary with the care mix moments by age bin for females.
        dict: A dictionary with the care mix moments by age bin for males.

    """
    df_wide = pd.read_csv(path_to_data, encoding="utf-8")

    # Transforming the dataframe from wide format to long format
    data = df_wide.melt(
        id_vars=["Versorgungsart", "sex", "age_group"],
        var_name="year",
        value_name="value",
    )

    data = data.rename(columns={"Versorgungsart": "type_of_care"})

    # Removing the "year_" prefix from the 'year' column and converting to int
    data["year"] = data["year"].str.replace("year_", "").astype(int)

    data["sex"] = (
        data["sex"]
        .replace({"männlich": 1, "weiblich": 2, "Insgesamt": 0, "insgesamt": 0})
        .astype(int)
    )

    age_group_replacements = {
        "unter 5 Jahre": 0,
        "5 bis unter 10 Jahre": 1,
        "10 bis unter 15 Jahre": 2,
        "15 bis unter 20 Jahre": 3,
        "20 bis unter 25 Jahre": 4,
        "25 bis unter 30 Jahre": 5,
        "30 bis unter 35 Jahre": 6,
        "35 bis unter 40 Jahre": 7,
        "40 bis unter 45 Jahre": 8,
        "45 bis unter 50 Jahre": 9,
        "50 bis unter 55 Jahre": 10,
        "55 bis unter 60 Jahre": 11,
        "60 bis unter 65 Jahre": 12,
        "65 bis unter 70 Jahre": 13,
        "70 bis unter 75 Jahre": 14,
        "75 bis unter 80 Jahre": 15,
        "80 bis unter 85 Jahre": 16,
        "85 bis unter 90 Jahre": 17,
        "90 bis unter 95 Jahre": 18,
        "95 Jahre und mehr": 19,
        "Insgesamt": 99,
    }
    data["age_group"] = data["age_group"].replace(age_group_replacements)

    # Updating 'type_of_care' column values
    type_of_care_mapping = {
        "Versorgung zu Hause allein durch Angehörige": "only_informal",
        "Versorgung zu Hause mit/durch ambul. Pflegedienste": "informal_and_home_care",
        "Vollstationäre Pflege": "nursing_home",
        "Pflegegrad 1 und teilstationäre Pflege": "care_degree_one_a",
        "Pflegegrad 1 u. nur landesrechtl. bzw. ohne Leist.": "care_degree_one_b",
        "Insgesamt": "total",
    }
    data["type_of_care"] = data["type_of_care"].map(type_of_care_mapping)

    data = data.rename(columns={"value": "number"})

    dict_female = create_moments_by_age_bins(data, sex=2, year=2017)

    series = pd.Series(dict_female) / 100
    series.to_csv(path_to_save, index=True)


def create_moments_by_age_bins(df, sex, year):
    """Create care mix moments by age bins as share of all care modes.

    Args:
        df (DataFrame): The dataset containing the data.
        sex (int): Sex category (0 for 'Insgesamt', 1 for 'männlich', 2 for 'weiblich').
        year (int): The year for which to plot the data.

    Returns:
        dict: A dictionary with the care mix moments by age bin.

    """
    sex_string = "female" if sex == FEMALE else "male"

    # Filter the DataFrame based on the provided sex and year
    data = df[(df["sex"] == sex) & (df["year"] == year)]
    data = data[
        ~data["type_of_care"].isin(["total", "care_degree_one_a", "care_degree_one_b"])
    ]
    data["number"] = data["number"].astype(int)

    data["age_group"] = pd.Categorical(data["age_group"], ordered=True)
    data = data[data["age_group"] != FINAL_AGE_GROUP]  # Filter out the "all" age group
    data = data.sort_values("age_group")

    # Create subsets
    only_informal_full = data[(data["type_of_care"] == "only_informal")]
    informal_and_home_care_full = data[
        (data["type_of_care"] == "informal_and_home_care")
    ]
    nursing_home_full = data[(data["type_of_care"] == "nursing_home")]

    only_informal = condense_last_two_rows(only_informal_full)
    informal_and_home_care = condense_last_two_rows(informal_and_home_care_full)
    nursing_home = condense_last_two_rows(nursing_home_full)

    share_combination_care_in_home_care = [
        0.4026772102524974,
        0.4990651870170441,
        0.5113503611840557,
        0.5425971988562102,
        0.5952632679491388,
        0.44028522574716733,
    ]

    share_informal_by_own_children = 0.5

    _only_informal = only_informal["number"] * share_informal_by_own_children

    _informal_and_home_care = informal_and_home_care["number"].tolist()
    _nursing_home = nursing_home["number"]

    _combination_care = 0.4 * np.array(_informal_and_home_care)
    _only_home_care = 0.6 * np.array(_informal_and_home_care)

    for val in (14, 15, 16, 17, 18):
        i = val - 14
        _combination_care[val] = (
            share_combination_care_in_home_care[i] * _informal_and_home_care[val]
        )
        _only_home_care[val] = (
            1 - share_combination_care_in_home_care[i]
        ) * _informal_and_home_care[val]

    _combination_care = _combination_care * share_informal_by_own_children

    total_by_age_group = (
        _only_informal.reset_index(drop=True)
        + _nursing_home.reset_index(drop=True)
        + _only_home_care
        + _combination_care
    )

    pure_formal_care = _only_home_care + np.array(_nursing_home)

    _only_informal = _only_informal.tolist()
    _nursing_home = _nursing_home.tolist()
    _only_home_care = _only_home_care.tolist()
    _combination_care = _combination_care.tolist()

    _only_informal = _calculate_percentage_share(_only_informal, total_by_age_group)
    _combination_care = _calculate_percentage_share(
        _combination_care,
        total_by_age_group,
    )
    _pure_formal_care = _calculate_percentage_share(
        pure_formal_care,
        total_by_age_group,
    )

    _only_home_care = _calculate_percentage_share(_only_home_care, total_by_age_group)
    _nursing_home = _calculate_percentage_share(_nursing_home, total_by_age_group)

    return create_moments_dictionary(
        sex=sex_string,
        start_index=12,
        end_age=90,
        age_step=5,
        formal_care=_pure_formal_care,
        combination_care=_combination_care,
        only_informal=_only_informal,
    )


def create_moments_dictionary(
    sex,
    start_index,
    end_age,
    age_step,
    formal_care,
    combination_care,
    only_informal,
):
    """Create moments dictionary for the care mix by age bin.

    Informal care is assumed to be provided by own children only. Other informal
    caregivers are not considered in the data. The share of combination care in
    home care is assumed to be constant across age groups.

    Args:
        sex(str): A string representing the sex, used in the key
            (e.g., "male" or "female").
        start_index (int): The starting index for the lists (nursing_home, etc.)
            corresponding to the age bin "60 to 65".
        end_age (int): The age to end at before the "95+" category.
        age_step (int): The step size between age bins.
        formal_care (list): A list containing the share of individuals receiving
            formal care, i.e. formal home care or nursing home, by age bin.
        combination_care (list): A list containing the share of individuals
            receiving a combination of formal and informal care by age bin.
        only_informal (list): A list containing the share of individuals receiving
            only informal care, assumed to be from children, by age bin.

    Returns:
        dict: A dictionary with the care mix moments by age bin.

    """
    age_bins = [(age, age + age_step) for age in range(60, end_age, age_step)] + [
        ("90", "+"),
    ]
    out = {}

    for i, (start, end) in enumerate(age_bins, start=start_index):
        age_key = f"{start}_to_{end}" if isinstance(end, int) else f"{start}+"

        out[f"share_informal_{sex}_age_{age_key}"] = (
            only_informal[i] if i < len(only_informal) else []
        )
        out[f"share_formal_{sex}_age_{age_key}"] = (
            formal_care[i] if i < len(formal_care) else []
        )
        out[f"share_combination_{sex}_age_{age_key}"] = (
            combination_care[i] if i < len(combination_care) else []
        )

    return out


def _calculate_percentage_share(numbers, total):
    return [n / total[i] * 100 if total[i] > 0 else 0 for i, n in enumerate(numbers)]


def create_moments_dictionary_all(
    sex,
    start_index,
    end_age,
    age_step,
    nursing_home,
    only_home_care,
    combination_care,
    only_informal,
):
    """Create moments dictionary for the care mix by age bin.

    Informal care is assumed to be provided by own children only. Other informal
    caregivers are not considered in the data. The share of combination care in
    home care is assumed to be constant across age groups.

    Args:
        sex(str): A string representing the sex, used in the key
            (e.g., "male" or "female").
        start_index (int): The starting index for the lists (nursing_home, etc.)
            corresponding to the age bin "60 to 65".
        end_age (int): The age to end at before the "95+" category.
        age_step (int): The step size between age bins.
        nursing_home (list): A list containing the share of individuals in nursing
            homes by age bin.
        only_home_care (list): A list containing the share of individuals receiving
            only formal home care by age bin.
        combination_care (list): A list containing the share of individuals
            receiving a combination of formal and informal care by age bin.
        only_informal (list): A list containing the share of individuals receiving
            only informal care, assumed to be from children, by age bin.

    Returns:
        dict: A dictionary with the care mix moments by age bin.

    """
    age_bins = [(age, age + age_step) for age in range(60, end_age, age_step)] + [
        ("90", "+"),
    ]
    out = {}

    for i, (start, end) in enumerate(age_bins, start=start_index):
        age_key = f"{start}_to_{end}" if isinstance(end, int) else f"{start}+"

        out[f"share_nursing_home_{sex}_age_{age_key}"] = (
            nursing_home[i] if i < len(nursing_home) else []
        )
        out[f"share_formal_home_care_{sex}_age_{age_key}"] = (
            only_home_care[i] if i < len(only_home_care) else []
        )
        out[f"share_combination_care_{sex}_age_{age_key}"] = (
            combination_care[i] if i < len(combination_care) else []
        )
        out[f"share_pure_informal_care_{sex}_age_{age_key}"] = (
            only_informal[i] if i < len(only_informal) else []
        )

    return out


def condense_last_two_rows(df, column_to_sum="number"):
    """Condenses the last two rows of a DataFrame.

    Sum up the specified column's values for these rows and removing the
    last row.

    Args:
        df (pd.DataFrame): The pandas DataFrame to be condensed.
        column_to_sum (str): The name of the column for which the last two
        rows' values will be summed.

    Returns:
        pd.DataFrame: A new pandas DataFrame with the last two rows condensed into one.

    """
    # Ensure the DataFrame is not empty and has more than one row
    if df.empty or len(df) < TWO_ROWS:
        raise ValueError("DataFrame must have at least two rows to condense.")

    # Sum the values of the specified column for the last two rows
    sum_last_two = df[column_to_sum].iloc[-2:].sum()

    # Update the second last row with this summed value
    df.loc[df.index[-2], column_to_sum] = sum_last_two

    # Remove the last row and return the updated DataFrame
    return df.iloc[:-1].copy()


def _check_age_group_sums(df, year, sex, type_of_care):
    """Check if numbers in age groups equal total.

    The numbers of different age groups (excluding 'total') sum to the 'total'
    age group for a given year, sex, and 'type_of_care' category.

    Args:
        df (DataFrame): The dataset containing the data.
        year (int): The year for which to perform the check.
        sex (int): Sex category (0 for 'Insgesamt', 1 for 'männlich', 2 for 'weiblich').
        type_of_care (str): The 'type_of_care' category to check.

    Returns:
        bool: True if the sums match, False otherwise.

    """
    # Filter the DataFrame based on the provided year, sex, and type_of_care
    filtered_df = df[
        (df["year"] == year) & (df["sex"] == sex) & (df["type_of_care"] == type_of_care)
    ]

    # Filter for age groups that are not 'total'
    age_groups = filtered_df[filtered_df["age_group"] != "total"]

    # Calculate the sum of numbers for age groups (excluding 'total')
    age_group_sum = age_groups["number"].sum()

    # Filter for 'total' age group
    total_age_group = filtered_df[filtered_df["age_group"] == "total"]

    # Get the number for the 'total' age group
    total_number = total_age_group["number"].to_numpy()[0]

    # Check if the sums match
    return age_group_sum == total_number
