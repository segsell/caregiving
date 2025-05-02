import numpy as np
import pandas as pd

from caregiving.model.shared import RETIREMENT, UNEMPLOYED

# =====================================================================================
# Filter Data
# =====================================================================================


def filter_years(df, start_year, end_year):
    df = df.loc[(slice(None), range(start_year, end_year + 1)), :]
    print(
        str(len(df))
        + " left after dropping people outside of estimation years "
        + f"{start_year} - {end_year}."
    )
    return df


def filter_below_age(df, age):
    df = df[df["age"] >= age]
    print(
        str(len(df)) + " left after dropping people under " + str(age) + " years old."
    )
    return df


def filter_above_age(df, age):
    df = df[df["age"] <= age]
    print(str(len(df)) + " left after dropping people over " + str(age) + " years old.")
    return df


def recode_sex(df):
    """Recode sex to 0(men) and 1(women), from SOEP definition 1(men) and 2(women)."""
    df.loc[:, "sex"] = df["sex"] - 1
    return df


def filter_data(merged_data, specs, lag_and_lead_buffer_years=True, event_study=False):
    """This function filters the data according to the model setup.

    Specifically, it filters out young people, women (if no_women=True), and years
    outside of estimation range. It leaves one year younger and one year below in the
    sample to construct lagged_choice.

    """
    if event_study:
        start_age = specs["start_age_event_study"]
        start_year = specs["start_year_event_study"]
        end_year = specs["end_year_event_study"]
    else:
        start_age = specs["start_age"]
        start_year = specs["start_year"]
        end_year = specs["end_year"]

    merged_data = filter_below_age(merged_data, start_age - 1)

    merged_data = recode_sex(merged_data)

    if lag_and_lead_buffer_years:
        start_year = start_year - 1
        end_year = end_year + 1

    merged_data = filter_years(merged_data, start_year, end_year)
    return merged_data


# =====================================================================================
# Lagged Variables
# =====================================================================================


def span_dataframe(df, start_year, end_year):
    """This function spans the DataFrame over the whole observation period from
    start_year to end_year, to create lagged and lead variables."""
    # Create full index with all possible combinations of pid and syear. Otherwise if
    # we just shift the data, people having missing years in their observations get
    # assigned variables from multi years back.
    pid_indexes = df.index.get_level_values(0).unique()

    full_index = pd.MultiIndex.from_product(
        [pid_indexes, range(start_year, end_year + 1)],
        names=["pid", "syear"],
    )
    full_container = pd.DataFrame(
        index=full_index, data=np.nan, dtype=float, columns=df.columns
    )
    full_container.update(df)

    if "hid" in full_container.columns.values:
        full_container["hid"] = full_container.groupby(["pid"])["hid"].transform("last")
    return full_container


def create_lagged_and_lead_variables(
    merged_data, specs, lead_job_sep=False, event_study=False
):
    """This function creates the lagged choice variable and drops missing lagged
    choices."""

    if event_study:
        start_year = specs["start_year_event_study"]
        end_year = specs["end_year_event_study"]
    else:
        start_year = specs["start_year"]
        end_year = specs["end_year"]

    full_container = span_dataframe(merged_data, start_year - 1, end_year + 1)

    full_container["lagged_choice"] = full_container.groupby(["pid"])["choice"].shift()

    if lead_job_sep:
        full_container["job_sep_this_year"] = full_container.groupby(["pid"])[
            "job_sep"
        ].shift(-1)

    merged_data = full_container[full_container["lagged_choice"].notna()]

    if lead_job_sep:
        merged_data = merged_data[merged_data["job_sep_this_year"].notna()]

    # We now have observations with a valid lagged or lead variable but not with
    # actual valid state variables. Delete those by looking at the choice variable.
    merged_data = merged_data[merged_data["choice"].notna()]

    # We left too young people in the sample to construct lagged choice. Delete those
    # now.
    merged_data = merged_data[merged_data["age"] >= specs["start_age"]]

    print(str(len(merged_data)) + " left after filtering missing lagged choices.")
    return merged_data


# =====================================================================================
# Model Restrictions
# =====================================================================================


def enforce_model_choice_restriction(df, specs):
    """Filter the choice data according to the model setup.

    Specifically, it filters out people retire too early, work too long, or come back
    from retirement,

    """

    retired_values = np.asarray(RETIREMENT).ravel().tolist()
    unemployed_values = np.asarray(UNEMPLOYED).ravel().tolist()

    max_ret_age = specs["max_ret_age"]
    min_ret_age = specs["min_ret_age"]

    # Filter out people who are retired before min_ret_age
    # df = df[~((df["choice"] == 0) & (df["age"] < min_ret_age))]
    df = df[~(df["choice"].isin(retired_values) & (df["age"] < min_ret_age))]

    # df = df[~((df["lagged_choice"] == 0) & (df["age"] <= min_ret_age))]
    df = df[~(df["lagged_choice"].isin(retired_values) & (df["age"] <= min_ret_age))]

    # Filter out people who are working after max_ret_age
    # df = df[~((df["choice"] != 0) & (df["age"] >= max_ret_age))]
    df = df[~((~df["choice"].isin(retired_values)) & (df["age"] >= max_ret_age))]

    # Filter out people who have not retirement as lagged choice after max_ret_age
    # df = df[~((df["lagged_choice"] != 0) & (df["age"] > max_ret_age))]
    df = df[~((~df["lagged_choice"].isin(retired_values)) & (df["age"] > max_ret_age))]

    print(
        str(len(df))
        + " left after dropping people who are retired before "
        + str(min_ret_age)
        + " or working after "
        + str(max_ret_age)
        + "."
    )

    # Filter out people who come back from retirement
    # df = df[(df["lagged_choice"] != 0) | (df["choice"] == 0)]
    df = df[
        (~df["lagged_choice"].isin(retired_values))
        | (df["choice"].isin(retired_values))
    ]

    # # Filter out people who are unemployed after sra
    post_sra = df["age"] - df["policy_state_value"]
    df = df[~((post_sra >= 0) & (df["choice"].isin(unemployed_values)))]
    df = df[~((post_sra >= 1) & (df["lagged_choice"].isin(unemployed_values)))]
    df = df[~((df["age"] >= specs["min_SRA"]) & (df["choice"].isin(unemployed_values)))]
    df = df[
        ~(
            (df["age"] > specs["min_SRA"])
            & (df["lagged_choice"].isin(unemployed_values))
        )
    ]

    # recode choice → retired for post‐SRA unemployed
    # mask1 = (post_sra >= 0) & df["choice"].isin(unemployed_values)
    # df.loc[mask1, "choice"] = 0
    # # recode lagged_choice → retired for post‐SRA unemployed
    # mask2 = (post_sra >= 1) & df["lagged_choice"].isin(unemployed_values)
    # df.loc[mask2, "lagged_choice"] = 0
    # # recode choice → retired once age ≥ min_SRA
    # mask3 = (df["age"] >= specs["min_SRA"]) & df["choice"].isin(unemployed_values)
    # df.loc[mask3, "choice"] = 0
    # # recode lagged_choice → retired once age > min_SRA
    # mask4 = (df["age"] > specs["min_SRA"]) & df[
    # "lagged_choice"].isin(unemployed_values)
    # df.loc[mask4, "lagged_choice"] = 0

    print(str(len(df)) + " left after dropping people who come back from retirement.")
    return df
