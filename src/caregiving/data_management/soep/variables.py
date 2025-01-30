"""Create SOEP variables."""

import numpy as np
import pandas as pd

PGEMPLST_UNEMPLOYED = 5
PGEMPLST_PART_TIME = 2
PGEMPLST_FULL_TIME = 1
PGSTIB_RETIREMENT = 13

PGSBIL_FACHHOCHSCHULREIFE = 3
PGSBIL_ABITUR = 4


# =====================================================================================
# Education
# =====================================================================================


def create_education_type(data):
    """This function creates a education type from pgpsbil in soep-pgen.

    The function uses a two category split of the population, encoding 1 if an
    individual has at least Fachhochschulreife.

    """
    data = data[data["pgpsbil"].notna()]
    data["education"] = 0
    data.loc[data["pgpsbil"] == PGSBIL_FACHHOCHSCHULREIFE, "education"] = 1
    data.loc[data["pgpsbil"] == PGSBIL_ABITUR, "education"] = 1

    print(str(len(data)) + " left after dropping people with missing education values.")

    return data


# =====================================================================================
# Experience
# =====================================================================================


def create_experience_variable_with_cap(data, exp_cap):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience.

    It also enforces an experience cap.

    """
    # Create experience variable
    data = create_experience_variable(data)
    # Enforce experience cap
    data.loc[data["experience"] > exp_cap, "experience"] = exp_cap
    return data


def create_experience_variable(data):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience and rounds the sum."""
    data = sum_experience_variables(data)
    return data


def sum_experience_variables(data):
    """This function sums the experience variables pgexpft and pgexppt.

    Part time experience is weighted by 0.5. The function returns a new column
    experience.

    """
    invalid_ft_exp = data["pgexpft"] < 0
    invalid_pt_exp = data["pgexppt"] < 0

    # Initialize empty experience column
    data["experience"] = np.nan

    # Check if years of part plus full time exceed age minus 14
    # (not allowed to work before)
    max_exp = data["age"] - 14
    exp_exceeding = ((data["pgexpft"] + data["pgexppt"]) - max_exp).clip(lower=0)
    # Deduct exceeding experience from part time experience.
    # Assume if worked both, you worked full
    data.loc[:, "pgexppt"] -= exp_exceeding

    # If both are valid use the sum
    data.loc[~invalid_ft_exp & ~invalid_pt_exp, "experience"] = (
        data.loc[~invalid_ft_exp & ~invalid_pt_exp, "pgexpft"]
        + 0.5 * data.loc[~invalid_ft_exp & ~invalid_pt_exp, "pgexppt"]
    )
    # If only one is valid use the valid one
    data.loc[invalid_ft_exp & ~invalid_pt_exp, "experience"] = (
        0.5 * data.loc[invalid_ft_exp & ~invalid_pt_exp, "pgexppt"]
    )
    data.loc[~invalid_ft_exp & invalid_pt_exp, "experience"] = data.loc[
        ~invalid_ft_exp & invalid_pt_exp, "pgexpft"
    ]
    # If both are invalid drop observations
    data = data[data["experience"].notna()]
    print(
        str(len(data)) + " left after dropping people with invalid experience values."
    )
    return data


# =====================================================================================
# Working Hours
# =====================================================================================


def generate_working_hours(data):
    """This function creates a working hours variable from pgvebzeit in soep-pgen.

    This means working hours = contractual hours per week. The function drops
    observations where working hours are missing.

    """
    data = data.rename(columns={"pgvebzeit": "working_hours"})
    data = data[data["working_hours"] >= 0]
    print(str(len(data)) + " left after dropping people with missing working hours.")
    return data


# =====================================================================================
# Job Hire & Fire
# =====================================================================================


def generate_job_separation_var(data):
    """This function generates a job separation variable.

    The function creates a new column job_sep which is 1 if the individual got fired
    from the last job. It uses plb0304_h from the soep pl data.

    """
    data.loc[:, "job_sep"] = 0
    data.loc[data["plb0304_h"].isin([1, 3, 5]), "job_sep"] = 1
    return data


def determine_observed_job_offers(data, working_choices, was_fired_last_period):
    """Determine if a job offer is observed and if so what it is. The function
    implements the following rule:

    Assume lagged choice is working (column "lagged_choice" is in working_choices),
    then the state is fully observed:
        - If individual continues working (column "choice" is in working choices):
            - There exists a job offer, i.e. equal to 1
        - If individual does not continue working (column "choice" is not in
        working choices):
            - Individual got fired then job offer equal 0
            - Individual was not fired then job offer equal to 1

    Assume lagged choice not working (column "lagged_choice" not in working_choices),
    then the state is partially observed:
        - If choice is in working_choices, then the state is fully observed and there
        is a job offer
        - If choice is not in working choices, then one is not observed

    Lagged choice equal to 0 (retired), will be dropped as only choice equal to 0
    is allowed

    Therefore the unobserved job offer states are, where individuals are unemployed and
    remain unemployed or retire.
    We mark unobsorved states by a state value of -99
    """
    working_this_period = data["choice"].isin(working_choices)
    was_working_last_period = data["lagged_choice"].isin(working_choices)

    data["job_offer"] = -99

    # Individuals working have job offer equal to 1 and are fully observed
    data.loc[working_this_period, "job_offer"] = 1

    # Individuals who are unemployed or retired and are fired this period have job offer
    # equal to 0. This includes individuals with lagged choice unemployment, as they
    # might be interviewed after firing.
    maskfired = (~working_this_period) & was_fired_last_period & was_working_last_period
    data.loc[maskfired, "job_offer"] = 0

    # Everybody who was not fired is also fully observed an has an job offer
    mask_not_fired = (
        (~working_this_period) & (~was_fired_last_period) & was_working_last_period
    )
    data.loc[mask_not_fired, "job_offer"] = 1
    return data


# =====================================================================================
# Work Status
# =====================================================================================


def create_working_status(df):
    df["work_status"] = np.nan

    soep_empl_status = df["pgstib"]

    # assign employment choices
    df.loc[soep_empl_status != PGSTIB_RETIREMENT, "work_status"] = 1

    # assign retirement status
    df.loc[soep_empl_status == PGSTIB_RETIREMENT, "work_status"] = 0
    return df


def create_choice_variable(data):
    """This function creates the choice variable for the structural model.

    0: retirement, 1: unemployed, 2: part-time, 3: full-time

    This function assumes retirees with part-time employment as full-time retirees.

    """
    data["choice"] = np.nan
    soep_empl_choice = data["pgemplst"]
    soep_empl_status = data["pgstib"]
    # rv_ret_choice = merged_data["STATUS_2"]

    # assign employment choices
    data.loc[soep_empl_choice == PGEMPLST_UNEMPLOYED, "choice"] = 1
    data.loc[soep_empl_choice == PGEMPLST_PART_TIME, "choice"] = 2
    data.loc[soep_empl_choice == PGEMPLST_FULL_TIME, "choice"] = 3

    # assign retirement choice
    data.loc[soep_empl_status == PGSTIB_RETIREMENT, "choice"] = 0
    # merged_data.loc[rv_ret_choice == "RTB"] = 2
    data = data[data["choice"].notna()]
    return data


# =====================================================================================
# Partner State
# =====================================================================================


def create_partner_state(df, filter_missing=False):
    """0: no partner, 1: working-age partner, 2: retired partner"""
    # has to be done for both state and lagged state
    # people with a partner whose choice is not observed stay in this category
    df = create_working_status(df)
    df = merge_couples(df)

    df.loc[:, "partner_state"] = np.nan
    # no partner (no parid)
    df.loc[:, "partner_state"] = np.where(df["parid"] < 0, 0, df["partner_state"])
    # working-age partner (choice 0, 1, 3)
    # Assign working partners to working state
    df.loc[:, "partner_state"] = np.where(
        df["work_status_p"] == 1, 1, df["partner_state"]
    )
    # retired partner (choice 2)
    df.loc[:, "partner_state"] = np.where(
        df["work_status_p"] == 0, 2, df["partner_state"]
    )
    if filter_missing:
        # drop nans
        df = df[df["partner_state"].notna()]
        print(
            str(len(df))
            + " obs. after dropping people with a partner whose choice is not observed."
        )
    return df


def merge_couples(df):
    """This function merges couples based on the 'parid' identifier.

    Partner variables are market '_p' in the merged dataset.

    """
    df = df.reset_index()
    df_partners = df.copy()

    # Assign nans to negative parids to merge nans to obs
    df_partners.loc[df_partners["parid"] < 0, "parid"] = np.nan

    merged_data = pd.merge(
        df,
        df_partners,
        how="left",
        left_on=["hid", "syear", "parid"],
        right_on=["hid", "syear", "pid"],
        suffixes=("", "_p"),
    )
    merged_data.set_index(["pid", "syear"], inplace=True)

    print(str(len(merged_data)) + " observations after merging couples.")
    return merged_data
