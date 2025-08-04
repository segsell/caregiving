"""Create SOEP variables."""

import numpy as np
import pandas as pd

from caregiving.utils import table

# from src.caregiving.model.shared import FULL_TIME, PART_TIME, UNEMPLOYED, RETIRED

PGEMPLST_UNEMPLOYED = 5
PGEMPLST_PART_TIME = 2
PGEMPLST_FULL_TIME = 1
PGSTIB_RETIREMENT = 13

PGSBIL_FACHHOCHSCHULREIFE = 3
PGSBIL_ABITUR = 4

KIDAGE_THRESHOLD_UPPER = 10

DECEASED = 4

DOES_NOT_APPLY = -2

MISSING_VALUE = -99

ANSWER_NO = 5


# =====================================================================================
# Retirement
# =====================================================================================


def create_policy_state(df, specs):

    # Step 1: Assign SRA from year of birth and survey year
    df["_policy_state_value"] = assign_policy_state_by_gebjahr(
        df["gebjahr"]
    )  # e.g., 65, 66, 67

    # Step 2: Map SRA values to grid (e.g., 65 → 0, 66 → 1, 67 → 2)
    df["policy_state_value"], df["policy_state"] = modify_policy_state(
        df["_policy_state_value"], specs
    )

    return df


def assign_policy_state_by_gebjahr(gebjahr):
    """This function creates the policy state according to the 2007 reform."""

    # Default state is 67
    policy_state = pd.Series(index=gebjahr.index, data=67, dtype=float)
    # Create masks for everyone born before 1964
    mask1 = (gebjahr <= 1964) & (gebjahr >= 1958)  # noqa: PLR2004
    mask2 = (gebjahr <= 1958) & (gebjahr >= 1947)  # noqa: PLR2004
    mask3 = gebjahr < 1947  # noqa: PLR2004
    policy_state.loc[mask1] = 67 - 2 / 12 * (1964 - gebjahr[mask1])  # noqa: PLR2004
    policy_state.loc[mask2] = 66 - 1 / 12 * (1958 - gebjahr[mask2])  # noqa: PLR2004
    policy_state.loc[mask3] = 65

    return policy_state


# def assign_policy_state_by_gebjahr_and_syear(df):
#     """Assigns statutory retirement age (SRA) by year of birth and survey year,
#     reflecting both pre- and post-2007 German pension rules.

#     Returns integer SRAs: 65, 66, or 67.
#     Supports both MultiIndex and column-based 'syear'.
#     """

#     # Get syear from column or index
#     if "syear" in df.columns:
#         syear = df["syear"]
#     elif isinstance(df.index, pd.MultiIndex) and "syear" in df.index.names:
#         syear = df.index.get_level_values("syear")
#     else:
#         raise ValueError("'syear' must be a column or a MultiIndex level in df.")

#     gebjahr = df["gebjahr"]
#     policy_state = pd.Series(index=gebjahr.index, dtype=int)

#     # Case 1: Pre-2007 — SRA is 65
#     mask_pre = syear < 2007  # noqa: PLR2004
#     policy_state.loc[mask_pre] = 65

#     # Case 2: Post-2007 — apply cohort-specific changes
#     mask_post = ~mask_pre
#     geb_post = gebjahr[mask_post]
#     temp_state = pd.Series(index=geb_post.index, data=67.0)

#     # Gradual increase to 67 (1958–1964)
#     mask1 = (geb_post <= 1964) & (geb_post >= 1958)  # noqa: PLR2004
#     temp_state.loc[mask1] = 67 - 2 / 12 * (1964 - geb_post[mask1])

#     # Gradual increase to 66 (1947–1957)
#     mask2 = (geb_post < 1958) & (geb_post >= 1947)  # noqa: PLR2004
#     temp_state.loc[mask2] = 66 - 1 / 12 * (1958 - geb_post[mask2])

#     # Born before 1947
#     mask3 = geb_post < 1947  # noqa: PLR2004
#     temp_state.loc[mask3] = 65

#     # Round to nearest integer year
#     policy_state.loc[mask_post] = temp_state.round().astype(int)
#     # policy_state.loc[mask_post] = (temp_state + 0.5).astype(int)

#     return policy_state


def modify_policy_state(policy_states, specs):
    """This function rounds policy state to the closest multiple of the policy
    expectations process grid size.

    min_SRA is set by the assumption of the belief process in the model.

    """
    min_SRA = specs["min_SRA"]
    SRA_grid_size = specs["SRA_grid_size"]
    policy_states = policy_states - min_SRA
    policy_id = np.around(policy_states / SRA_grid_size).astype(int)
    policy_states_values = min_SRA + policy_id * SRA_grid_size

    return policy_states_values, policy_id


# =====================================================================================
# Education
# =====================================================================================


def create_education_type(data, drop_missing=True):
    """This function creates a education type from pgpsbil in soep-pgen.

    The function uses a two category split of the population, encoding 1 if an
    individual has at least Fachhochschulreife.

    """
    data["education"] = 0
    data.loc[data["pgpsbil"] < 0, "education"] = np.nan

    data.loc[data["pgpsbil"] == PGSBIL_FACHHOCHSCHULREIFE, "education"] = 1
    data.loc[data["pgpsbil"] == PGSBIL_ABITUR, "education"] = 1

    if drop_missing:
        data = data[data["education"].notna()]

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


def create_experience_variable(data, drop_invalid=True):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience and rounds the sum."""
    data = sum_experience_variables(data, drop_invalid=drop_invalid)
    return data


def sum_experience_variables(data, drop_invalid=True):
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
    if drop_invalid:
        data = data[data["experience"].notna()]

        print(
            str(len(data))
            + " left after dropping people with invalid experience values."
        )

    return data


# =====================================================================================
# Working Hours
# =====================================================================================


def generate_working_hours(data, include_actual_hours=False, drop_missing=True):
    """This function creates a working hours variable from pgvebzeit in soep-pgen.

    This means working hours = contractual hours per week. The function drops
    observations where working hours are missing.

    """
    data = data.rename(columns={"pgvebzeit": "working_hours"})

    if drop_missing:
        data = data[data["working_hours"] >= 0]
        print(
            str(len(data)) + " left after dropping people with missing working hours."
        )
    else:
        data.loc[data["working_hours"] == DOES_NOT_APPLY, "working_hours"] = 0
        data.loc[data["working_hours"] < 0, "working_hours"] = np.nan
        # data = data[data["working_hours"] >= 0]

        print(
            str(len(data)) + " left after dropping people with missing working hours."
        )

    if include_actual_hours:
        data = data.rename(columns={"pgtatzeit": "working_hours_actual"})
        data.loc[
            data["working_hours_actual"] == DOES_NOT_APPLY, "working_hours_actual"
        ] = 0
        data.loc[data["working_hours_actual"] < 0, "working_hours_actual"] = np.nan

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


# =====================================================================================
# Age of youngest child
# =====================================================================================


def create_kidage_youngest(df):
    """Create age of the youngest child."""
    df["syear"] = df.index.get_level_values("syear")

    child_cols = [col for col in df.columns if col.startswith("kidgeb")]
    for col in child_cols:
        df.loc[df[col] < 0, col] = np.nan

    print(df["syear"].value_counts(dropna=False))

    # print("Frequency table for pmonin:")
    # print(df["pmonin"].value_counts(dropna=False))
    # # Replace negative values in pmonin with NaN
    # df.loc[df["pmonin"] < 0, "pmonin"] = np.nan

    # # Calculate percentage of valid pmonin observations
    # valid_syear_mask = df["syear"].notna() & (df["syear"] >= 0)
    # total_obs = valid_syear_mask.sum()

    # valid_pmonin_mask = df["pmonin"].notna() & (df["pmonin"] > 0) & valid_syear_mask
    # valid_pmonin = valid_pmonin_mask.sum()

    # percentage_valid_pmonin = (
    #     (valid_pmonin / total_obs * 100) if total_obs > 0 else np.nan
    # )

    valid_birthyears = df[child_cols].where(df[child_cols] > 0)
    df["yng_birthyear"] = valid_birthyears.max(axis=1)

    df["kidage_youngest"] = df["syear"] - df["yng_birthyear"]

    #  Modify kidage_youngest:
    # - Values >= 10 are set to 10.
    # - Values < 0 are set to 10.
    # - NA values are set to 10.
    # df["kidage_youngest"] = np.where(
    #     (df["kidage_youngest"].isna())
    #     | (df["kidage_youngest"] >= KIDAGE_THRESHOLD_UPPER)
    #     | (df["kidage_youngest"] < 0),
    #     KIDAGE_THRESHOLD_UPPER,
    #     df["kidage_youngest"],
    # )
    df["kidage_youngest"] = np.where(
        # (df["kidage_youngest"].isna()) | (df["kidage_youngest"] < 0),
        (df["kidage_youngest"].isna()),
        # | (df["kidage_youngest"] >= KIDAGE_THRESHOLD_UPPER),
        MISSING_VALUE,
        df["kidage_youngest"],
    )

    print(
        "Distribution of kidage youngest:"
        + str(df["kidage_youngest"].value_counts(dropna=False))
    )

    # Clean up
    df.drop(columns=["syear", "yng_birthyear"] + child_cols, inplace=True)

    return df


# =====================================================================================
# Health (good/bad)
# =====================================================================================


# def create_health_var(data, drop_missing=True):
#     """Create the health variable in the soep-PEQUIV dataset.

#     - m11126: Self-Rated Health Status
#     - m11124: Disability Status of Individual

#     The function replaces the following values in the health variables:
#     - [-1] keine Angabe
#     - [-2] trifft nicht zu
#     - [-5] in Fragebogenversion nicht enthalten

#     with np.nan and converts the variables to float.

#     The function uses a two category split of the population, encoding 1 if an
#     individual has good health and 0 if an individual has bad health.

#     """
#     data = data[data["m11126"] >= 0]
#     print(
#         str(len(data))
#         + " observations left after dropping people with missing health data."
#     )

#     data = data[data["m11124"] >= 0]
#     print(
#         str(len(data))
#         + " observations left after dropping people with missing disability data."
#     )

#     # create health state = 0 if bad health, 1 if good health
#     data["health"] = 0
#     data.loc[data["m11126"].isin([1, 2, 3]) & data["m11124"].isin([0]), "health"] = 1

#     return data


def create_health_var_good_bad(data, drop_missing=True):
    """
    Create the health variable in the soep-PEQUIV dataset.

    Good health = 1, bad health = 0.

    Variables:
    - m11126: Self-Rated Health Status (1-5 for valid responses)
    - m11124: Disability Status of Individual (0 or 1 for valid responses)

    """

    if drop_missing:
        data = data[data["m11126"] >= 0]
        print(
            f"{len(data)} observations left after dropping people with "
            "missing health data."
        )
        data = data[data["m11124"] >= 0]
        print(
            f"{len(data)} observations left after dropping people with "
            "missing disability data."
        )
    else:
        data.loc[data["m11126"] < 0, "m11126"] = np.nan
        data.loc[data["m11124"] < 0, "m11124"] = np.nan

    data["health"] = np.nan
    data.loc[(data["m11126"].isin([4, 5])) | (data["m11124"] == 1), "health"] = 0
    data.loc[(data["m11126"].isin([1, 2, 3])) & (data["m11124"] == 0), "health"] = 1

    return data


def create_health_var_good_medium_bad(data, drop_missing=True):
    """
    Create the health variable in the soep-PEQUIV dataset.

    Good health = 2, medium (satisfactory) health 1, bad health = 0.

    Variables:
    - m11126: Self-Rated Health Status (1-5 for valid responses)
    - m11124: Disability Status of Individual (0 or 1 for valid responses)

    """

    if drop_missing:
        data = data[data["m11126"] >= 0]
        print(
            f"{len(data)} observations left after dropping people with "
            "missing health data."
        )
        data = data[data["m11124"] >= 0]
        print(
            f"{len(data)} observations left after dropping people with "
            "missing disability data."
        )
    else:
        data.loc[data["m11126"] < 0, "m11126"] = np.nan
        data.loc[data["m11124"] < 0, "m11124"] = np.nan

    data["health"] = np.nan
    data.loc[(data["m11126"].isin([4, 5])) | (data["m11124"] == 1), "health"] = 0
    data.loc[(data["m11126"].isin([3])), "health"] = 1
    data.loc[(data["m11126"].isin([1, 2])) & (data["m11124"] == 0), "health"] = 2

    return data


def clean_health_create_states(data):
    """Create lead and lagged health variable.

    The function replaces the health variable with 1 if both the previous and next
    health are 1.

    """

    # replace health with 1 if both previous and next health are 1
    data["lag_health"] = data.groupby(["pid"])["health"].shift(1)
    data["lead_health"] = data.groupby(["pid"])["health"].shift(-1)

    # one year bad health in between two years of good health
    # is still considered good health
    data.loc[
        (data["lag_health"] == 1) & (data["lead_health"] == 1),
        "health",
    ] = 1
    data.loc[
        (data["lag_health"] == 2) & (data["lead_health"] == 2),  # noqa: PLR2004
        "health",
    ] = 2

    # update lead_healt
    data["lead_health"] = data.groupby(["pid"])["health"].shift(-1)
    # update lag_health
    data["lag_health"] = data.groupby(["pid"])["health"].shift(1)

    return data


def create_nursing_home(data):
    """Create nursing home variable.

    https://paneldata.org/soep-core/datasets/hl/hlf0155_h

    # 1 Schüler- / Jugendlichen- / Studentenwohnheim
    # 2 Wohnheim für Berufstätige
    # 3 Altenheim / Pflegeheim / Seniorenresidenz
    # 4 Sonstiges Heim / Unterkunft
    # 5 Nein

    """

    data["nursing_home"] = np.select(
        [
            data["hlf0155_h"].isin([3, 4]),
            data["hlf0155_h"].isin([1, 2]),
            data["hlf0155_h"] == ANSWER_NO,
        ],
        [1, np.nan, 0],
        default=np.nan,
    )

    return data
