import numpy as np


def _print_filter(before, after, msg):
    pct = (after - before) / before * 100 if before > 0 else 0
    print(f"{after} {msg} ({pct:+.2f}%)")


def create_experience_variable_with_cap(data, exp_cap, filter_missings=True):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience.
    It also enforces an experience cap.
    """
    # Create experience variable
    data = create_experience_and_working_years(data, filter_missings=filter_missings)
    # Enforce experience cap
    data.loc[data["experience"] > exp_cap, "experience"] = exp_cap

    return data


def create_experience_and_working_years(data, filter_missings=True):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience. Additional it creates the working years  # noqa: E501
    as sum of both.
    In both variables we ensure only valid experience by scaling part time experience.
    """
    data = scale_part_time_exp(data)
    data = sum_experience_variables(data, filter_missings=filter_missings)
    data = raw_working_years(data, filter_missings=filter_missings)
    return data


def create_experience(data, filter_missings=True):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience.
    We ensure only valid experience by scaling part time experience.
    """
    data = scale_part_time_exp(data)
    data = sum_experience_variables(data, filter_missings=filter_missings)
    return data


def sum_experience_variables(data, filter_missings=True):
    """This function sums the experience variables pgexpft and pgexppt.
    Part time experience is weighted by 0.5. The function returns a new column
    experience.
    """
    invalid_ft_exp = data["pgexpft"] < 0
    invalid_pt_exp = data["pgexppt"] < 0

    # Initialize empty experience column
    data.loc[:, "experience"] = np.nan

    # If both are valid use the sum
    mask_both_valid = ~invalid_ft_exp & ~invalid_pt_exp
    data.loc[mask_both_valid, "experience"] = (
        data.loc[mask_both_valid, "pgexpft"]
        + 0.5 * data.loc[mask_both_valid, "pgexppt"]
    )
    # If only one is valid use the valid one
    mask_pt_valid = invalid_ft_exp & ~invalid_pt_exp
    data.loc[mask_pt_valid, "experience"] = 0.5 * data.loc[mask_pt_valid, "pgexppt"]
    mask_ft_valid = ~invalid_ft_exp & invalid_pt_exp
    data.loc[mask_ft_valid, "experience"] = data.loc[mask_ft_valid, "pgexpft"]

    if filter_missings:
        # If both are invalid drop observations
        before = len(data)
        data = data[data["experience"].notna()]
        _print_filter(
            before,
            len(data),
            "left after dropping people with invalid experience values",
        )

    return data


def raw_working_years(data, filter_missings=True):
    """This function sums the experience variables pgexpft and pgexppt.
    Part time experience is NOT weighted (counted as full years).
    The function returns a new column 'working_years'.
    """
    data = data.copy()
    invalid_ft_exp = data["pgexpft"] < 0
    invalid_pt_exp = data["pgexppt"] < 0

    # Initialize empty experience column
    data.loc[:, "working_years"] = np.nan

    # If both are valid use the sum
    mask_both_valid = ~invalid_ft_exp & ~invalid_pt_exp
    data.loc[mask_both_valid, "working_years"] = (
        data.loc[mask_both_valid, "pgexpft"] + data.loc[mask_both_valid, "pgexppt"]
    )
    # If only one is valid use the valid one
    mask_pt_valid = invalid_ft_exp & ~invalid_pt_exp
    data.loc[mask_pt_valid, "working_years"] = data.loc[mask_pt_valid, "pgexppt"]
    mask_ft_valid = ~invalid_ft_exp & invalid_pt_exp
    data.loc[mask_ft_valid, "working_years"] = data.loc[mask_ft_valid, "pgexpft"]

    if filter_missings:
        # If both are invalid drop observations
        before = len(data)
        data = data[data["working_years"].notna()]
        _print_filter(
            before, len(data), "left after dropping people with invalid working years"
        )

    return data


def scale_part_time_exp(data):
    """Check if years of part plus full time exceed age minus 14 (not  # noqa: E501
    allowed to work before)"""
    max_years = data["age"] - 14
    exp_exceeding = ((data["pgexpft"] + data["pgexppt"]) - max_years).clip(lower=0)
    # # Deduct exceeding experience from part time experience. Assume if worked b
    # oth, you worked full
    data.loc[:, "pgexppt"] -= exp_exceeding
    return data
