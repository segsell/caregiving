"""Empirical Variance-Covariance Matrix."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product

from caregiving.config import BLD
from caregiving.model.shared import BAD_HEALTH
from caregiving.moments._task_create_empirical_moments import deflate_income_and_wealth


# @pytask.mark.skip(reason="Compare mean with other module")
def task_create_empirical_var(
    path_to_hh_weight: Path = BLD / "data" / "estimation_data_hh_weight.csv",
    # path_to_parent_child_hh_weight: Path = BLD
    # / "data"
    # / "parent_child_data_hh_weight.csv",
    path_to_cpi: Path = BLD / "moments" / "cpi_germany.csv",
    path_to_save: Annotated[Path, Product] = BLD
    / "moments"
    / "empirical_moments_var.csv",
) -> None:
    """Create empirical covariance matrix for the model."""
    dat_hh_weight = pd.read_csv(path_to_hh_weight)
    cpi_data = pd.read_csv(path_to_cpi)

    dat = dat_hh_weight.copy()
    dat = deflate_income_and_wealth(dat, cpi_data)

    # parent_hh_weight = pd.read_csv(path_to_parent_child_hh_weight)
    # weight = "hh_weight"
    # intensive_care_var = "intensive_care_no_other"

    # ================================================================================
    # Parent child data (mother)
    # ================================================================================

    # parent = parent_hh_weight.copy()

    # parent["informal_care_child_weighted"] = (
    #     parent["informal_care_child"] * parent[weight]
    # )
    # parent["home_care_weighted"] = parent["home_care"] * parent[weight]
    # parent["formal_care_weighted"] = parent["formal_care"] * parent[weight]
    # parent["combination_care_weighted"] = parent["combination_care"] * parent[weight]
    # parent["no_combination_care_weighted"] = (
    #     parent["no_combination_care"] * parent[weight]
    # )

    # parent["only_informal_weighted"] = parent["only_informal"] * parent[weight]
    # parent["only_home_care_weighted"] = parent["only_home_care"] * parent[weight]

    # dat["no_intensive_informal_weighted"] = dat["no_intensive_informal"] * dat[weight]
    # dat["intensive_care_no_other_weighted"] = (
    #     dat["intensive_care_no_other"] * dat[weight]
    # )

    # parent["no_home_care_weighted"] = parent["no_home_care"] * parent[weight]
    # parent["no_informal_care_child_weighted"] = (
    #     parent["no_informal_care_child"] * parent[weight]
    # )
    # parent["only_formal_weighted"] = parent["only_formal"] * parent[weight]
    # parent["no_only_formal_weighted"] = parent["no_only_formal"] * parent[weight]
    # parent["no_only_informal_weighted"] = parent["no_only_informal"] * parent[weight]

    # mother = parent[(parent["gender"] == FEMALE)]

    # parent["only_informal_care_child_weighted"] = (
    #     parent["informal_care_child"] * parent[weight]
    # )
    # parent["no_only_informal_care_child_weighted"] = (
    #     parent["no_informal_care_child"] * parent[weight]
    # )
    # parent["no_home_care_weighted"] = parent["no_home_care"] * parent[weight]
    # parent["no_formal_care_weighted"] = parent["no_formal_care"] * parent[weight]
    # parent["no_informal_care_child_weighted"] = (
    #     parent["no_informal_care_child"] * parent[weight]
    # )

    # ================================================================================
    # SOEP
    # ================================================================================

    employment_by_age_soep = get_var_employment_by_age_soep()

    # ols_coeffs_savings_rate = get_var_coefficients_savings_rate_regression_soep()

    # employment_by_age_bin_non_caregivers_soep = (
    #     get_var_employment_by_age_bin_non_informal_caregivers_soep()
    # )
    # employment_by_age_bin_caregivers_soep = (
    #     get_var_employment_by_age_bin_informal_parental_caregivers_soep()
    # )

    # share_informal_care_by_age_bin = (
    #     get_var_share_informal_maternal_care_by_age_bin_soep()
    # )

    employment_transitions_soep = get_var_employment_transitions_soep()

    # ================================================================================
    # SHARE
    # ================================================================================

    # caregiving_by_mother_health_and_presence_of_sibling = (
    #     get_var_caregiving_status_by_mother_health_and_presence_of_sibling(
    #         mother,
    #         sibling_var="has_two_daughters",
    #         weight=weight,
    #     )
    # )

    # care_transitions_estimation_data = get_var_care_transitions_from_estimation_data(
    #     dat,
    #     intensive_care_var=intensive_care_var,
    # )
    # care_transitions_parent_child_data = get_care_transitions_from_parent_child_data(
    #     parent,
    # )

    # ================================================================================
    # Combine variances
    # ================================================================================

    all_vars = pd.concat(
        [
            employment_by_age_soep,
            # ols_coeffs_savings_rate,
            # employment_by_age_bin_non_caregivers_soep,
            # employment_by_age_bin_caregivers_soep,
            #
            # share_informal_care_by_age_bin,
            # caregiving_by_mother_health_and_presence_of_sibling,
            #
            employment_transitions_soep,
            # care_transitions_estimation_data,
            # care_transitions_parent_child_data,
        ],
        ignore_index=False,
        axis=0,
    )

    all_vars.to_csv(path_to_save)


def get_var_care_transitions_from_estimation_data(
    dat,
    intensive_care_var,
):
    """Calculate transitions and their unweighted variances."""
    # Collecting transitions and associated data for variance calculation
    transitions = {
        "no_care_to_no_informal_care": get_care_transition_unweighted(
            dat,
            previous_choice="no_intensive_informal",
            current_choice="no_intensive_informal",
        ),
        "no_care_to_informal_care": get_care_transition_unweighted(
            dat,
            previous_choice="no_intensive_informal",
            current_choice=intensive_care_var,
        ),
        "informal_care_to_no_informal_care": get_care_transition_unweighted(
            dat,
            previous_choice=intensive_care_var,
            current_choice="no_intensive_informal",
        ),
        "informal_care_to_informal_care": get_care_transition_unweighted(
            dat,
            previous_choice=intensive_care_var,
            current_choice=intensive_care_var,
        ),
    }

    var_no_care_to_no_informal_care = calculate_unweighted_variance(
        transitions["no_care_to_no_informal_care"],
    )
    var_no_care_to_informal_care = calculate_unweighted_variance(
        transitions["no_care_to_informal_care"],
    )
    var_informal_care_to_no_informal_care = calculate_unweighted_variance(
        transitions["informal_care_to_no_informal_care"],
    )
    var_informal_care_to_informal_care = calculate_unweighted_variance(
        transitions["informal_care_to_informal_care"],
    )

    return pd.Series(
        {
            "no_care_to_no_informal_care": var_no_care_to_no_informal_care,
            "no_care_to_informal_care": var_no_care_to_informal_care,
            "informal_care_to_no_informal_care": var_informal_care_to_no_informal_care,
            "informal_care_to_informal_care": var_informal_care_to_informal_care,
        },
    )


def get_care_transitions_from_parent_child_data(parent):
    """Get unweighted care transitions from parent-child dataset."""
    # Define transitions and their transitions
    transitions = {
        "no_informal_to_no_formal": get_care_transition_unweighted(
            parent,
            previous_choice="no_informal_care_child",
            current_choice="no_formal_care",
        ),
        "no_informal_to_formal": get_care_transition_unweighted(
            parent,
            previous_choice="no_informal_care_child",
            current_choice="formal_care",
        ),
        "informal_to_no_formal": get_care_transition_unweighted(
            parent,
            previous_choice="informal_care_child",
            current_choice="no_formal_care",
        ),
        "informal_to_formal": get_care_transition_unweighted(
            parent,
            previous_choice="informal_care_child",
            current_choice="formal_care",
        ),
        "no_formal_to_no_informal": get_care_transition_unweighted(
            parent,
            previous_choice="no_formal_care",
            current_choice="no_informal_care_child",
        ),
        "no_formal_to_informal": get_care_transition_unweighted(
            parent,
            previous_choice="no_formal_care",
            current_choice="informal_care_child",
        ),
        "formal_to_no_informal": get_care_transition_unweighted(
            parent,
            previous_choice="formal_care",
            current_choice="no_informal_care_child",
        ),
        "formal_to_informal": get_care_transition_unweighted(
            parent,
            previous_choice="formal_care",
            current_choice="informal_care_child",
        ),
        "no_formal_to_no_formal": get_care_transition_unweighted(
            parent,
            previous_choice="no_formal_care",
            current_choice="no_formal_care",
        ),
        "no_formal_to_formal": get_care_transition_unweighted(
            parent,
            previous_choice="no_formal_care",
            current_choice="formal_care",
        ),
        "formal_to_no_formal": get_care_transition_unweighted(
            parent,
            previous_choice="formal_care",
            current_choice="no_formal_care",
        ),
        "formal_to_formal": get_care_transition_unweighted(
            parent,
            previous_choice="formal_care",
            current_choice="formal_care",
        ),
    }

    var_no_informal_to_no_formal = calculate_unweighted_variance(
        transitions["no_informal_to_no_formal"],
    )
    var_no_informal_to_formal = calculate_unweighted_variance(
        transitions["no_informal_to_formal"],
    )
    var_informal_to_no_formal = calculate_unweighted_variance(
        transitions["informal_to_no_formal"],
    )
    var_informal_to_formal = calculate_unweighted_variance(
        transitions["informal_to_formal"],
    )
    var_no_formal_to_no_informal = calculate_unweighted_variance(
        transitions["no_formal_to_no_informal"],
    )
    var_no_formal_to_informal = calculate_unweighted_variance(
        transitions["no_formal_to_informal"],
    )
    var_formal_to_no_informal = calculate_unweighted_variance(
        transitions["formal_to_no_informal"],
    )
    var_formal_to_informal = calculate_unweighted_variance(
        transitions["formal_to_informal"],
    )

    var_no_formal_to_no_formal = calculate_unweighted_variance(
        transitions["no_formal_to_no_formal"],
    )
    var_no_formal_to_formal = calculate_unweighted_variance(
        transitions["no_formal_to_formal"],
    )
    var_formal_to_no_formal = calculate_unweighted_variance(
        transitions["formal_to_no_formal"],
    )
    var_formal_to_formal = calculate_unweighted_variance(
        transitions["formal_to_formal"],
    )

    return pd.Series(
        {
            "no_informal_to_no_formal": var_no_informal_to_no_formal,
            "no_informal_to_formal": var_no_informal_to_formal,
            "informal_to_no_formal": var_informal_to_no_formal,
            "informal_to_formal": var_informal_to_formal,
            "no_formal_to_no_informal": var_no_formal_to_no_informal,
            "no_formal_to_informal": var_no_formal_to_informal,
            "formal_to_no_informal": var_formal_to_no_informal,
            "formal_to_informal": var_formal_to_informal,
            "no_formal_to_no_formal": var_no_formal_to_no_formal,
            "no_formal_to_formal": var_no_formal_to_formal,
            "formal_to_no_formal": var_formal_to_no_formal,
            "formal_to_formal": var_formal_to_formal,
        },
    )


def get_var_caregiving_status_by_mother_health_and_presence_of_sibling(
    mother,
    # sibling_var,
    weight,
):
    no_care_mother_bad_health = get_weighted_variance_one_condition(
        dat=mother,
        moment_unweighted="no_care",
        condition_var="health",
        condition_val=BAD_HEALTH,
        weight=weight,
    )
    informal_care_mother_bad_health = get_weighted_variance_one_condition(
        dat=mother,
        moment_unweighted="informal_care_child_no_comb",
        condition_var="health",
        condition_val=BAD_HEALTH,
        weight=weight,
    )
    formal_care_mother_bad_health = get_weighted_variance_one_condition(
        dat=mother,
        moment_unweighted="formal_care_no_comb",
        condition_var="health",
        condition_val=BAD_HEALTH,
        weight=weight,
    )
    comb_care_mother_bad_health = get_weighted_variance_one_condition(
        dat=mother,
        moment_unweighted="combination_care",
        condition_var="health",
        condition_val=BAD_HEALTH,
        weight=weight,
    )

    # no_care_mother_bad_health_sibling = get_weighted_variance_two_conditions(
    #     dat=mother,
    #     moment_unweighted="no_care",
    #     condition_var_one="health",
    #     condition_val_one=BAD_HEALTH,
    #     condition_var_two=sibling_var,
    #     condition_val_two=True,
    #     weight=weight,
    # )
    # informal_mother_bad_health_sibling = get_weighted_variance_two_conditions(
    #     dat=mother,
    #     moment_unweighted="informal_care_child_no_comb",
    #     condition_var_one="health",
    #     condition_val_one=BAD_HEALTH,
    #     condition_var_two=sibling_var,
    #     condition_val_two=True,
    #     weight=weight,
    # )
    # formal_mother_bad_health_sibling = get_weighted_variance_two_conditions(
    #     dat=mother,
    #     moment_unweighted="formal_care_no_comb",
    #     condition_var_one="health",
    #     condition_val_one=BAD_HEALTH,
    #     condition_var_two=sibling_var,
    #     condition_val_two=True,
    #     weight=weight,
    # )
    # comb_mother_bad_health_sibling = get_weighted_variance_two_conditions(
    #     dat=mother,
    #     moment_unweighted="combination_care",
    #     condition_var_one="health",
    #     condition_val_one=BAD_HEALTH,
    #     condition_var_two=sibling_var,
    #     condition_val_two=True,
    #     weight=weight,
    # )

    return pd.Series(
        {
            "no_care_mother_health_2": no_care_mother_bad_health,
            "informal_care_mother_health_2": informal_care_mother_bad_health,
            "formal_care_mother_health_2": formal_care_mother_bad_health,
            "combination_care_mother_health_2": comb_care_mother_bad_health,
        },
    )


# ================================================================================
# Conditional variance
# ================================================================================


def get_weighted_variance_one_condition(
    dat,
    moment_unweighted,
    condition_var,
    condition_val,
    weight,
):
    """Calculate the weighted variance of a moment.

    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

    """
    subset = dat.loc[
        (dat[condition_var] == condition_val),
        [moment_unweighted, weight],
    ]

    # Calculate the weighted mean
    total_weight = subset[weight].sum()
    weighted_mean = (subset[moment_unweighted] * subset[weight]).sum() / total_weight

    # Calculate the unbiased weighted variance
    return ((subset[moment_unweighted] - weighted_mean) ** 2 * subset[weight]).sum() / (
        total_weight - 1
    )


def get_weighted_variance_two_conditions(
    dat,
    moment_unweighted,
    condition_var_one,
    condition_val_one,
    condition_var_two,
    condition_val_two,
    weight,
):
    """Calculate the weighted variance of a moment.

    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

    """
    subset = dat.loc[
        (dat[condition_var_one] == condition_val_one)
        & (dat[condition_var_two] == condition_val_two),
        [moment_unweighted, weight],
    ]

    # Calculate the weighted mean
    total_weight = subset[weight].sum()
    weighted_mean = (subset[moment_unweighted] * subset[weight]).sum() / total_weight

    # Calculate the unbiased weighted variance
    return ((subset[moment_unweighted] - weighted_mean) ** 2 * subset[weight]).sum() / (
        total_weight - 1
    )


# ================================================================================
# Care transitions
# ================================================================================


def get_care_transition_unweighted(dat, previous_choice, current_choice):
    """Calculate unweighted transition probability."""
    transition_cases = dat.loc[
        (dat[f"lagged_{previous_choice}"] == True) & (dat[current_choice] == True)
    ].shape[0]

    total_cases = dat.loc[
        (dat[f"lagged_{previous_choice}"] == True) & (dat[current_choice].notna())
    ].shape[0]

    transition_probability = transition_cases / total_cases

    return {
        "total_cases": total_cases,
        "transition_cases": transition_cases,
        "transition_probability": transition_probability,
    }


def get_care_transition_weighted(dat, previous_choice, current_choice, weight):
    """Get weighted caregiving transition probabilities."""
    lagged_choice_true = dat.loc[dat[f"lagged_{previous_choice}"] == True]

    sum_weights_lagged = lagged_choice_true.loc[
        lagged_choice_true[current_choice].notna(),
        weight,
    ].sum()
    weighted_sum_current = (
        lagged_choice_true[current_choice] * lagged_choice_true[weight]
    ).sum()

    transition_prob = weighted_sum_current / sum_weights_lagged

    return {
        "sum_weights": sum_weights_lagged,
        "weighted_prob_sum": weighted_sum_current,
        "transition_probability": transition_prob,
    }


def calculate_unweighted_variance(transition_info):
    """Calculate the variance of an unweighted transition probability."""
    p = transition_info["transition_probability"]
    n = transition_info["total_cases"]

    if n == 0:
        return None  # Avoid division by zero

    return (p * (1 - p)) / n


def calculate_weighted_variance(dat, transition_info, current_choice, weight):
    """Calculate variance of a weighted transition probability.

    # weighted_variance = variance_numerator / sum_weights if sum_weights != 0 else None

    """
    sum_weights = transition_info["sum_weights"]
    weighted_prob_sum = transition_info["weighted_prob_sum"]
    weighted_mean_prob = weighted_prob_sum / sum_weights

    # Calculate variance
    return (
        (dat[current_choice] - weighted_mean_prob) ** 2 * dat[weight]
    ).sum() / sum_weights


# ================================================================================
# SOEP
# ================================================================================


def get_var_employment_by_age_soep():
    return pd.Series(
        {
            # not working
            "not_working_age_40": 0.23113014,
            "not_working_age_41": 0.22536342,
            "not_working_age_42": 0.21826035,
            "not_working_age_43": 0.21502003,
            "not_working_age_44": 0.21009474,
            "not_working_age_45": 0.20878081,
            "not_working_age_46": 0.20681589,
            "not_working_age_47": 0.20720435,
            "not_working_age_48": 0.20282751,
            "not_working_age_49": 0.20040272,
            "not_working_age_50": 0.20118694,
            "not_working_age_51": 0.20555282,
            "not_working_age_52": 0.20931419,
            "not_working_age_53": 0.21369371,
            "not_working_age_54": 0.21956498,
            "not_working_age_55": 0.22156134,
            "not_working_age_56": 0.22739821,
            "not_working_age_57": 0.23206562,
            "not_working_age_58": 0.24170685,
            "not_working_age_59": 0.24740751,
            "not_working_age_60": 0.25006227,
            "not_working_age_61": 0.23980666,
            "not_working_age_62": 0.21280786,
            "not_working_age_63": 0.18163850,
            "not_working_age_64": 0.12445516,
            "not_working_age_65": 0.09296116,
            "not_working_age_66": 0.05217511,
            "not_working_age_67": 0.03961297,
            "not_working_age_68": 0.02881498,
            "not_working_age_69": 0.02459085,
            # "not_working_age_70": 0.01906649,
            # part-time
            "part_time_age_40": 0.23401160,
            "part_time_age_41": 0.23425810,
            "part_time_age_42": 0.23667154,
            "part_time_age_43": 0.23631351,
            "part_time_age_44": 0.23755958,
            "part_time_age_45": 0.23565479,
            "part_time_age_46": 0.23415695,
            "part_time_age_47": 0.23113179,
            "part_time_age_48": 0.22897852,
            "part_time_age_49": 0.23048351,
            "part_time_age_50": 0.22678326,
            "part_time_age_51": 0.22224495,
            "part_time_age_52": 0.21939385,
            "part_time_age_53": 0.21536914,
            "part_time_age_54": 0.21359061,
            "part_time_age_55": 0.21572239,
            "part_time_age_56": 0.21206485,
            "part_time_age_57": 0.20911575,
            "part_time_age_58": 0.20091090,
            "part_time_age_59": 0.19037875,
            "part_time_age_60": 0.17945396,
            "part_time_age_61": 0.14911182,
            "part_time_age_62": 0.12070135,
            "part_time_age_63": 0.10360206,
            "part_time_age_64": 0.06778166,
            "part_time_age_65": 0.05432950,
            "part_time_age_66": 0.03509204,
            "part_time_age_67": 0.02814561,
            "part_time_age_68": 0.02023185,
            "part_time_age_69": 0.01750941,
            # "part_time_age_70": 0.01369605,
            # full-time
            "full_time_age_40": 0.194402472,
            "full_time_age_41": 0.202829440,
            "full_time_age_42": 0.207558680,
            "full_time_age_43": 0.211739796,
            "full_time_age_44": 0.214537702,
            "full_time_age_45": 0.218732332,
            "full_time_age_46": 0.222462318,
            "full_time_age_47": 0.225854759,
            "full_time_age_48": 0.231114669,
            "full_time_age_49": 0.231180281,
            "full_time_age_50": 0.234055072,
            "full_time_age_51": 0.235080914,
            "full_time_age_52": 0.234896240,
            "full_time_age_53": 0.234895853,
            "full_time_age_54": 0.231976330,
            "full_time_age_55": 0.228765000,
            "full_time_age_56": 0.226167061,
            "full_time_age_57": 0.223319929,
            "full_time_age_58": 0.215128948,
            "full_time_age_59": 0.208390214,
            "full_time_age_60": 0.193918073,
            "full_time_age_61": 0.169618258,
            "full_time_age_62": 0.138896814,
            "full_time_age_63": 0.106464246,
            "full_time_age_64": 0.067279820,
            "full_time_age_65": 0.043941292,
            "full_time_age_66": 0.018452411,
            "full_time_age_67": 0.012182171,
            "full_time_age_68": 0.008956428,
            "full_time_age_69": 0.007345192,
            # "full_time_age_70": 0.005524702,
        },
    )


def get_var_coefficients_savings_rate_regression_soep():
    return pd.Series(
        {
            "savings_rate_constant": 0.000548694929,
            "savings_rate_age": 0.005125798833,
            "savings_rate_age_squared": 0.001756701331,
            "savings_rate_high_education": 0.000000488544,
            "savings_rate_part_time": 0.001620693283,
            "savings_rate_full_time": 0.002157233215,
            # "savings_rate_informal_care": 0.000301182374,
        },
    )


# var
def get_var_employment_by_age_bin_informal_parental_caregivers_soep():
    return pd.Series(
        {
            "not_working_age_40_45": 0.24125176,
            "not_working_age_45_50": 0.21812266,
            "not_working_age_50_55": 0.22134030,
            "not_working_age_55_60": 0.24387144,
            "not_working_age_60_65": 0.19318271,
            "not_working_age_65_70": 0.04345898,
            "part_time_age_40_45": 0.23340033,
            "part_time_age_45_50": 0.23210090,
            "part_time_age_50_55": 0.23742719,
            "part_time_age_55_60": 0.21778835,
            "part_time_age_60_65": 0.11591068,
            "part_time_age_65_70": 0.02687942,
            "full_time_age_40_45": 0.17430175,
            "full_time_age_45_50": 0.21535956,
            "full_time_age_50_55": 0.20284277,
            "full_time_age_55_60": 0.19216314,
            "full_time_age_60_65": 0.11141806,
            "full_time_age_65_70": 0.01756678,
        },
    )


def get_var_employment_by_age_bin_non_informal_caregivers_soep():
    return pd.Series(
        {
            "not_working_age_40_45": 0.21760557,
            "not_working_age_45_50": 0.20314369,
            "not_working_age_50_55": 0.20803696,
            "not_working_age_55_60": 0.23444000,
            "not_working_age_60_65": 0.21893375,
            "not_working_age_65_70": 0.04955567,
            "part_time_age_40_45": 0.23638008,
            "part_time_age_45_50": 0.23230882,
            "part_time_age_50_55": 0.21822528,
            "part_time_age_55_60": 0.20502764,
            "part_time_age_60_65": 0.12856359,
            "part_time_age_65_70": 0.03196961,
            "full_time_age_40_45": 0.208697726,
            "full_time_age_45_50": 0.227366151,
            "full_time_age_50_55": 0.236358320,
            "full_time_age_55_60": 0.223407124,
            "full_time_age_60_65": 0.142552176,
            "full_time_age_65_70": 0.018857367,
        },
    )


def get_var_share_informal_maternal_care_by_age_bin_soep():
    return pd.Series(
        {
            "share_informal_care_40_45": 0.03505911,
            "share_informal_care_45_50": 0.04650906,
            "share_informal_care_50_55": 0.06010956,
            "share_informal_care_55_60": 0.06852833,
            "share_informal_care_60_65": 0.05891915,
            "share_informal_care_65_70": 0.03491249,
        },
    )


def get_var_employment_transitions_soep():
    """Get variance of employment transitions."""
    return pd.Series(
        {
            "not_working_to_not_working": 0.0000014030153746,
            "not_working_to_part_time": 0.0000011223523576,
            "not_working_to_full_time": 0.0000003261830112,
            "part_time_to_not_working": 0.0000026726781986,
            "part_time_to_part_time": 0.0000041901074618,
            "part_time_to_full_time": 0.0000019742078027,
            "full_time_to_not_working": 0.0000015082989226,
            "full_time_to_part_time": 0.0000019227102296,
            "full_time_to_full_time": 0.0000032040667450,
        },
    )
