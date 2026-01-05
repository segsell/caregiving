import pickle as pkl

import jax.numpy as jnp
import numpy as np

from caregiving.model.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)
from caregiving.model.shared import RETIREMENT_CHOICES, SEX
from caregiving.model.wealth_and_budget.wages import calc_hourly_wage


def add_experience_and_pp_specs(
    estimation_sample,
    specs,
    exp_factor_for_credited_periods,
    path_to_save_experience_threshold_very_long_insured,
    path_to_save_max_pp_retirement,
    path_to_save_max_exp_diff_period_working,
    path_to_save_pp_for_exp_by_sex_edu,
):
    specs = add_very_long_insured_specs(
        specs,
        exp_factor_for_credited_periods,
        path_to_save_experience_threshold_very_long_insured=path_to_save_experience_threshold_very_long_insured,
    )
    specs = create_max_experience_working(
        estimation_sample,
        specs=specs,
        path_to_save_max_exp_diff_period_working=path_to_save_max_exp_diff_period_working,
    )
    specs = create_pension_points_per_exp(
        estimation_sample,
        specs=specs,
        path_to_save_pp_for_exp_by_sex_edu=path_to_save_pp_for_exp_by_sex_edu,
    )
    specs = create_max_pension_point(specs, path_to_save_max_pp_retirement)

    return specs


def create_max_pension_point(specs, path_to_save_max_pp_retirement):
    # Now calculate maximum experience bonus accross sexes and add them.
    # We can ensure with that, that the rescaled experience is always between  0 and 1.
    # First, age of fresh retirement (so age after retirement is chosen) is min_SRA + 1 and last age of fresh
    # retirement can be max_ret_age + 1
    ret_periods = (
        np.arange(specs["min_SRA"] + 1, specs["max_ret_age"] + 2) - specs["start_age"]
    )
    max_pension_points_retirement = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], len(ret_periods), 2),
        dtype=float,
    )

    for sex_var in range(specs["n_sexes"]):
        for edu_var in range(specs["n_education_types"]):
            for i, period in enumerate(ret_periods):
                for health_id, _health in enumerate([1, 2]):
                    max_exp_period = specs["max_exps_period_working"][period]

                    # The largest bonus can be obtained by being informed and working after the
                    # longest after the SRA.
                    pension_points = calc_pension_points_for_experience(
                        period=period,
                        sex=sex_var,
                        experience_years=max_exp_period,
                        education=edu_var,
                        partner_state=np.array(1),
                        model_specs=specs,
                    )
                    max_pension_points_retirement[sex_var, edu_var, i, health_id] = (
                        pension_points
                    )

    # Get the maximum experience diff across periods
    max_pp_retirement = max_pension_points_retirement[SEX].max()

    np.savetxt(
        path_to_save_max_pp_retirement,
        [max_pp_retirement],
    )

    specs["max_pp_retirement"] = max_pp_retirement

    return specs


def create_max_experience_working(
    estimation_sample, specs, path_to_save_max_exp_diff_period_working
):
    # Initial experience
    # Filter for all non-retirement choices (unemployed, part-time, full-time)
    retirement_choices = np.asarray(RETIREMENT_CHOICES).ravel().tolist()
    df_working = estimation_sample[
        ~estimation_sample["lagged_choice"].isin(retirement_choices)
    ]
    max_exp_diff_period_working = float(
        (df_working["experience"] - df_working["period"]).max()
    )
    np.savetxt(
        path_to_save_max_exp_diff_period_working,
        [max_exp_diff_period_working],
    )

    # Calculate the maximum experience one can have in a working state.
    max_exp_working = (
        specs["max_ret_age"] - specs["start_age"] + max_exp_diff_period_working
    )
    # Now span for each period the maximum experience for working periods.
    max_exps_period_working = np.arange(
        max_exp_diff_period_working, max_exp_working + 2
    )
    # Lowest period very long insured
    min_period_very_long_insured = specs["min_SRA"] - 2 - specs["start_age"]
    # Assign the maximum experience for all the periods one can choose very long insured
    max_exps_period_working[min_period_very_long_insured:] = max_exps_period_working[-1]

    # And now create sex specific interpolation grid points for experience exact at very long insured threshold
    exp_thresholds_very_long_insured = specs["experience_threshold_very_long_insured"]
    exp_thresholds_not_very_long_insured = exp_thresholds_very_long_insured - 0.5
    # Now duplicate the grid point
    all_exp_thresholds_very_long_insured = np.append(
        exp_thresholds_very_long_insured,
        exp_thresholds_not_very_long_insured,
    )

    specs["very_long_insured_grid_points"] = (
        all_exp_thresholds_very_long_insured / max_exps_period_working[-1]
    )
    specs["max_exps_period_working"] = jnp.asarray(max_exps_period_working)

    return specs


def add_very_long_insured_specs(
    specs,
    exp_factor_for_credited_periods,
    path_to_save_experience_threshold_very_long_insured,
):
    """This function adds experience thresholds to be eligible for very long insured retirement path.
    We scale the 45 year of credited periods threshold by a sex specific multiplicator of experience.

    """

    exp_thresholds = np.zeros((2,), dtype=float)
    for sex_var, sex_label in enumerate(["men", "women"]):
        exp_thresholds[sex_var] = (
            45 / exp_factor_for_credited_periods.loc[f"experience_{sex_label}"].iloc[0]
        )

        # We count experience in half years, so round up to next 0.5
        exp_thresholds[sex_var] = np.ceil(exp_thresholds[sex_var] * 2) / 2

    specs["experience_threshold_very_long_insured"] = exp_thresholds
    np.savetxt(
        path_to_save_experience_threshold_very_long_insured,
        exp_thresholds,
    )
    return specs


def create_pension_points_per_exp(
    struct_sample, specs, path_to_save_pp_for_exp_by_sex_edu
):
    # Create grid for experience to pension points mapping
    max_exp = specs["max_exps_period_working"].max() + 2
    exp_grid = np.arange(0, max_exp)

    type_specific_init_exp = (
        struct_sample[
            (struct_sample["period"] == 0) & struct_sample["choice"].isin([2, 3])
        ]
        .groupby(["sex", "education"])["experience"]
        .mean()
        .unstack()
        .values
    )

    hourly_wages_per_exp = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], len(exp_grid)), dtype=float
    )
    age = specs["start_age"]
    for sex_var in range(specs["n_sexes"]):
        for edu_var in range(specs["n_education_types"]):
            for id_exp, exp in enumerate(exp_grid):
                hourly_wages_per_exp[sex_var, edu_var, id_exp] = calc_hourly_wage(
                    sex=sex_var,
                    education=edu_var,
                    experience_years=exp,
                    income_shock=0,
                    model_specs=specs,
                )
                if exp > type_specific_init_exp[sex_var, edu_var]:
                    age += 1
                    age = min(age, specs["max_est_age_labor"])

    pp_per_exp = np.zeros_like(hourly_wages_per_exp)
    for sex_var in range(specs["n_sexes"]):
        for edu_var in range(specs["n_education_types"]):
            for exp in range(len(exp_grid)):
                # Sum hourly wages including current experience and divide
                # by mean hourly wage to get pension points
                mean_hourly_wage = specs["mean_hourly_ft_wage"][sex_var, edu_var]
                pp_per_exp[sex_var, edu_var, exp] = (
                    hourly_wages_per_exp[sex_var, edu_var, : exp + 1].sum()
                    / mean_hourly_wage
                )

    pkl.dump(pp_per_exp, open(path_to_save_pp_for_exp_by_sex_edu, "wb"))
    specs["pp_for_exp_by_sex_edu"] = jnp.asarray(pp_per_exp)

    return specs
