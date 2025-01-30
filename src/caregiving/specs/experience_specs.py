import jax.numpy as jnp
import numpy as np
import pandas as pd

from caregiving.model.state_space import calc_experience_years_for_pension_adjustment


def create_max_experience(path_dict, specs):
    # max initial experience
    data_decision = pd.read_pickle(
        path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )
    max_exp_diff_data = (data_decision["experience"] - data_decision["period"]).max()

    max_exp_diffs_per_period = np.zeros(specs["n_periods"], dtype=float)

    # Assign up to min_SRA period the maximum experience difference from the data
    min_SRA_period = specs["min_SRA"] - specs["start_age"]
    max_exp_diffs_per_period[: min_SRA_period + 1] = max_exp_diff_data

    # Now calculate maximum experience bonus across sexes and add them.
    # We can ensure with that, that the rescaled experience is always between  0 and 1.
    # First, age of fresh retirement is min_SRA + 1 and last age of fresh retirement
    # can be max_ret_age + 1
    ret_periods = (
        np.arange(specs["min_SRA"] + 1, specs["max_ret_age"] + 2) - specs["start_age"]
    )
    max_exp_diff_to_periods = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], len(ret_periods)),
        dtype=float,
    )

    for sex_var in range(specs["n_sexes"]):
        for edu_var in range(specs["n_education_types"]):
            for i, period in enumerate(ret_periods):
                max_exp_period = period + max_exp_diff_data
                new_exp = calc_experience_years_for_pension_adjustment(
                    period=period,
                    sex=sex_var,
                    experience_years=max_exp_period,
                    education=edu_var,
                    policy_state=0,
                    informed=1,
                    options=specs,
                )
                max_exp_diff_to_periods[sex_var, edu_var, i] = new_exp - period

    # Get the maximum experience diff across periods
    max_across_ret_periods = max_exp_diff_to_periods.max()
    # Assign the maximum experience diff to the respective periods
    # max_exp_diffs_per_period[ret_periods] = max_across_ret_periods
    #
    # # Get the maximum over all and assign to the rest of the periods,
    # # always minus 1
    # max_exp_diff = max_exp_diff_to_periods.max()
    # last_periods = np.arange(ret_periods[-1] + 1, specs["n_periods"])
    # exp_diff_reduction = last_periods - ret_periods[-1]
    # max_exp_diffs_per_period[last_periods] = max_exp_diff - exp_diff_reduction
    total_max = np.maximum(max_across_ret_periods, max_exp_diff_data)
    max_exp_diffs_per_period[:] = total_max

    np.savetxt(
        path_dict["first_step_results"] + "max_exp_diffs_per_period.txt",
        max_exp_diffs_per_period,
    )

    return jnp.asarray(max_exp_diffs_per_period)
