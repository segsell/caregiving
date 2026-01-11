"""Function that simulates the model for a given scenario."""

import jax
import numpy as np

from caregiving.model.experience_baseline_model import construct_experience_years
from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    PART_TIME,
    RETIREMENT,
    SEX,
    UNEMPLOYED,
)

jax.config.update("jax_enable_x64", True)


def simulate_scenario(model_solved, initial_states, model_specs):
    sim_df = model_solved.simulate(
        states_initial=initial_states,
        seed=model_specs["seed"],
    )

    sim_df.reset_index(inplace=True)

    # Filter to only rows where health != DEAD and consumption is not NaN
    # Do this early to avoid creating additional variables for rows we'll drop
    sim_df = sim_df[(sim_df["health"] != DEAD) & (sim_df["consumption"].notna())].copy()

    sim_df = create_additional_variables(sim_df, model_specs)

    return sim_df


def simulate_scenario_slim(model_solved, initial_states, model_specs):
    sim_df = model_solved.simulate(
        states_initial=initial_states,
        seed=model_specs["seed"],
    )

    sim_df.reset_index(inplace=True)

    # Filter to only rows where health != DEAD and consumption is not NaN
    # Do this early to avoid creating additional variables for rows we'll drop
    sim_df = sim_df[(sim_df["health"] != DEAD) & (sim_df["consumption"].notna())].copy()

    # Create hard-coded sex variable
    sim_df.loc[:, "sex"] = SEX

    # Create additional variables
    sim_df.loc[:, "age"] = sim_df["period"] + model_specs["start_age"]

    return sim_df


# ==============================================================================
# Additional variables related to the budget equation (see budget_equation.py).
# ==============================================================================


def create_additional_variables(df, specs):
    """Wrapper function to create additional variables in the simulated dataframe."""
    df = df.copy()

    df = _create_income_variables(df, specs)
    df = _transform_states_into_variables(df, specs)
    df = _compute_working_hours(df, specs)
    df = _compute_actual_retirement_age(df)
    df = create_real_utility(df, specs)

    return df


def _create_income_variables(df, specs):
    """Create income related variables in the simulated dataframe.

    Note: check budget equation first! They may already be there (under "aux").
    """
    df = df.copy()

    # Create income vars:
    # First, total income as the difference between wealth at the beginning of
    # next period and savings.
    df.loc[:, "total_income"] = df["assets_begin_of_period"] - df.groupby("agent")[
        "savings"
    ].shift(1)

    # periodic savings and savings rate
    df.loc[:, "savings_dec"] = df["total_income"] - df["consumption"]
    df.loc[:, "savings_rate"] = df["savings_dec"] / df["total_income"]

    # Create gross own income (without pension income)
    retirement_values = RETIREMENT.ravel().tolist()
    unemployed_values = UNEMPLOYED.ravel().tolist()
    part_time_values = PART_TIME.ravel().tolist()
    full_time_values = FULL_TIME.ravel().tolist()
    work_values = part_time_values + full_time_values

    df.loc[:, "gross_own_income"] = (
        df["choice"].isin(retirement_values) * df["gross_retirement_income"]
        + df["choice"].isin(unemployed_values) * 0
        + df["choice"].isin(work_values) * df["gross_labor_income"]
    )

    return df


def _transform_states_into_variables(df, specs):
    """Transform state variables into more interpretable variables."""
    df = df.copy()

    # Create hard-coded sex variable
    df.loc[:, "sex"] = SEX
    retirement_values = RETIREMENT.ravel().tolist()
    df.loc[:, "is_retired"] = df["choice"].isin(retirement_values)

    # Create additional variables
    df.loc[:, "age"] = df["period"] + specs["start_age"]

    # Create experience years
    df.loc[:, "exp_years"] = construct_experience_years(
        float_experience=df["experience"].values,
        period=df["period"].values,
        is_retired=df["is_retired"].values,
        model_specs=specs,
    )

    return df


def _compute_working_hours(df, specs):
    """Compute working hours based on employment choice and demographics."""
    df = df.copy()
    sex_var = SEX

    # Initialize working_hours column
    df.loc[:, "working_hours"] = 0.0

    part_time_values = PART_TIME.ravel().tolist()
    full_time_values = FULL_TIME.ravel().tolist()

    for edu_var in range(specs["n_education_types"]):
        # Full-time work
        mask_ft = (
            df["choice"].isin(full_time_values)
            # & (df["sex"] == sex_var)
            & (df["education"] == edu_var)
        )
        df.loc[mask_ft, "working_hours"] = specs["av_annual_hours_ft"][sex_var, edu_var]

        # Part-time work
        mask_pt = (
            df["choice"].isin(part_time_values)
            # & (df["sex"] == sex_var)
            & (df["education"] == edu_var)
        )
        df.loc[mask_pt, "working_hours"] = specs["av_annual_hours_pt"][sex_var, edu_var]

    return df


def _compute_actual_retirement_age(df):
    """Compute actual retirement age based on choice variable."""
    df = df.copy()

    retirement_values = RETIREMENT.ravel().tolist()
    df_retirement = df[df["choice"].isin(retirement_values)].copy()
    actual_retirement_ages = df_retirement.groupby("agent")["age"].min()
    df.loc[:, "actual_retirement_age"] = df["agent"].map(actual_retirement_ages)

    return df


def create_realized_taste_shock(df, specs):
    """Create realized taste shock variable based on actual choice."""
    df.loc[:, "real_taste_shock"] = np.nan
    for choice in range(specs["n_choices"]):
        df.loc[df["choice"] == choice, "real_taste_shock"] = df.loc[
            df["choice"] == choice, f"taste_shocks_{choice}"
        ]
    return df


def create_real_utility(df, specs):
    """Create realized utility (utility + realized taste shock)."""
    df = create_realized_taste_shock(df, specs)
    df.loc[:, "real_util"] = df["utility"] + df["real_taste_shock"]

    return df
