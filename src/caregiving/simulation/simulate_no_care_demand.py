"""Simulation helper for the no-care-demand counterfactual.

Mirrors the baseline simulate_scenario but uses the reduced 4-state
choice arrays from shared_no_care_demand to assign working hours, etc.

"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods

from caregiving.model.shared import DEAD, SEX
from caregiving.model.shared_no_care_demand import (
    FULL_TIME_NO_CARE_DEMAND,
    PART_TIME_NO_CARE_DEMAND,
    RETIREMENT_NO_CARE_DEMAND,
    UNEMPLOYED_NO_CARE_DEMAND,
    WORK_NO_CARE_DEMAND,
)
from caregiving.model.state_space import construct_experience_years
from caregiving.model.state_space_no_care_demand import create_state_space_functions
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive_no_care_demand import (
    create_utility_functions,
)
from caregiving.model.wealth_and_budget.budget_equation_no_care_demand import (
    budget_constraint,
)
from caregiving.model.wealth_and_budget.pensions import (
    calc_gross_pension_income,
)
from caregiving.model.wealth_and_budget.transfers import (
    calc_child_benefits,
    calc_unemployment_benefits,
)
from caregiving.model.wealth_and_budget.wages_no_care_demand import (
    calculate_gross_labor_income,
)

# ==============================================================================
# Additional variables related to the budget equation (see budget_equation.py).
# ==============================================================================


def create_additional_variables_no_care_demand(df, specs):
    """Wrapper function to create additional variables in the simulated dataframe."""
    df = df.copy()

    df = _create_income_variables_no_care_demand(df, specs)
    df = _transform_states_into_variables(df, specs)
    df = _compute_working_hours_no_care_demand(df, specs)
    df = _compute_actual_retirement_age_no_care_demand(df)
    df = create_real_utility(df, specs)

    return df


def _create_income_variables_no_care_demand(df, specs):
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
    # df.loc[:, "total_income"] = (
    #     df.groupby("agent")["assets_begin_of_period"].shift(-1) - df["savings"]
    # )

    # periodic savings and savings rate
    df.loc[:, "savings_dec"] = df["total_income"] - df["consumption"]
    df.loc[:, "savings_rate"] = df["savings_dec"] / df["total_income"]

    # Create gross own income (without pension income)
    retirement_values = RETIREMENT_NO_CARE_DEMAND.ravel().tolist()
    unemployed_values = UNEMPLOYED_NO_CARE_DEMAND.ravel().tolist()
    part_time_values = PART_TIME_NO_CARE_DEMAND.ravel().tolist()
    full_time_values = FULL_TIME_NO_CARE_DEMAND.ravel().tolist()
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

    # Create additional variables
    df.loc[:, "age"] = df["period"] + specs["start_age"]

    # Create experience years
    df.loc[:, "exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df["period"].values,
        max_exp_diffs_per_period=specs["max_exp_diffs_per_period"],
    )

    return df


def _compute_working_hours_no_care_demand(df, specs):
    """Compute working hours based on employment choice and demographics."""
    df = df.copy()
    sex_var = SEX

    # Initialize working_hours column
    df.loc[:, "working_hours"] = 0.0

    part_time_values = PART_TIME_NO_CARE_DEMAND.ravel().tolist()
    full_time_values = FULL_TIME_NO_CARE_DEMAND.ravel().tolist()

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


def _compute_actual_retirement_age_no_care_demand(df):
    """Compute actual retirement age based on choice variable."""
    df = df.copy()

    retirement_values = RETIREMENT_NO_CARE_DEMAND.ravel().tolist()
    df_retirement = df[df["choice"].isin(retirement_values)].copy()
    actual_retirement_ages = df_retirement.groupby("agent")["age"].min()
    df.loc[:, "actual_retirement_age"] = df["agent"].map(actual_retirement_ages)

    return df


def create_realized_taste_shock(df, specs):
    """Create realized taste shock variable based on actual choice."""
    df.loc[:, "real_taste_shock"] = np.nan
    for choice in range(4):
        df.loc[df["choice"] == choice, "real_taste_shock"] = df.loc[
            df["choice"] == choice, f"taste_shocks_{choice}"
        ]

    return df


def create_real_utility(df, specs):
    """Create realized utility (utility + realized taste shock)."""
    df = create_realized_taste_shock(df, specs)
    df.loc[:, "real_util"] = df["utility"] + df["real_taste_shock"]

    return df


# =====================================================================================
# OLD Simulate scenario
# =====================================================================================


def simulate_scenario_no_care_demand(
    model,
    solution,
    initial_states,
    wealth_agents,
    params,
    model_specs,
    seed,
) -> pd.DataFrame:
    """Simulate the counterfactual model and return a DataFrame."""

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        wealth_initial=wealth_agents,
        # n_periods=model_specs["model_params"]["n_periods"],
        n_periods=model_specs["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model,
        model_sim=model,
    )
    df = create_simulation_df(sim_dict)

    # Add derived variables
    df["age"] = df.index.get_level_values("period") + model_specs["start_age"]

    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_specs["max_exp_diffs_per_period"],
    )

    # Assign working hours
    df["working_hours"] = 0.0
    part_time_values = PART_TIME_NO_CARE_DEMAND.ravel().tolist()
    full_time_values = FULL_TIME_NO_CARE_DEMAND.ravel().tolist()

    sex_var = SEX
    for edu_var in range(model_specs["n_education_types"]):
        # full-time
        df.loc[
            df["choice"].isin(full_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_specs["av_annual_hours_ft"][sex_var, edu_var]
        # part-time
        df.loc[
            df["choice"].isin(part_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_specs["av_annual_hours_pt"][sex_var, edu_var]

    # Income variables
    df["assets_begin_of_period"] = df["savings"] + df["consumption"]
    df["total_income"] = (
        df.groupby("agent")["assets_begin_of_period"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["assets_begin_of_period"].shift(
        -1
    ) - df["savings"] * (1 + model_specs["interest_rate"])

    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # ===============================================================================
    # Gross labor income computation
    # ===============================================================================
    work_values = part_time_values + full_time_values

    # Convert pandas Series to numpy arrays for JAX
    lagged_choice_array = np.asarray(df["lagged_choice"])
    experience_years_array = np.asarray(df["exp_years"])
    education_array = np.asarray(df["education"])
    income_shock_array = np.asarray(df["income_shock"])

    # Vectorized gross labor income calculation
    vectorized_calc_gross_labor_income = jax.vmap(
        lambda lc, exp, edu, shock: calculate_gross_labor_income(
            lagged_choice=lc,
            experience_years=exp,
            education=edu,
            sex=sex_var,
            income_shock=shock,
            model_specs=model_specs,
        )
    )
    gross_labor_income_array = vectorized_calc_gross_labor_income(
        lagged_choice_array,
        experience_years_array,
        education_array,
        income_shock_array,
    )
    df["gross_labor_income"] = gross_labor_income_array * df["lagged_choice"].isin(
        work_values
    )

    # Mother age
    df["mother_age"] = (
        df["age"].to_numpy()
        + model_specs["mother_age_diff"][df["education"].to_numpy()]
    )

    return df


def simulate_career_costs_no_care_demand(
    model,
    solution,
    initial_states,
    wealth_agents,
    params,
    options,
    seed,
) -> pd.DataFrame:
    """Ultra-fast career costs simulation for no-care-demand model.

    This function runs the simulation and computes individual income components
    directly from the simulation dictionary for maximum speed, using the restricted
    choice sets from the no-care-demand counterfactual.
    """

    # Run the simulation to get sim_dict
    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        wealth_initial=wealth_agents,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model,
        model_sim=model,
    )

    # Build simulation DataFrame from sim_dict with income components
    df = build_simulation_df_with_income_components_no_care_demand(
        sim_dict, options, params
    )

    return df


def build_simulation_df_with_income_components_no_care_demand(
    sim_dict, options, params
):
    """Build simulation DataFrame and compute income components efficiently."""

    df = create_simulation_df(sim_dict)

    # Create additional variables
    model_params = options["model_params"]
    df["age"] = df.index.get_level_values("period") + model_params["start_age"]

    # Create experience years
    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_params["max_exp_diffs_per_period"],
    )

    # Assign working hours for choice 1 (unemployed)
    df["working_hours"] = 0.0

    # Use no-care-demand choice arrays
    part_time_values = PART_TIME_NO_CARE_DEMAND.ravel().tolist()
    full_time_values = FULL_TIME_NO_CARE_DEMAND.ravel().tolist()
    work_values = WORK_NO_CARE_DEMAND.ravel().tolist()
    retirement_values = RETIREMENT_NO_CARE_DEMAND.ravel().tolist()

    sex_var = SEX

    for edu_var in range(model_params["n_education_types"]):

        # full-time
        df.loc[
            df["choice"].isin(full_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_params["av_annual_hours_ft"][sex_var, edu_var]

        # part-time
        df.loc[
            df["choice"].isin(part_time_values) & (df["education"] == edu_var),
            "working_hours",
        ] = model_params["av_annual_hours_pt"][sex_var, edu_var]

    # Compute gross labor income using JAX vmap for vectorization
    # Convert pandas Series to numpy arrays for JAX
    lagged_choice_array = np.asarray(df["lagged_choice"])
    experience_years_array = np.asarray(df["exp_years"])
    education_array = np.asarray(df["education"])
    income_shock_array = np.asarray(df["income_shock"])
    savings_array = np.asarray(df["savings"])
    has_partner_int_array = np.asarray((df["partner_state"] > 0).astype(int))
    periods_array = np.asarray(df.index.get_level_values("period"))

    # ===============================================================================
    # Gross labor and pension income
    # ===============================================================================

    # Female gross labor income
    vectorized_calc_gross_labor_income = jax.vmap(
        lambda lc, exp, edu, shock: calculate_gross_labor_income(
            lagged_choice=lc,
            experience_years=exp,
            education=edu,
            sex=sex_var,
            income_shock=shock,
            model_specs=model_params,
        )
    )
    gross_labor_income_array = vectorized_calc_gross_labor_income(
        lagged_choice_array,
        experience_years_array,
        education_array,
        income_shock_array,
    )
    df["gross_labor_income"] = gross_labor_income_array * df["lagged_choice"].isin(
        work_values
    )

    # Female gross pension income
    vectorized_calc_gross_pension_income = jax.vmap(
        lambda exp, edu: calc_gross_pension_income(
            experience_years=exp,
            education=edu,
            sex=sex_var,
            options=model_params,
        )
    )
    gross_pension_income_array = vectorized_calc_gross_pension_income(
        experience_years_array,
        education_array,
    )
    df["gross_pension_income"] = gross_pension_income_array * df["lagged_choice"].isin(
        retirement_values
    )

    # ===============================================================================
    # Benefits and costs (NO CARE BENEFITS/COSTS IN NO-CARE-DEMAND MODEL)
    # ===============================================================================

    # Female unemployment benefits
    vectorized_calc_unemployment_benefits = jax.vmap(
        lambda assets, edu, has_partner_int, period: calc_unemployment_benefits(
            assets=assets,
            sex=sex_var,
            education=edu,
            has_partner_int=has_partner_int,
            period=period,
            model_specs=model_params,
        )
    )
    unemployment_benefits_array = vectorized_calc_unemployment_benefits(
        savings_array,
        education_array,
        has_partner_int_array,
        periods_array,
    )
    df["unemployment_benefits"] = unemployment_benefits_array

    # Child benefits
    vectorized_calc_child_benefits = jax.vmap(
        lambda edu, has_partner_int, period: calc_child_benefits(
            education=edu,
            sex=sex_var,
            has_partner_int=has_partner_int,
            period=period,
            options=model_params,
        )
    )
    child_benefits_array = vectorized_calc_child_benefits(
        education_array,
        has_partner_int_array,
        periods_array,
    )
    df["child_benefits"] = child_benefits_array

    # Calculate total individual income following budget equation logic
    # Total net income = labor income + pension income + child benefits
    df["total_gross_income"] = (
        df["gross_labor_income"] + df["child_benefits"] + df["gross_pension_income"]
    )

    # Apply maximum with unemployment benefits (following budget equation)
    # df["total_income"] = np.maximum(
    #     df["total_gross_income"], df["unemployment_benefits"]
    # ) * (df["health"] != DEAD)
    df["total_income"] = np.where(
        df["health"] != DEAD,
        np.maximum(df["total_gross_income"], df["unemployment_benefits"]),
        0,
    )

    return df
