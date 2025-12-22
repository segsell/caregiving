"""Function that simulates the model for a given scenario."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from caregiving.model.shared import (
    DEAD,
    FULL_TIME,
    INFORMAL_CARE,
    PARENT_DEAD,
    PART_TIME,
    RETIREMENT,
    SEX,
    UNEMPLOYED,
    WORK,
    is_full_time,
    is_part_time,
    is_retired,
    is_unemployed,
    is_working,
)
from caregiving.model.state_space import (
    construct_experience_years,
    create_state_space_functions,
)
from caregiving.model.state_space_job_retention import (
    create_state_space_functions as create_state_space_functions_job_retention,
)
from caregiving.model.utility.bequest_utility import (
    create_final_period_utility_functions,
)
from caregiving.model.utility.utility_functions_additive import create_utility_functions
from caregiving.model.wealth_and_budget.budget_equation import budget_constraint
from caregiving.model.wealth_and_budget.pensions import (
    calc_gross_pension_income,
    calc_pensions_after_ssc,
)
from caregiving.model.wealth_and_budget.transfers import (
    calc_care_benefits_and_costs,
    calc_child_benefits,
    calc_unemployment_benefits,
)
from caregiving.model.wealth_and_budget.wages import (
    calc_labor_income_after_ssc,
    calculate_gross_labor_income,
)
from caregiving.utils import table
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods

jax.config.update("jax_enable_x64", True)


def setup_model_for_simulation_baseline(path_to_model, options):
    """Setup baseline model for simulation with correct utility functions."""
    return load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
        sim_model=True,
    )


def setup_model_for_simulation_job_retention(path_to_model, options):
    """Setup job retention model for simulation with correct utility functions."""
    return load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions_job_retention(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
        sim_model=True,
    )


def simulate_scenario(
    model,
    # solution_endog_grid,
    # solution_value,
    # solution_policy,
    solution,
    initial_states,
    wealth_agents,
    params,
    options,
    seed,
) -> pd.DataFrame:
    """Simulate the model for given parametrization and model solution."""

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
    df = create_simulation_df(sim_dict)

    # Create additional variables
    model_params = options["model_params"]
    max_ret_age = model_params["max_ret_age"]

    df["age"] = df.index.get_level_values("period") + model_params["start_age"]

    # Create experience years
    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_params["max_exp_diffs_per_period"],
    )

    # Assign working hours for choice 1 (unemployed)
    df["working_hours"] = 0.0

    part_time_values = PART_TIME.ravel().tolist()
    full_time_values = FULL_TIME.ravel().tolist()
    retirement_values = RETIREMENT.ravel().tolist()
    work_values = part_time_values + full_time_values

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

    # Create income vars:
    # First wealth at the beginning of period as the sum of savings and consumption
    df["assets_begin_of_period"] = df["savings"] + df["consumption"]

    # Then total income as the difference between wealth at the beginning
    # of next period and savings
    df["total_income"] = (
        df.groupby("agent")["assets_begin_of_period"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["assets_begin_of_period"].shift(
        -1
    ) - df["savings"] * (1 + options["interest_rate"])

    # periodic savings and savings rate
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # ===============================================================================
    # Gross labor income computation
    # ===============================================================================

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
            options=model_params,
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

    # # Caregiving
    # df["informal_care"] = np.nan
    # df["formal_care"] = np.nan

    # alive_and_demand = (
    #     (df["health"] != DEAD)
    #     & (df["mother_health"] != PARENT_DEAD)
    #     & (df["care_demand"] == 1)
    # )

    # df.loc[alive_and_demand & (df["choice"].isin(INFORMAL_CARE)), "informal_care"] = 1
    # df.loc[alive_and_demand & (~df["choice"].isin(INFORMAL_CARE)),
    # "informal_care"] = 0

    # df.loc[alive_and_demand & (~df["choice"].isin(INFORMAL_CARE)), "formal_care"] = 1
    # df.loc[alive_and_demand & (df["choice"].isin(INFORMAL_CARE)), "formal_care"] = 0

    df["mother_age"] = (
        df["age"].to_numpy()
        + model_params["mother_age_diff"][df["education"].to_numpy()]
    )

    # Drop all agents (entirely) who work after the maximum retirement age
    # Identify agents who work (not retired) after max_ret_age
    agents_working_after_ret = (
        df.loc[(~df["choice"].isin(retirement_values)) & (df["age"] > max_ret_age)]
        .index.get_level_values("agent")
        .unique()
    )

    # Drop all rows for those agents
    df = df[~df.index.get_level_values("agent").isin(agents_working_after_ret)]

    return df


def simulate_career_costs(
    model,
    solution,
    initial_states,
    wealth_agents,
    params,
    options,
    seed,
) -> pd.DataFrame:
    """Ultra-fast career costs simulation: build from sim_dict and compute income.

    This function runs the simulation and computes individual income components
    directly from the simulation dictionary for maximum speed.
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
    df = build_simulation_df_with_income_components(sim_dict, options, params)

    return df


def build_simulation_df_with_income_components(sim_dict, options, params):
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

    part_time_values = PART_TIME.ravel().tolist()
    full_time_values = FULL_TIME.ravel().tolist()
    work_values = WORK.ravel().tolist()
    retirement_values = RETIREMENT.ravel().tolist()

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
    care_demand_array = np.asarray(df["care_demand"])

    # ===============================================================================
    # Gross abor and pension income
    # ===============================================================================

    # Female gross labor income
    vectorized_calc_gross_labor_income = jax.vmap(
        lambda lc, exp, edu, shock: calculate_gross_labor_income(
            lagged_choice=lc,
            experience_years=exp,
            education=edu,
            sex=sex_var,
            income_shock=shock,
            options=model_params,
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
    # Benefits and costs
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

    # Care benefits and costs
    vectorized_calc_care_benefits_and_costs = jax.vmap(
        lambda lc, edu, care_demand: calc_care_benefits_and_costs(
            lagged_choice=lc,
            education=edu,
            care_demand=care_demand,
            options=model_params,
        )
    )
    care_benefits_costs_array = vectorized_calc_care_benefits_and_costs(
        lagged_choice_array,
        education_array,
        care_demand_array,
    )
    df["care_benefits_and_costs"] = care_benefits_costs_array

    # Calculate total individual income following budget equation logic
    # Total net income = labor income + pension income + child benefits + care benefits
    df["total_gross_income"] = (
        df["gross_labor_income"]
        + df["gross_pension_income"]
        + df["child_benefits"]
        + df["care_benefits_and_costs"]
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
