"""Simulation helper for the no-care-demand counterfactual.

Mirrors the baseline simulate_scenario but uses the reduced 4-state
choice arrays from shared_no_care_demand to assign working hours, etc.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.pre_processing.setup_model import load_and_setup_model
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


def setup_model_for_simulation_no_care_demand(path_to_model, options):
    """Setup no-care-demand model for simulation with correct utility functions."""
    return load_and_setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
        path=path_to_model,
        sim_model=True,
    )


def simulate_scenario_no_care_demand(
    model,
    solution,
    initial_states,
    wealth_agents,
    params,
    options,
    seed,
) -> pd.DataFrame:
    """Simulate the counterfactual model and return a DataFrame."""

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

    # Add derived variables
    model_params = options["model_params"]
    df["age"] = df.index.get_level_values("period") + model_params["start_age"]

    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_params["max_exp_diffs_per_period"],
    )

    # Assign working hours
    df["working_hours"] = 0.0
    part_time_values = PART_TIME_NO_CARE_DEMAND.ravel().tolist()
    full_time_values = FULL_TIME_NO_CARE_DEMAND.ravel().tolist()

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

    # Income variables
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["wealth_at_beginning"].shift(
        -1
    ) - df["savings"] * (1 + params["interest_rate"])

    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # Mother age
    df["mother_age"] = (
        df["age"].to_numpy()
        + model_params["mother_age_diff"][
            df["has_sister"].to_numpy(), df["education"].to_numpy()
        ]
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
    # Benefits and costs (NO CARE BENEFITS/COSTS IN NO-CARE-DEMAND MODEL)
    # ===============================================================================

    # Female unemployment benefits
    vectorized_calc_unemployment_benefits = jax.vmap(
        lambda savings, edu, has_partner_int, period: calc_unemployment_benefits(
            savings=savings,
            sex=sex_var,
            education=edu,
            has_partner_int=has_partner_int,
            period=period,
            options=model_params,
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
    df["total_net_income"] = (
        df["gross_labor_income"]
        + df["child_benefits"]
        # df["gross_labor_income"] + df["gross_pension_income"] + df["child_benefits"]
    )

    # Apply maximum with unemployment benefits (following budget equation)
    # df["total_income"] = np.maximum(
    #     df["total_net_income"], df["unemployment_benefits"]
    # ) * (df["health"] != DEAD)
    df["total_income"] = np.where(
        df["health"] != DEAD,
        np.maximum(df["total_net_income"], df["unemployment_benefits"]),
        0,
    )

    return df
