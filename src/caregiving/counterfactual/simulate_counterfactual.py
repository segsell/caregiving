"""Function that simulates the model for a given scenario."""

import numpy as np
import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods

from caregiving.model.shared import (
    DEAD,
    FULL_TIME_CHOICES,
    INFORMAL_CARE,
    PARENT_DEAD,
    PART_TIME,
    SEX,
)
from caregiving.model.state_space import construct_experience_years
from caregiving.utils import table


def simulate_counterfactual_npv(
    model,
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
    full_time_values = FULL_TIME_CHOICES.ravel().tolist()

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
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]

    # Then total income as the difference between wealth at the beginning
    # of next period and savings
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["wealth_at_beginning"].shift(
        -1
    ) - df["savings"] * (1 + options["interest_rate"])

    # periodic savings and savings rate
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # # Caregiving
    # df["informal_care"] = df["choice"].isin(np.asarray(INFORMAL_CARE))

    # # Care ever
    # df["care_ever"] = df.groupby("agent")["informal_care"].transform(
    #     lambda x: x.cumsum().clip(upper=1)
    # )

    # # Sum care
    # df["sum_informal_care"] = df.groupby("agent")["informal_care"].transform(
    #     lambda x: x.cumsum()
    # )
    # df = create_care_flags(df)

    # # 0. Flag care on the *full* df_sim first (only once is enough)
    # df["is_care"] = df["choice"].isin(np.asarray(INFORMAL_CARE))

    # PARAMETERS
    # AGE_FOCUS = 75
    AGE_MIN, AGE_MAX = 40, 70
    beta = params["beta"]  # discount factor

    # # ---------------------------------------------------------------
    # # 1. Agents alive / observed at the focus age
    # alive_at_focus = (df["age"] == AGE_FOCUS) & (df["health"] != DEAD)

    # ids_at_age = df.index.get_level_values("agent")[alive_at_focus].unique()

    # # ---------------------------------------------------------------
    # # 2. Keep their entire life histories
    # mask = df.index.get_level_values("agent").isin(ids_at_age)

    # # ---------------------------------------------------------------
    # # 3. Person-level informal-care aggregates (written in place)
    # df.loc[mask, "care_sum"] = (
    #     df.loc[mask]
    #     .groupby(level="agent")["is_care"]
    #     .transform("sum")  # total care years
    # )

    # df.loc[mask, "care_ever"] = (
    #     df.loc[mask]
    #     .groupby(level="agent")["is_care"]
    #     .transform("any")  # at least one care year
    #     .astype(int)  # 1/0 instead of True/False
    # )

    # # ---------------------------------------------------------------
    # # 4. Fill rows for agents who died before AGE_FOCUS, etc.
    # df[["care_sum", "care_ever"]] = df[
    # ["care_sum", "care_ever"]
    # ].fillna(0).astype(int)

    # ===============================================================
    # Net present value (NPV) of total income, ages 30-80
    # ===============================================================

    # 1. Restrict to the evaluation window
    mask_income = df["age"].between(AGE_MIN, AGE_MAX)

    # 2. Per-row discounted income (present value at age 30)
    df.loc[mask_income, "disc_income"] = df.loc[mask_income, "total_income"] * beta ** (
        df.loc[mask_income, "age"] - AGE_MIN
    )  # period = age-30

    # 3. Agent-level NPV, written back row-aligned
    df.loc[mask_income, "npv_income_30_80"] = (
        df.loc[mask_income].groupby(level="agent")["disc_income"].transform("sum")
    )

    # 4. Zero for everyone else (never reached 30, etc.)
    df["npv_income_30_80"] = df["npv_income_30_80"].fillna(0)

    return df


def create_care_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create care ever and sum care variables."""
    # Caregiving
    df["informal_care"] = df["choice"].isin(np.asarray(INFORMAL_CARE))

    # Care ever
    df["care_ever"] = df.groupby("agent")["informal_care"].transform(
        lambda x: x.cumsum().clip(upper=1)
    )

    # Sum care
    df["sum_informal_care"] = df.groupby("agent")["informal_care"].transform(
        lambda x: x.cumsum()
    )

    # Care demand ever
    df["care_demand_ever"] = df.groupby("agent")["care_demand"].transform(
        lambda x: x.cumsum().clip(upper=1)
    )

    return df


def compute_npv(df, age_min, age_max, beta):
    # 1. Restrict to the evaluation window
    mask_income = df["age"].between(age_min, age_max)

    # 2. Per-row discounted income (present value at age 30)
    df.loc[mask_income, "disc_income"] = df.loc[mask_income, "total_income"] * beta ** (
        df.loc[mask_income, "age"] - age_min
    )  # period = age-30

    # 3. Agent-level NPV, written back row-aligned
    df.loc[mask_income, "npv_income_30_80"] = (
        df.loc[mask_income].groupby(level="agent")["disc_income"].transform("sum")
    )

    # 4. Zero for everyone else (never reached 30, etc.)
    df["npv_income_30_80"] = df["npv_income_30_80"].fillna(0)
