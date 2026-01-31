"""Transform and correct data for estimation.

This module contains functions to load, scale, and correct empirical data
for use in likelihood estimation, including experience scaling and wealth adjustment.
"""

import numpy as np
import pandas as pd

from caregiving.model.experience_baseline_model import scale_experience_years
from caregiving.model.pension_system.experience_stock import (
    calc_pension_points_form_experience,
)
from caregiving.model.shared import RETIREMENT, SEX, is_retired
from dcegm.asset_correction import adjust_observed_assets


def load_and_scale_correct_data(data_decision, model_class):
    """Load, scale, and correct data for estimation.

    This function:
    1. Creates age from period
    2. Transforms experience from years to normalized scale
    3. Sets assets_begin_of_period from wealth
    4. Corrects wealth to include non-pension retirement income (optional)
    5. Creates states_dict and adjusts observed assets
    6. Creates informed probability (optional)

    Parameters
    ----------
    data_decision : pd.DataFrame
        DataFrame with empirical data. Must contain columns:
        - 'period' or 'age'
        - 'experience' (in years)
        - 'wealth' or 'assets_begin_of_period'
        - 'lagged_choice' or 'choice'
        - Other state variables as required by model
    model_class : Any
        Model class instance with model_specs, model_structure, model_funcs

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed and corrected data, including:
        - 'age': Age calculated from period
        - 'experience': Normalized experience (0-1 scale)
        - 'assets_begin_of_period': Adjusted assets
        - Additional income variables from aux outputs (if available)
    """
    df = data_decision.copy()
    model_specs = model_class.model_specs

    # Create age from period if not already present
    if "age" not in df.columns:
        if "period" not in df.columns:
            raise ValueError(
                "DataFrame must contain either 'age' or 'period' column. "
                f"Available columns: {list(df.columns)}"
            )
        df["age"] = df["period"] + model_specs["start_age"]

    df["experience_years"] = df["experience"].values

    # Determine retirement status
    retirement_values = np.asarray(RETIREMENT).ravel().tolist()
    is_retired_arr = df["lagged_choice"].isin(retirement_values).values

    # Scale experience
    df["experience"] = scale_experience_years(
        experience_years=df["experience_years"].values,
        period=(
            df["period"].values
            if "period" in df.columns
            else df["age"].values - model_specs["start_age"]
        ),
        is_retired=is_retired_arr,
        model_specs=model_specs,
    )

    df["assets_begin_of_period"] = df["wealth"].values / model_specs["wealth_unit"]

    # Optional: Correct wealth to include non-pension retirement income
    df = correct_wealth_to_include_non_pension_retirement_income(
        df=df,
        model_specs=model_specs,
    )

    # Create states_dict
    states_dict = create_states_dict(df=df, model_class=model_class)

    # Adjust observed assets
    # Note: adjust_observed_assets in dcegm doesn't return aux_outs,
    # so we can't extract income variables directly
    adjusted_assets = adjust_observed_assets(
        observed_states_dict=states_dict,
        params={},  # Empty params for now, can be passed if needed
        model_class=model_class,
    )

    df["assets_begin_of_period"] = adjusted_assets

    return df


def create_states_dict(df, model_class):
    """Create a dictionary of states from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with state variables
    model_class : Any
        Model class instance with model_structure

    Returns
    -------
    dict
        Dictionary mapping state variable names to numpy arrays
    """
    model_structure = model_class.model_structure

    states_dict = {}
    for name in model_structure["discrete_states_names"]:
        if name and name in df.columns:
            states_dict[name] = df[name].values.copy()

    states_dict["experience"] = df["experience"].values
    states_dict["assets_begin_of_period"] = df["assets_begin_of_period"].values

    # Add care_demand and mother_dead if needed (set to zeros if not present)
    if "care_demand" not in df.columns:
        states_dict["care_demand"] = np.zeros(len(df), dtype=np.uint8)
    else:
        states_dict["care_demand"] = df["care_demand"].values

    if "mother_dead" not in df.columns:
        states_dict["mother_dead"] = np.zeros(len(df), dtype=np.uint8)
    else:
        states_dict["mother_dead"] = df["mother_dead"].values

    return states_dict


def calc_pension_annuity_value(pension_payments, payment_years, interest_rate):
    """Calculate pension annuity value.

    Calculates perpetuity of getting pension income after SRA,
    then subtracts perpetuity after death.

    Parameters
    ----------
    pension_payments : array-like
        Annual pension payment amounts
    payment_years : array-like
        Number of years payments will be received
    interest_rate : float
        Interest rate for discounting

    Returns
    -------
    array-like
        Present value of pension annuity
    """
    discount_factor = 1 / (1 + interest_rate)
    annuity_factor = 1 / (1 - discount_factor)
    pension_annuity_value = (
        pension_payments * annuity_factor
        - pension_payments * annuity_factor * discount_factor**payment_years
    )
    return pension_annuity_value


def correct_wealth_to_include_non_pension_retirement_income(df, model_specs):
    """Correct wealth to include non-pension retirement income.

    This function assumes that non-pension retirement income is a fraction
    of pension wealth. It computes pension points, calculates pension annuity value,
    and adds a fraction to assets_begin_of_period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with experience_years, education, sex, age
    model_specs : dict
        Model specifications including:
        - annual_pension_point_value
        - interest_rate
        - wealth_unit
        - life_exp: array of shape (sex, education) with life expectancies

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected assets_begin_of_period
    """
    df = df.copy()

    # Get sex values (default to SEX if not in dataframe)
    sex_values = df["sex"].values if "sex" in df.columns else np.full(len(df), SEX)

    # Calculate pension points from experience years
    pension_points = calc_pension_points_form_experience(
        education=df["education"].values,
        sex=sex_values,
        experience_years=df["experience_years"].values,
        model_specs=model_specs,
    )

    # Calculate current pension
    df["current_pension"] = pension_points * model_specs["annual_pension_point_value"]

    # Get life expectancy for each individual
    # life_exp is array of shape (sex, education)
    life_expectancy = model_specs["life_exp"][sex_values, df["education"].values]

    # Policy state value is min_SRA (standard retirement age)
    # Payments start at max(min_SRA, age)
    min_SRA = model_specs["min_SRA"]
    age_payments_start = np.maximum(min_SRA, df["age"].values)
    age_payments_end = np.maximum(life_expectancy, df["age"].values)
    df["payment_years"] = age_payments_end - age_payments_start
    df["years_until_payment_starts"] = age_payments_start - df["age"].values

    # Calculate pension annuity value using the provided function
    interest_rate = model_specs["interest_rate"]
    df["pension_annuity_value"] = calc_pension_annuity_value(
        pension_payments=df["current_pension"].values,
        payment_years=df["payment_years"].values,
        interest_rate=interest_rate,
    )

    # Discount to present day
    discount_factor = 1 / (1 + interest_rate)
    df["pension_wealth"] = (
        df["pension_annuity_value"]
        * discount_factor ** df["years_until_payment_starts"]
    )

    # Add fraction of pension wealth to assets_begin_of_period
    # Default fraction is 0.4 (40% of pension wealth as non-pension retirement income)
    wealth_correction_fraction = 0.4
    df["wealth_correction"] = (
        wealth_correction_fraction * df["pension_wealth"] / model_specs["wealth_unit"]
    )
    df["assets_begin_of_period"] = (
        df["assets_begin_of_period"] + df["wealth_correction"]
    )

    return df
