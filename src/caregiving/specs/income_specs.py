import numpy as np
import pandas as pd
from jax import numpy as jnp

from caregiving.model.shared import FULL_TIME_CHOICES, PART_TIME


def add_income_specs(
    specs,
    wage_params,
    partner_wage_params_men,
    partner_wage_params_women,
    avg_working_hours,
    mean_annual_wage,
):
    """Add income specs."""

    # wages
    (
        specs["gamma_0"],
        specs["gamma_1"],
        specs["income_shock_scale"],
    ) = process_wage_params(specs, wage_params)

    # unemployment benefits
    (
        specs["annual_unemployment_benefits"],
        specs["annual_unemployment_benefits_housing"],
        specs["annual_child_unemployment_benefits"],
    ) = calc_annual_unemployment_benefits(specs)

    specs["annual_child_benefits"] = specs["monthly_child_benefits"] * 12

    # pensions
    specs["annual_pension_point_value"] = calc_annual_pension_point_value(specs)

    # partner income
    (
        specs["annual_partner_wage"],
        specs["annual_partner_pension"],
    ) = calculate_partner_income(
        specs, partner_wage_params_men, partner_wage_params_women
    )

    specs = add_average_population_hours(specs, avg_working_hours, mean_annual_wage)

    # Add minimum wage
    specs["annual_min_wage_pt"], specs["annual_min_wage_ft"] = add_pt_and_ft_min_wage(
        specs
    )

    return specs


def calc_annual_unemployment_benefits(specs):
    """Calculate annual unemployment benefits."""

    annual_unemployment_benefits = specs["monthly_unemployment_benefits"] * 12
    annual_unemployment_benefits_housing = (
        specs["monthly_unemployment_benefits_housing"] * 12
    )
    annual_child_unemployment_benefits = (
        specs["monthly_child_unemployment_benefits"] * 12
    )
    return (
        annual_unemployment_benefits,
        annual_unemployment_benefits_housing,
        annual_child_unemployment_benefits,
    )


def add_average_population_hours(specs, pop_averages, mean_annual_wage):
    """Add average population hours to specs."""

    # part_time_values = np.asarray(PART_TIME).ravel().tolist()
    # full_time_values = np.asarray(FULL_TIME).ravel().tolist()
    part_time_values = [2]
    full_time_values = [3]

    # Assign population averages
    av_annual_hours_pt = np.zeros(
        (specs["n_sexes"], specs["n_education_types"]), dtype=float
    )
    av_annual_hours_ft = np.zeros(
        (specs["n_sexes"], specs["n_education_types"]), dtype=float
    )

    for edu_var, _ in enumerate(specs["education_labels"]):
        for sex_var, _ in enumerate(specs["sex_labels"]):
            mask = (pop_averages["education"] == edu_var) & (
                pop_averages["sex"] == sex_var
            )
            av_annual_hours_pt[sex_var, edu_var] = pop_averages.loc[
                mask & pop_averages["choice"].isin(part_time_values), "annual_hours"
            ].values[0]

            av_annual_hours_ft[sex_var, edu_var] = pop_averages.loc[
                mask & pop_averages["choice"].isin(full_time_values), "annual_hours"
            ].values[0]

    specs["av_annual_hours_pt"] = jnp.asarray(av_annual_hours_pt)
    specs["av_annual_hours_ft"] = jnp.asarray(av_annual_hours_ft)

    # Create auxiliary mean hourly full time wage for pension calculation
    # (see appendix)
    specs["mean_hourly_ft_wage"] = jnp.asarray(mean_annual_wage / av_annual_hours_ft)
    # ?! Why larger for women?!

    return specs


def add_pt_and_ft_min_wage(specs):
    """Compute the annual minimum wage for part-time and full-time workers.

    Type-specific as hours are different between types.

    """
    annual_min_wage_pt = np.zeros((specs["n_education_types"], 2))

    for edu in range(specs["n_education_types"]):
        for sex in (0, 1):
            hours_ratio = (
                specs["av_annual_hours_pt"][edu][sex]
                / specs["av_annual_hours_ft"][edu][sex]
            )
            annual_min_wage_pt[edu][sex] = specs["monthly_min_wage"] * hours_ratio * 12

    return jnp.asarray(annual_min_wage_pt), specs["monthly_min_wage"] * 12


def calc_annual_pension_point_value(specs):
    """Generate average pension point value weighted by east and west pensions."""

    pension_point_value = (
        0.75 * specs["monthly_pension_point_value_west_2010"]
        + 0.25 * specs["monthly_pension_point_value_east_2010"]
    )
    return pension_point_value * 12


def process_wage_params(specs, wage_params):
    """Process wage parameters."""
    wage_params.reset_index(inplace=True)

    gamma_0 = np.zeros((specs["n_sexes"], specs["n_education_types"]), dtype=float)
    gamma_1 = np.zeros((specs["n_sexes"], specs["n_education_types"]), dtype=float)

    for edu_id, edu_label in enumerate(specs["education_labels"]):
        for sex_id, sex in enumerate(specs["sex_labels"]):
            mask = (wage_params["education"] == edu_label) & (wage_params["sex"] == sex)
            gamma_0[sex_id, edu_id] = wage_params.loc[
                mask & (wage_params["parameter"] == "constant"), "value"
            ].values[0]
            gamma_1[sex_id, edu_id] = wage_params.loc[
                mask & (wage_params["parameter"] == "ln_exp"), "value"
            ].values[0]

    mask = (
        (wage_params["education"] == "all")
        & (wage_params["sex"] == "all")
        & (wage_params["parameter"] == "income_shock_std")
    )
    income_shock_scale = wage_params.loc[mask, "value"].values[0]

    return jnp.asarray(gamma_0), jnp.asarray(gamma_1), income_shock_scale


def calculate_partner_income(specs, partner_wage_params_men, partner_wage_params_women):
    """Calculate income of working-age partner."""

    periods = np.arange(0, specs["n_periods"], dtype=float)

    # Limit periods to the one of max retirement age as we restricted our estimation
    # sample until then.
    # For the predictions after max retirement age we use the last max retirement period
    not_predicted_periods = np.where(
        periods > specs["max_ret_age"] - specs["start_age"]
    )[0]
    periods[not_predicted_periods] = specs["max_ret_age"] - specs["start_age"]

    partner_monthly_income = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], specs["n_periods"]), dtype=float
    )

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        if sex_label == "Women":
            params = partner_wage_params_women
        else:
            params = partner_wage_params_men

        for edu_var, edu_label in enumerate(specs["education_labels"]):
            mask = params["education"] == edu_label
            partner_monthly_income[sex_var, edu_var, :] = (
                params.loc[mask, "constant"].values[0]
                + params.loc[mask, "period"].values[0] * periods
                + params.loc[mask, "period_sq"].values[0] * periods**2
            )
    annual_partner_income = partner_monthly_income * 12

    # Quasi wealth hack
    annual_partner_pension = (
        annual_partner_income[:, :, ~not_predicted_periods].mean(axis=2) * 0.48
    )

    return jnp.asarray(annual_partner_income), jnp.asarray(annual_partner_pension)
