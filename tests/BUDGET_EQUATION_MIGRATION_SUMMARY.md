# Budget Equation Migration Summary

## Overview
This document summarizes the differences between the outdated test file (`test_budget_constraint.py`) and the current `budget_equation.py` implementation.

## Key Differences

### 1. Function Signature Changes

#### Current Implementation (`budget_equation.py`)
```python
def budget_constraint(
    period,
    education,
    lagged_choice,
    experience,
    partner_state,
    care_demand,
    mother_dead,
    asset_end_of_previous_period,
    income_shock_previous_period,
    model_specs,  # NO params argument
):
```

#### Outdated Test Assumptions
- Test passes `params={"interest_rate": ...}` argument (REMOVED)
- Test uses `sex` parameter (REMOVED - now hardcoded as `SEX = 1`)

### 2. Experience Calculation

#### Current Implementation
- Uses `construct_experience_years()` function to convert float experience to experience years
- Takes `float_experience`, `period`, `is_retired`, `model_specs` as inputs
- Returns experience in years (not a normalized 0-1 value)
- Formula: `float_experience * scale` where scale depends on retirement status and period

#### Outdated Test Assumptions
- Manually calculates experience as: `experience / max_init_exp_period`
- Uses `max_exp_diffs_per_period` from specs (now uses `max_exps_period_working`)
- Doesn't account for retirement status in experience scaling

### 3. Wage/Labor Income Calculation

#### Current Implementation
- Uses `calc_labor_income_after_ssc()` function
- Takes `experience_years` (not raw experience)
- Uses `calc_hourly_wage()` with `gamma_0` and `gamma_ln_exp` from specs
- Formula: `exp(gamma_0 + gamma_ln_exp * ln(experience_years + 1) + income_shock)`
- Applies minimum wage check after calculation
- Returns both `labor_income_after_ssc` and `gross_labor_income`

#### Outdated Test Assumptions
- Manually calculates hourly wage using `gamma_array` (old format)
- Formula: `exp(gamma_array[sex, education] + gamma_array[sex, education] * log(experience + 1) + income_shock)`
- Directly accesses `av_annual_hours_pt/ft` and `annual_min_wage_pt/ft`
- Doesn't use centralized wage calculation functions

### 4. Pension/Retirement Income Calculation

#### Current Implementation
- Uses `calc_pensions_after_ssc()` function
- Takes `pension_points` (which is `experience_years` for retirees)
- Calculates gross pension as: `annual_pension_point_value * pension_points`
- Applies health/LTC contributions only (no pension/unemployment contributions)
- Handles early/late retirement adjustments internally via experience years

#### Outdated Test Assumptions
- Manually calculates pension points using old formula:
  ```python
  total_pens_points = (exp(gamma_0) / (gamma_1 + 1)) * ((exp + 1)^(gamma_1 + 1) - 1) / mean_wage_all
  ```
- Manually applies early retirement penalty/late retirement bonus
- Uses `get_next_period_experience()` for fresh retirees (not used in budget constraint)
- Directly calculates pension from pension points

### 5. Inheritance Handling

#### Current Implementation
- Checks `mother_dead == PARENT_RECENTLY_DEAD` (state 1) for inheritance
- Uses `calc_inheritance_amount()` which takes `period`, `lagged_choice`, `education`, `model_specs` (NO `mother_dead`)
- Uses precomputed `inheritance_amount_mat` from specs (indexed by sex, period, education, care_type)
- Uses `draw_inheritance_outcome()` for stochastic inheritance (takes `asset_end_of_previous_period`)
- Formula: `bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount`

#### Outdated Test Assumptions
- Uses `mother_dead=PARENT_LONGER_DEAD` (state 2) in tests
- `calc_inheritance_amount()` was called with `mother_dead` parameter (REMOVED)
- No stochastic inheritance drawing (`draw_inheritance_outcome`)

### 6. Unemployment Benefits

#### Current Implementation
- Function signature: `calc_unemployment_benefits(assets, sex, education, has_partner_int, period, model_specs)`
- Note: Parameter order is `assets, sex, education, ...` (sex before education)

#### Outdated Test Assumptions
- Test calls with order: `assets, education, sex, ...` (education before sex) - WRONG ORDER

### 7. Partner Income

#### Current Implementation
- Uses `calc_partner_income_after_ssc()` which returns tuple:
  `(partner_income_after_ssc, gross_partner_income, gross_partner_pension)`
- All three values are returned and used in government budget calculations

#### Outdated Test Assumptions
- Test manually calculates partner income based on partner_state
- Doesn't separate gross partner income from gross partner pension

### 8. Total Income Calculation

#### Current Implementation
- Formula: `total_income = max(total_net_household_income + child_benefits, household_unemployment_benefits)`
- Includes child benefits in the max calculation
- Then adds: `total_income_plus_interest = total_income + interest + care_benefits_and_costs + bequest_from_parent`

#### Outdated Test Assumptions
- Formula: `total_income = max(total_net_household_income, household_unemployment_benefits)`
- Child benefits not included in max calculation
- Comment mentions "without child benefits and care benefits - they are overwritten on line 135" (outdated)

### 9. Government Budget Components

#### Current Implementation
- Uses `calc_government_budget_components()` function
- Receives all gross income components separately:
  - `gross_labor_income`, `gross_retirement_income`
  - `gross_partner_income`, `gross_partner_pension`
- Returns tuple: `(income_tax_total, own_ssc, partner_ssc, total_tax_revenue, government_expenditures, net_government_budget)`
- All values stored in `budget_aux` dictionary, scaled by `wealth_unit`

#### Outdated Test Assumptions
- Test manually calculates income tax using `calc_net_household_income()`
- Doesn't account for all gross income components separately
- Test structure suggests older government budget calculation approach

### 10. Return Values

#### Current Implementation
- Returns: `(assets_begin_of_period / wealth_unit, aux)`
- `aux` dictionary contains extensive information including:
  - All income components (scaled by wealth_unit)
  - Government budget components (scaled by wealth_unit)
  - Care benefits and costs
  - Child benefits
  - Inheritance information
  - Income shocks

#### Outdated Test Assumptions
- Test expects only wealth value
- Test doesn't validate `budget_aux` contents (except in one test function)

### 11. Income Shock Handling

#### Current Implementation
- For period 0: uses `model_specs["income_shock_mean"]` (defaults to 0.0)
- For period > 0: uses `income_shock_previous_period`
- Stored in `budget_aux["income_shock_for_labor"]`

#### Outdated Test Assumptions
- Test uses `income_shock_previous_period=0` for all periods
- No special handling for period 0

### 12. Mother Death State Handling

#### Current Implementation
- Uses `PARENT_RECENTLY_DEAD = 1` to check if inheritance should be paid
- Only inheritance is paid when `mother_dead == PARENT_RECENTLY_DEAD`
- `PARENT_LONGER_DEAD = 2` means no inheritance (mother died in previous periods)

#### Outdated Test Assumptions
- Tests use `PARENT_LONGER_DEAD` expecting no inheritance (correct intent, but should use `PARENT_RECENTLY_DEAD` for inheritance tests)
- May need separate test cases for inheritance scenarios

## Summary of Required Test Updates

1. **Remove `params` argument** from all `budget_constraint()` calls
2. **Remove `sex` parameter** from function calls (hardcoded as `SEX = 1`)
3. **Use `construct_experience_years()`** to convert experience instead of manual calculation
4. **Use `calc_labor_income_after_ssc()`** instead of manual wage calculation
5. **Use `calc_pensions_after_ssc()`** instead of manual pension calculation
6. **Fix `calc_unemployment_benefits()` parameter order** (sex before education)
7. **Update inheritance handling** to use current functions and `PARENT_RECENTLY_DEAD` for inheritance tests
8. **Update total income calculation** to match current formula (includes child_benefits in max)
9. **Use returned `budget_aux`** for validation instead of manual recalculation where appropriate
10. **Handle period 0 income shock** correctly
11. **Use `calc_government_budget_components()`** results from `budget_aux` for validation
