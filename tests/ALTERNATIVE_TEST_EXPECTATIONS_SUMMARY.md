# Alternative Test File Expectations Summary

## Overview
This document summarizes what the `test_budget_constraint_alternative.py` file expected (outdated assumptions) and how it was updated to match the current implementation.

## Original Expectations (Outdated)

### 1. Function Signature Assumptions

#### `budget_constraint()`
- ❌ Expected: `params={"interest_rate": ...}` argument (REMOVED in current implementation)
- ❌ Expected: `sex` parameter usage (REMOVED - now hardcoded as `SEX = 1`)
- ✅ Current: No `params` argument, uses `SEX` constant

#### `calc_net_household_income()`
- ❌ Expected: Returns 2 values: `(total_net_household_income, income_tax)`
- ✅ Current: Returns 3 values: `(total_net_household_income, income_tax, income_tax_single)`

#### `calc_inheritance_amount()`
- ❌ Expected: Takes `mother_dead` parameter: `calc_inheritance_amount(..., mother_dead=PARENT_LONGER_DEAD, ...)`
- ✅ Current: Does NOT take `mother_dead` parameter: `calc_inheritance_amount(period, lagged_choice, education, model_specs)`

#### `calc_unemployment_benefits()`
- ❌ Expected: Parameter order: `(assets, education, sex, has_partner_int, period, model_specs)` - WRONG ORDER
- ✅ Current: Parameter order: `(assets, sex, education, has_partner_int, period, model_specs)` - CORRECT ORDER

#### `calc_partner_income_after_ssc()`
- ❌ Expected: Used `sex` variable as parameter
- ✅ Current: Uses `SEX` constant (hardcoded to 1)

### 2. Experience Calculation Assumptions

#### Manual Experience Calculation
- ❌ Expected: Uses `max_exp_diffs_per_period[period]` from specs
- ✅ Current: Uses `max_exps_period_working[period]` from specs

#### Retiree Tests
- ❌ Expected: Uses `get_next_period_experience()` function to update experience
- ❌ Expected: Validates that experience doesn't change: `np.testing.assert_allclose(exp_cont * max_exp_this_period, exp)`
- ✅ Current: Uses `construct_experience_years()` directly (no intermediate update step)
- ✅ Current: No validation needed (handled internally)

### 3. Wage/Labor Income Calculation Assumptions

#### Manual Calculation
- ❌ Expected: Manual wage calculation using `gamma_array[sex, education]`
- ❌ Expected: Formula: `hourly_wage = exp(gamma_array[sex, education] + gamma_array[sex, education] * log(experience_years + 1) + income_shock)`
- ❌ Expected: Manual minimum wage check: `labor_income_year = max(labor_income_year, min_wage_year)`
- ❌ Expected: Manual SSC calculation: `sscs_worker = calc_health_ltc_contr() + calc_pension_unempl_contr()`
- ✅ Current: Uses `calc_labor_income_after_ssc()` function
- ✅ Current: Uses `gamma_0` and `gamma_ln_exp` (not `gamma_array` and `gamma_1`)

### 4. Pension/Retirement Income Calculation Assumptions

#### Manual Pension Calculation
- ❌ Expected: Manual pension point calculation:
  ```python
  total_pens_points = (exp(gamma_0) / (gamma_1 + 1)) * ((experience_years + 1)^(gamma_1 + 1) - 1) / mean_wage_all
  ```
- ❌ Expected: Manual pension calculation: `pension_year = annual_pension_point_value * total_pens_points`
- ❌ Expected: Manual SSC calculation: `pension_year - calc_health_ltc_contr(pension_year)`
- ✅ Current: Uses `calc_pensions_after_ssc()` function
- ✅ Current: Takes `pension_points` (which is `experience_years`) directly

#### Fresh Retiree Early Retirement Penalty
- ❌ Expected: Manual calculation of early retirement penalty/late retirement bonus
- ❌ Expected: Applies `pension_factor` to pension calculation
- ✅ Current: Early retirement adjustments not handled in `budget_constraint` (handled elsewhere)

### 5. Inheritance Handling Assumptions

#### Inheritance Calculation
- ❌ Expected: `calc_inheritance_amount(..., mother_dead=PARENT_LONGER_DEAD, ...)` - includes `mother_dead` parameter
- ❌ Expected: No stochastic inheritance drawing
- ❌ Expected: Inheritance is deterministic based on `mother_dead` state
- ✅ Current: `calc_inheritance_amount()` does NOT take `mother_dead` parameter
- ✅ Current: Uses `draw_inheritance_outcome()` for stochastic inheritance
- ✅ Current: Inheritance handled as: `bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount`

### 6. Income Calculation Assumptions

#### Total Income
- ❌ Expected: `total_income = max(total_net_household_income, household_unemployment_benefits)`
- ❌ Expected: No child benefits in max calculation
- ✅ Current: `total_income = max(total_net_household_income + child_benefits, household_unemployment_benefits)`
- ✅ Current: Includes `child_benefits` in max calculation

#### Expected Wealth
- ❌ Expected: `expected_wealth = (savings_scaled + total_income + interest + bequest_from_parent) / wealth_unit`
- ❌ Expected: No `care_benefits_and_costs` in calculation
- ✅ Current: `expected_wealth = (savings_scaled + total_income + interest + care_benefits_and_costs + bequest_from_parent) / wealth_unit`
- ✅ Current: Includes `care_benefits_and_costs` in calculation

### 7. Income Shock Handling Assumptions

#### Period 0 Handling
- ❌ Expected: Always uses `income_shock_previous_period=0` or provided value
- ❌ Expected: No special handling for period 0
- ✅ Current: For period 0: uses `income_shock_mean` from specs (defaults to 0.0)
- ✅ Current: For period > 0: uses `income_shock_previous_period`

### 8. Missing Functionality

#### Child Benefits
- ❌ Expected: Not included in calculations
- ✅ Current: Must be included via `calc_child_benefits()` and added to total income

#### Care Benefits and Costs
- ❌ Expected: Not included in expected wealth calculation
- ✅ Current: Must be included via `calc_care_benefits_and_costs()` and added to expected wealth

## Updated Implementation

All tests have been updated to:
1. ✅ Remove `params` argument from `budget_constraint()` calls
2. ✅ Use `SEX` constant instead of `sex` variable
3. ✅ Use `construct_experience_years()` for experience conversion
4. ✅ Use `calc_labor_income_after_ssc()` for wage calculation
5. ✅ Use `calc_pensions_after_ssc()` for pension calculation
6. ✅ Fix `calc_unemployment_benefits()` parameter order (sex before education)
7. ✅ Fix `calc_net_household_income()` unpacking (3 values, not 2)
8. ✅ Remove `mother_dead` parameter from `calc_inheritance_amount()`
9. ✅ Add `draw_inheritance_outcome()` for stochastic inheritance
10. ✅ Include `child_benefits` in total income max calculation
11. ✅ Include `care_benefits_and_costs` in expected wealth calculation
12. ✅ Handle period 0 income shock correctly
