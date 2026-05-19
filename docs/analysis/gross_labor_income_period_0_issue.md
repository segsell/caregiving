# Issue: `gross_labor_income` is NaN in Period 0

## Summary

In period 0 (age 30, first period of simulation), `gross_labor_income` is NaN, even though:
- `lagged_choice` exists (from initial states)
- Consumption is correctly computed (not NaN)
- Labor income should be computable based on `lagged_choice`

## Root Cause

The issue is that `gross_labor_income` is computed in the budget constraint using `income_shock_previous_period`, but in period 0 there is no previous period, so `income_shock_previous_period` is undefined/NaN.

### 1. How `gross_labor_income` is Computed

From `src/caregiving/model/wealth_and_budget/budget_equation_no_care_demand.py` (line 83-90):
- The budget constraint calls `calc_labor_income_after_ssc` with `income_shock=income_shock_previous_period` (line 88)
- This goes into the wage equation (line 66-67 of `wages_no_care_demand.py`):
  ```python
  hourly_wage = jnp.exp(
      gamma_0 + gamma_1 * jnp.log(experience_years + 1) + income_shock
  )
  ```
- If `income_shock_previous_period` is NaN, then `hourly_wage` is NaN, and `gross_labor_income` is NaN

### 2. When the Budget Constraint is Called

From `dcegm/src/dcegm/simulation/sim_utils.py`:
- The budget constraint is called in `transition_to_next_period` (line 220-227)
- Income shocks are drawn for the **next period** (line 190-195)
- These shocks are passed as `income_shocks_of_period` to compute next period's assets
- The budget constraint is called with `income_shock_previous_period=income_shock_draw` (line 110 of `law_of_motion.py`)

### 3. The Problem in Period 0

For period 0:
- The budget constraint is called to compute period 1's `assets_begin_of_period`
- At this point, income shocks are drawn for period 1
- These shocks are passed as `income_shock_previous_period` to the budget constraint
- But `gross_labor_income` in the `budget_aux` dictionary is computed for the **current period** (period 0 in this case)
- Since there is no period -1, `income_shock_previous_period` for period 0's labor income doesn't exist
- However, the budget constraint uses the same `income_shock_previous_period` parameter for computing labor income

**The issue:** When computing `gross_labor_income` for period 0, the budget constraint uses `income_shock_previous_period` which is actually the shock for period 1 (not period 0). But more fundamentally, for period 0, there is no previous period's shock, so it should use 0 (the mean of the shock distribution).

### 4. Why Consumption Works

Consumption is computed correctly because:
- The budget constraint calculates `assets_begin_of_period` using `total_income`
- `total_income` aggregates multiple sources (partner income, unemployment benefits, retirement income, child benefits, etc.)
- Even if labor income is NaN in period 0, other income sources allow consumption to be computed
- The DCEGM algorithm correctly computes `assets_begin_of_period` using the full `total_income`

## Solution

The budget constraint should handle period 0 specially. For period 0, when computing `gross_labor_income`, it should use `income_shock_previous_period = 0.0` (the mean of the shock distribution, as specified in `specs.yaml` line 142: `income_shock_mean: 0.0`).

**Implementation:**
In the budget constraint, check if `period == 0` and use `model_specs["income_shock_mean"]` (0.0) instead of `income_shock_previous_period` for labor income calculation.

**Fix applied:**
```python
# For period 0, use mean income shock (0.0) since there's no previous period
income_shock_for_labor = jnp.where(
    period == 0,
    model_specs["income_shock_mean"],
    income_shock_previous_period,
)
```

This fix has been applied to all three budget equation files:
- `src/caregiving/model/wealth_and_budget/budget_equation_no_care_demand.py` (lines 82-90)
- `src/caregiving/model/wealth_and_budget/budget_equation.py` (lines 79-87)
- `src/caregiving/model/wealth_and_budget/budget_equation_caregiving_leave_with_job_retention.py` (lines 96-104)

## Files Involved

- `src/caregiving/model/wealth_and_budget/budget_equation_no_care_demand.py` (lines 82-90) ✅ Fixed
- `src/caregiving/model/wealth_and_budget/budget_equation.py` (lines 79-87) ✅ Fixed
- `src/caregiving/model/wealth_and_budget/budget_equation_caregiving_leave_with_job_retention.py` (lines 96-104) ✅ Fixed
- `src/caregiving/model/wealth_and_budget/wages_no_care_demand.py` (lines 44-87)
- `src/caregiving/specs.yaml` (line 142: `income_shock_mean: 0.0`)

## Notes

- `gross_labor_income` comes from the `budget_aux` dictionary returned by the budget constraint (line 184 of `budget_equation_no_care_demand.py`)
- The fix ensures that for period 0, `income_shock_for_labor = 0.0` (the mean), which prevents NaN in `gross_labor_income`
- All three models (baseline, no care demand, and caregiving leave) have been fixed
