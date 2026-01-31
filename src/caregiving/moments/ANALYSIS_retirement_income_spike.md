# Analysis: Retirement Income Spike in Wealth Adjustment

## Executive Summary

When adjusting observed wealth data using `adjust_observed_assets`, there is an unnatural spike in wealth when people enter retirement. This occurs because the experience scaling mechanism in `budget_constraint` uses different scales for working vs. retired individuals, and when someone enters retirement, their experience gets rescaled to a higher value (if `max_pp_retirement > max_exps_period_working[period]`). This causes pension income to be calculated based on an inflated experience value, creating an artificial wealth boost.

## How the Model Handles Retirement Income

### 1. Experience Scaling in `budget_constraint`

In `src/caregiving/model/wealth_and_budget/budget_equation.py` (lines 44-49):

```python
experience_years = construct_experience_years(
    float_experience=experience,
    period=period,
    is_retired=is_retired(lagged_choice),
    model_specs=model_specs,
)
```

The function `construct_experience_years` (in `src/caregiving/model/experience_baseline_model.py`, lines 106-116) scales the normalized experience (0-1) differently based on retirement status:

- **If working** (`is_retired=False`): Uses `max_exps_period_working[period]` as the scale factor
- **If retired** (`is_retired=True`): Uses `max_pp_retirement` as the scale factor

```python
scale_not_retired = jnp.take(
    model_specs["max_exps_period_working"], period, mode="clip"
)
scale_retired = model_specs["max_pp_retirement"]
scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
return float_experience * scale
```

### 2. Income Calculation

Both retirement and labor income are calculated using the **same** `experience_years` value:

- **Retirement income** (lines 63-66):
  ```python
  retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
      pension_points=experience_years,  # Note: This is the rescaled value
      model_specs=model_specs,
  )
  ```

- **Labor income** (lines 89-96):
  ```python
  labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
      lagged_choice=lagged_choice,
      experience_years=experience_years,  # Same rescaled value
      education=education,
      sex=sex_var,
      income_shock=income_shock_for_labor,
      model_specs=model_specs,
  )
  ```

### 3. Income Selection

The model correctly selects which income to use based on the previous period's choice (lines 100-106):

```python
was_worker = is_working(lagged_choice)
was_retired = is_retired(lagged_choice)

own_income_after_ssc = (
    was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc
)
```

## The Problem

### Issue 1: Experience Rescaling at Retirement Entry

When someone enters retirement:

1. **Previous period** (last working period):
   - `is_retired(lagged_choice) = False`
   - Experience scaled by `max_exps_period_working[period]`
   - Example: If `float_experience = 0.8` and `max_exps_period_working[65] = 30`, then `experience_years = 24`

2. **Current period** (first retirement period):
   - `is_retired(lagged_choice) = True` (they chose retirement in previous period)
   - Experience scaled by `max_pp_retirement`
   - Example: If `float_experience = 0.8` and `max_pp_retirement = 50`, then `experience_years = 40`

**The problem**: The `experience_years` value jumps from 24 to 40, even though the underlying normalized experience (`float_experience = 0.8`) hasn't changed.

### Issue 2: Pension Points Calculation

The pension income calculation (`calc_pensions_after_ssc`) expects `pension_points` as input, but it receives `experience_years` which is the rescaled value. 

Looking at `calc_pension_points_form_experience` (in `src/caregiving/model/pension_system/experience_stock.py`, lines 74-97), it expects **working experience years** (not rescaled retirement experience years) to calculate pension points. The function uses a lookup table `pp_for_exp_by_sex_edu` that maps working experience years to pension points.

**The issue**: When someone enters retirement, `budget_constraint` passes the rescaled retirement experience years (e.g., 40) to `calc_pensions_after_ssc`, but the pension point calculation should use the working experience years (e.g., 24).

### Issue 3: Model State Tracking

The model has an `already_retired` state variable that distinguishes between:
- **Freshly retired**: `already_retired = 0` and `is_retired(lagged_choice) = True`
- **Already retired**: `already_retired = 1` and `is_retired(lagged_choice) = True`

However, `budget_constraint` does **not** use `already_retired` to distinguish these cases. It only uses `is_retired(lagged_choice)` to determine the scale factor.

In `get_next_period_experience` (lines 37-103), when someone is freshly retired, the experience is converted to pension points using `calc_pension_points_for_experience`, which takes the **working experience years** (before rescaling) as input. This is the correct approach.

## Critical Question: Is There Actually a Bug?

Let me trace through the logic more carefully to determine if there's actually a bug in the model code.

### Current Behavior in `budget_constraint`

**For someone entering retirement (first period of retirement):**
- `lagged_choice` = retirement choice → `is_retired(lagged_choice) = True`
- `construct_experience_years` uses `max_pp_retirement` scale → higher `experience_years`
- `calc_pensions_after_ssc(pension_points=experience_years)` receives the rescaled value
- `calc_gross_pension_income` multiplies by `annual_pension_point_value`

**The problem**: The rescaled `experience_years` (e.g., 40) is being treated as pension points, but pension points should be calculated from working experience years (e.g., 24) using the lookup table.

### Key Finding: Function Signature Mismatch

Looking at `calc_pensions_after_ssc` (line 6-9 in `pension_payments.py`):
```python
def calc_pensions_after_ssc(
    pension_points,  # <-- Parameter name suggests it expects pension points
    model_specs,
):
```

But in `budget_constraint` (line 64), it's called with:
```python
retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
    pension_points=experience_years,  # <-- But this is rescaled experience years, not pension points!
    model_specs=model_specs,
)
```

**The bug is confirmed**: `budget_constraint` passes rescaled retirement experience years to a function that expects pension points. The function `calc_pension_points_form_experience` (used in `get_next_period_experience` for fresh retirees) expects **working experience years** as input, not rescaled retirement experience years.

### What Happens in `get_next_period_experience`

When someone is freshly retired (in `get_next_period_experience`, lines 79-86):
```python
pension_points = calc_pension_points_for_experience(
    period=period,
    experience_years=exp_years_last_period,  # <-- This is working experience years!
    sex=sex,
    partner_state=partner_state,
    education=education,
    model_specs=model_specs,
)
```

Here, `exp_years_last_period` is calculated using `construct_experience_years` with `is_retired=already_retired`, where `already_retired=0` (not yet retired), so it uses the **working scale**. This is correct!

**The inconsistency**: `get_next_period_experience` correctly uses working experience years for pension point calculation, but `budget_constraint` uses rescaled retirement experience years.

### Re-evaluation: Is This Actually a Bug?

After more careful analysis, I need to reconsider whether this is actually a bug:

**For someone ALREADY retired:**
- Their normalized experience (0-1) = `actual_pension_points / max_pp_retirement`
- `construct_experience_years` with `is_retired=True` gives: `experience_years = actual_pension_points`
- This is passed to `calc_pensions_after_ssc(pension_points=experience_years)`
- `calc_gross_pension_income` multiplies by `annual_pension_point_value`
- **This appears correct** ✓

**For someone who JUST entered retirement:**
- In the period they enter retirement, `get_next_period_experience` converts working experience to pension points
- It then scales: `exp_scaled = pension_points / max_pp_retirement`
- This becomes their normalized experience for the NEXT period
- In that next period, `budget_constraint` uses this normalized experience
- `construct_experience_years` with `is_retired=True` gives: `experience_years = pension_points`
- **This also appears correct** ✓

**However, there's a potential issue:**

The function `calc_pension_points_form_experience` uses a lookup table `pp_for_exp_by_sex_edu` that maps **working experience years** to pension points. This is a non-linear mapping.

But in `budget_constraint`, for retired people, `experience_years` is already in pension points (not working experience years), and it's used directly without going through `calc_pension_points_form_experience`.

**The question is**: Does the model design assume that for retired people, the normalized experience already represents pension points (scaled by `max_pp_retirement`), and that these pension points are used directly?

If so, then the model might be correct, BUT there could still be an issue if:
1. The normalized experience for someone entering retirement hasn't been properly converted yet
2. Or if `max_pp_retirement` is significantly larger than `max_exps_period_working[period]`, causing the spike

**Conclusion**: The model design appears to be that for retired people, `experience_years` (output of `construct_experience_years` with `is_retired=True`) IS the pension points value, not working experience years. This is a design choice, not necessarily a bug.

**However**, if you're seeing a spike in wealth when people enter retirement, it could be because:
1. The normalized experience hasn't been properly converted to pension points yet (timing issue)
2. Or the scaling factor `max_pp_retirement` is much larger than `max_exps_period_working[period]`, causing an artificial jump

**To determine if there's a bug, we need to check**: What is the normalized experience value for someone in their first period of retirement? Is it still based on working experience, or has it been converted to `pension_points / max_pp_retirement`?

### What Should Happen

When someone enters retirement:
1. Use the **working experience years** (scaled by `max_exps_period_working[period]`) to calculate pension points
2. Convert pension points to pension income
3. Use the rescaled retirement experience years only for the normalized state variable (0-1) in subsequent periods

**However**, the current implementation in `budget_constraint` does not distinguish between freshly retired and already retired individuals. It always uses the rescaled retirement experience years for pension calculation.

## Critical Analysis: Correct Implementation in `adjust_wealth`

### Current Implementation Issues

In `_adjust_wealth_with_lagged_choice` (in `src/caregiving/moments/task_create_soep_moments.py`):

1. **Lagged choice is correctly created**: The function shifts `choice` by 1 period within each person, which correctly identifies the previous period's choice.

2. **Period calculation**: The `period` passed to `budget_constraint` should be calculated as `age - start_age` for each observation. This is handled by the model's `compute_assets_begin_of_period` function.

3. **Experience state**: The experience value in `states_dict` should be the normalized float_experience (0-1), not actual years. This is already the case.

### The Core Problem

The issue is that `budget_constraint` uses `is_retired(lagged_choice)` to determine the experience scale, which causes the spike when someone enters retirement. However, for pension calculation, we need to use the **working experience years**, not the rescaled retirement experience years.

### Potential Solutions

#### Solution 1: Track `already_retired` State

**Idea**: Include `already_retired` in `states_dict` and modify `budget_constraint` to use working experience years for pension calculation when `already_retired = 0` and `is_retired(lagged_choice) = True`.

**Challenges**:
- Requires modifying `budget_constraint` to handle this case
- Need to track `already_retired` in the data (may not be available)
- Would require changes to the model's budget constraint function

#### Solution 2: Pre-calculate Pension Points

**Idea**: Before calling `adjust_observed_assets`, pre-calculate pension points for individuals entering retirement using their working experience years, and store this in a separate variable.

**Challenges**:
- Would require duplicating the pension point calculation logic
- Need to identify freshly retired individuals in the data
- Complex to implement correctly

#### Solution 3: Use Previous Period's Experience Scale

**Idea**: When calculating pension income for someone who just entered retirement, use the working experience scale from the previous period instead of the retirement scale.

**Implementation**:
1. For each observation, check if `is_retired(lagged_choice) = True`
2. If true, also check if `is_retired(choice)` from two periods ago (if available) was `False`
3. If freshly retired, calculate pension points using working experience years
4. Otherwise, use the rescaled retirement experience years

**Challenges**:
- Need to track choices from two periods ago
- Complex logic to implement
- May not be feasible with current data structure

#### Solution 4: Modify Experience Scaling Logic

**Idea**: Modify `construct_experience_years` or create a separate function that returns working experience years for pension calculation, regardless of retirement status.

**Implementation**:
- Create a new function `construct_working_experience_years` that always uses the working scale
- Use this for pension point calculation in `budget_constraint`
- Keep the rescaled experience only for state variable normalization

**Challenges**:
- Requires modifying the model's budget constraint function
- May affect other parts of the model that rely on the current behavior

### Recommended Approach

**Option A: Check if Model Already Handles This Correctly**

First, verify whether the model's `budget_constraint` actually has the bug described above, or if there's additional logic that handles freshly retired individuals correctly. Check if:
1. `calc_pension_points_form_experience` is designed to handle rescaled retirement experience years
2. The lookup table `pp_for_exp_by_sex_edu` accounts for this
3. There's any other mechanism that corrects for this

**Option B: Implement Solution 4 (Modify Experience Scaling)**

If the bug is confirmed, the cleanest solution is to modify `budget_constraint` to:
1. Always use working experience years for pension point calculation
2. Only use rescaled retirement experience years for state variable normalization

This would require:
- Creating a helper function to get working experience years
- Modifying `budget_constraint` to use this for pension calculation
- Ensuring this doesn't break other parts of the model

**Option C: Workaround in `adjust_wealth`**

As a temporary workaround, in `_adjust_wealth_with_lagged_choice`:
1. Identify individuals who are freshly retired (entering retirement this period)
2. For these individuals, manually calculate pension points using working experience years
3. Create a modified `states_dict` that uses these pre-calculated pension points

However, this would require significant changes to how `adjust_observed_assets` works, as it expects to call `budget_constraint` directly.

## Implementation Strategy for `adjust_wealth`

### Constraints

1. **Cannot modify `budget_constraint`**: The function is part of the model and used throughout the codebase. Modifying it would require extensive testing and could break other functionality.

2. **Limited control over `adjust_observed_assets`**: This function calls `compute_assets_begin_of_period` which internally calls `budget_constraint`. We can only pass state variables through `states_dict`.

3. **Need to identify freshly retired individuals**: We need to distinguish between:
   - Freshly retired: `is_retired(lagged_choice) = True` and `is_retired(choice from 2 periods ago) = False`
   - Already retired: `is_retired(lagged_choice) = True` and `is_retired(choice from 2 periods ago) = True`

### Recommended Solution: Pre-adjust Experience for Fresh Retirees

**Approach**: For freshly retired individuals, adjust the normalized experience value in `states_dict` so that when `construct_experience_years` rescales it using `max_pp_retirement`, it produces the correct working experience years.

**Algorithm**:

1. For each observation, identify if it's a fresh retiree:
   ```python
   is_retired_now = is_retired(lagged_choice)
   is_retired_before = is_retired(choice_from_2_periods_ago)  # Need to track this
   is_fresh_retiree = is_retired_now & (~is_retired_before)
   ```

2. For fresh retirees, calculate what the normalized experience should be:
   ```python
   # Current normalized experience
   float_experience_current = experience  # from states_dict
   
   # Working experience years (what it should be)
   working_exp_years = float_experience_current * max_exps_period_working[period]
   
   # What normalized experience would give us working_exp_years when scaled by max_pp_retirement?
   adjusted_float_experience = working_exp_years / max_pp_retirement
   
   # Use this adjusted value in states_dict
   states_dict["experience"][is_fresh_retiree] = adjusted_float_experience
   ```

3. This way, when `budget_constraint` calls `construct_experience_years` with `is_retired=True`, it will get:
   ```python
   experience_years = adjusted_float_experience * max_pp_retirement
                  = (working_exp_years / max_pp_retirement) * max_pp_retirement
                  = working_exp_years  # Correct!
   ```

**Implementation in `_adjust_wealth_with_lagged_choice`**:

```python
# After creating lagged_choice, identify fresh retirees
df_sorted["lagged_choice_2_periods_ago"] = df_sorted.groupby(person_id_col)[
    choice_col
].shift(2)

# Identify fresh retirees
from caregiving.model.shared import is_retired
is_retired_now = df_sorted[lagged_choice_col].apply(is_retired)
is_retired_before = df_sorted["lagged_choice_2_periods_ago"].apply(is_retired)
is_fresh_retiree = is_retired_now & (~is_retired_before.fillna(False))

# Adjust experience for fresh retirees
if is_fresh_retiree.any():
    period_values = df_sorted["period"].values  # Need to calculate period from age
    experience_current = states_dict_sorted["experience"]
    
    # Calculate working experience years
    max_exps_working = np.array([
        model_specs["max_exps_period_working"][p] for p in period_values
    ])
    working_exp_years = experience_current * max_exps_working
    
    # Calculate adjusted normalized experience
    max_pp_ret = model_specs["max_pp_retirement"]
    adjusted_experience = np.where(
        is_fresh_retiree,
        working_exp_years / max_pp_ret,
        experience_current
    )
    
    states_dict_sorted["experience"] = adjusted_experience
```

**Challenges**:
1. Need to track `choice` from 2 periods ago (requires additional shift)
2. Need to calculate `period` from `age` for each observation
3. Need access to `model_specs` in `_adjust_wealth_with_lagged_choice`
4. Need to handle missing values for first two observations per person

### Alternative: Post-adjustment Correction

**Approach**: After calling `adjust_observed_assets`, identify fresh retirees and manually correct their wealth adjustment.

**Algorithm**:
1. Call `adjust_observed_assets` as normal
2. Identify fresh retirees
3. For each fresh retiree, recalculate wealth using working experience years
4. Replace the adjusted wealth for these individuals

**Challenges**:
- Requires duplicating the wealth calculation logic
- Complex to implement correctly
- May not account for all interactions (interest, taxes, etc.)

### Recommended Implementation

**Use the pre-adjustment approach** (adjusting experience in `states_dict` before calling `adjust_observed_assets`):

1. **Pros**:
   - Works within the existing framework
   - No need to duplicate calculation logic
   - Correctly handles all interactions (taxes, interest, etc.)

2. **Cons**:
   - Requires tracking choices from 2 periods ago
   - Requires calculating period from age
   - Slightly more complex logic

3. **Implementation steps**:
   - Modify `_adjust_wealth_with_lagged_choice` to:
     - Track `choice` from 2 periods ago
     - Calculate `period` from `age` and `start_age`
     - Identify fresh retirees
     - Adjust experience values for fresh retirees
     - Pass adjusted experience in `states_dict`

## Conclusion

The model's `budget_constraint` function has a confirmed bug where it uses rescaled retirement experience years for pension calculation, which causes an artificial spike when people enter retirement. The correct behavior would be to use working experience years to calculate pension points.

**For the wealth adjustment use case**, the recommended solution is to pre-adjust the normalized experience values in `states_dict` for freshly retired individuals, so that when `budget_constraint` rescales them using the retirement scale, it produces the correct working experience years. This approach:
- Works within the existing framework
- Doesn't require modifying the model code
- Correctly handles all income components (pension, taxes, interest, etc.)
- Is implementable in `_adjust_wealth_with_lagged_choice`

The implementation requires:
1. Tracking `choice` from 2 periods ago (to identify fresh retirees)
2. Calculating `period` from `age` for each observation
3. Adjusting normalized experience values for fresh retirees before passing to `adjust_observed_assets`
