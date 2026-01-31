# Debug Analysis: Wealth Spike at Retirement in `adjust_observed_assets`

## Problem Statement

When using `adjust_observed_assets` in `create_df_wealth`, there is an erroneous spike in wealth around retirement age. This document systematically traces through the code to identify the root cause.

## Step 1: Which Experience is Used?

### In `create_df_wealth` (line 2145):
```python
states_dict["experience"] = df_wealth["experience"].values
```

**Question**: What does `df_wealth["experience"]` contain?

**Answer**: The experience value from the data. This should be the **normalized experience** (0-1 scale), as this is how experience is stored in the model's state space.

### In `budget_constraint` (line 44-49):
```python
experience_years = construct_experience_years(
    float_experience=experience,  # <-- This is the normalized 0-1 value
    period=period,
    is_retired=is_retired(lagged_choice),
    model_specs=model_specs,
)
```

**Key Finding**: The experience passed to `budget_constraint` is the **normalized float_experience (0-1)**, not actual years.

## Step 2: Is Rescaling Happening at Retirement?

### In `construct_experience_years` (experience_baseline_model.py, lines 106-116):
```python
def construct_experience_years(float_experience, period, is_retired, model_specs):
    scale_not_retired = jnp.take(
        model_specs["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = model_specs["max_pp_retirement"]
    scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
    return float_experience * scale
```

**Key Finding**:
- When `is_retired=False`: Uses `max_exps_period_working[period]` as scale
- When `is_retired=True`: Uses `max_pp_retirement` as scale

**If `max_pp_retirement > max_exps_period_working[period]`**, then when someone enters retirement:
- Previous period: `float_experience * max_exps_period_working[period]` = working experience years
- Current period: `float_experience * max_pp_retirement` = **higher value** (if scales differ)

**This is where the rescaling happens!**

## Step 3: Are Fresh Retired Individuals Detected Correctly?

### In `_adjust_wealth_with_lagged_choice` (lines 2243-2262):
```python
# Create lagged_choice within each person
df_sorted[lagged_choice_col] = df_sorted.groupby(person_id_col)[
    choice_col
].shift(1)
```

**Analysis**:
- `lagged_choice` is created by shifting `choice` by 1 period within each person
- This correctly identifies the previous period's choice
- `is_retired(lagged_choice)` will be `True` for someone who chose retirement in the previous period

**However**: There's no distinction between:
- **Freshly retired**: Just entered retirement (first period of retirement)
- **Already retired**: Been retired for multiple periods

**Key Issue**: The experience value in the data might still be on the **working scale** for someone who just entered retirement, but `construct_experience_years` will rescale it to the **retirement scale**.

## Step 4: Is Pension Income or Labor Income Computed?

### In `budget_constraint` (lines 63-66, 89-96, 104-106):
```python
# Both are calculated using the SAME experience_years value
retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
    pension_points=experience_years,  # <-- Rescaled value
    model_specs=model_specs,
)

labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
    lagged_choice=lagged_choice,
    experience_years=experience_years,  # <-- Same rescaled value
    ...
)

# Then selection happens
was_worker = is_working(lagged_choice)
was_retired = is_retired(lagged_choice)

own_income_after_ssc = (
    was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc
)
```

**Key Finding**:
- Both labor and pension income use the **same** `experience_years` value
- The selection (`was_worker` vs `was_retired`) determines which income is used
- For retired people, `experience_years` is rescaled by `max_pp_retirement`
- This rescaled value is passed to `calc_pensions_after_ssc` as `pension_points`

**Potential Issue**: If `max_pp_retirement > max_exps_period_working[period]`, then:
- `experience_years` jumps up when someone enters retirement
- This higher value is used as `pension_points` in `calc_pensions_after_ssc`
- But `calc_pensions_after_ssc` expects pension points calculated from **working experience years**, not rescaled retirement experience years

## Step 5: When and Where is Experience Scale Rescaled?

### In the Model's State Transition (`get_next_period_experience`, lines 37-103):

**For someone entering retirement**:
1. `fresh_retired = (already_retired == 0) & retired_this_period`
2. `exp_years_last_period = construct_experience_years(..., is_retired=already_retired)`
   - Since `already_retired=0`, this uses **working scale**
3. `pension_points = calc_pension_points_for_experience(experience_years=exp_years_last_period, ...)`
   - Uses **working experience years** to calculate pension points ✓
4. `exp_years_this_period = pension_points` (for fresh retirees)
5. `exp_scaled = scale_experience_years(experience_years=pension_points, period=period, is_retired=True, ...)`
   - Scales pension points back to 0-1 using **retirement scale** ✓

**Result**: For someone who just entered retirement, their normalized experience becomes `pension_points / max_pp_retirement`.

### In `budget_constraint` (called by `adjust_observed_assets`):

**For someone in their first period of retirement**:
1. `lagged_choice` = retirement choice (they chose retirement in previous period)
2. `is_retired(lagged_choice) = True`
3. `experience` = normalized experience from data
4. **Question**: What is this normalized experience value?
   - If it's from the **previous period** (before retirement), it's still on the working scale
   - If it's from the **current period** (after retirement), it should be `pension_points / max_pp_retirement`

5. `construct_experience_years(..., is_retired=True)` uses `max_pp_retirement` scale
6. If the normalized experience is still on working scale, then:
   - `experience_years = (working_scale_experience / max_exps_working) * max_pp_retirement`
   - This gives a **higher value** than it should be!

## Root Cause Hypothesis

**The Problem**: When someone enters retirement, their normalized experience in the data might still be based on the **working scale**, but `budget_constraint` rescales it using the **retirement scale**.

**Example**:
- Person has 24 years of working experience
- Normalized experience = 24 / 30 = 0.8 (using working scale)
- They enter retirement
- In `budget_constraint`, `construct_experience_years` with `is_retired=True` gives:
  - `experience_years = 0.8 * max_pp_retirement = 0.8 * 50 = 40`
- But the correct pension points should be calculated from 24 years, not 40!

**The Fix**: For freshly retired individuals, we need to:
1. Identify them (first period of retirement)
2. Use their **working experience years** (before rescaling) to calculate pension points
3. Or adjust the normalized experience to account for the scale difference

## Verification Steps Needed

1. **Check what the experience value is in the data** for someone who just entered retirement:
   - Is it still on the working scale (0-1 normalized by working max)?
   - Or has it been converted to pension points scale (0-1 normalized by retirement max)?

2. **Check if `already_retired` state is available** in the data:
   - If yes, we can identify fresh retirees
   - If no, we need to infer from `lagged_choice` and previous periods

3. **Check the scale factors**:
   - What is `max_exps_period_working[65]` (typical retirement age)?
   - What is `max_pp_retirement`?
   - Is `max_pp_retirement > max_exps_period_working[65]`?

4. **Trace through a specific example**:
   - Person at age 64: working, experience = 0.8 (normalized by working scale)
   - Person at age 65: retired, experience = ? (what is this value?)
   - What does `construct_experience_years` return in each case?

## Recommended Debugging Code

Add diagnostic output to `_adjust_wealth_with_lagged_choice` to check:

```python
# After creating lagged_choice, add diagnostic checks
df_sorted["is_retired_now"] = df_sorted[lagged_choice_col].isin(retirement_values)
df_sorted["is_retired_prev"] = df_sorted.groupby(person_id_col)[lagged_choice_col].shift(1).isin(retirement_values)
df_sorted["is_fresh_retiree"] = df_sorted["is_retired_now"] & (~df_sorted["is_retired_prev"].fillna(False))

# Check experience values for fresh retirees
if df_sorted["is_fresh_retiree"].any():
    fresh_retirees = df_sorted[df_sorted["is_fresh_retiree"]]
    print(f"Fresh retirees: {len(fresh_retirees)}")
    print(f"Experience range: {fresh_retirees['experience'].min()} to {fresh_retirees['experience'].max()}")
    print(f"Age range: {fresh_retirees['age'].min()} to {fresh_retirees['age'].max()}")

    # Calculate what experience_years would be
    period_values = fresh_retirees["age"].values - specs["start_age"]
    max_exps_working = np.array([specs["max_exps_period_working"][p] for p in period_values])
    max_pp_ret = specs["max_pp_retirement"]

    working_exp_years = fresh_retirees["experience"].values * max_exps_working
    retirement_exp_years = fresh_retirees["experience"].values * max_pp_ret

    print(f"Working exp years (if still on working scale): {working_exp_years}")
    print(f"Retirement exp years (if rescaled): {retirement_exp_years}")
    print(f"Difference: {retirement_exp_years - working_exp_years}")
```
