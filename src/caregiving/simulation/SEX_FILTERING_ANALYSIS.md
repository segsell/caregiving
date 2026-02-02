# Analysis: Where is filtering to `sex == SEX` applied?

This document identifies all locations where data is filtered to `sex == SEX` (where `SEX` from `shared.py` denotes women, value = 1) in `task_plot_model_fit_estimated_params.py` and all plotting functions it calls.

## Summary

**SEX constant value**: `SEX = 1` (women) from `caregiving.model.shared`

## Filtering Locations

### 1. In `task_plot_model_fit_estimated_params.py`

#### Line 241: Simulated data
```python
df_sim["sex"] = SEX
```
- **Action**: Sets sex for all simulated data to `SEX` (women)
- **Note**: This ensures simulated data is always for women

#### Line 652: Test function
```python
df_gender = df[df["sex"] == SEX].copy()
```
- **Location**: `test_choice_shares_sum_to_one` function
- **Action**: Filters both empirical and simulated data to women only
- **Purpose**: Ensures choice shares sum to 1 test is done only for women

### 2. In `plot_model_fit.py` plotting functions

#### `plot_wealth_by_age_and_education` (lines 31-122)
- **No explicit filtering**: Assumes data is already filtered
- **Note**: Data is filtered upstream in `create_df_wealth` (see below)

#### `plot_wealth_by_age_bins_and_education` (lines 124-250)
- **No explicit filtering**: Assumes data is already filtered
- **Note**: Data is filtered upstream in `create_df_wealth` (see below)

#### `plot_choice_shares_by_education` (lines 296-398)
- **Line 345**: `sex = SEX` (local variable)
- **Line 357**: `emp_edu = data_emp[(data_emp["sex"] == sex) & (data_emp["education"] == edu_var)]`
- **Action**: Filters empirical data to women only, by education
- **Note**: Simulated data is not filtered here (assumed already filtered)

#### `plot_choice_shares_overall` (lines 400-496)
- **Line 452**: `sex = SEX` (local variable)
- **Line 458**: `emp_all = data_emp[data_emp["sex"] == sex]`
- **Action**: Filters empirical data to women only
- **Note**: Simulated data is not filtered here (assumed already filtered)

#### `plot_choice_shares_by_education_age_bins` (lines 498-674)
- **Line 563**: `sex = SEX` (local variable)
- **Line 592**: `emp_sub = emp[(emp["sex"] == sex) & (emp["education"] == i)]`
- **Action**: Filters empirical data to women only, by education
- **Note**: Simulated data is not filtered here (assumed already filtered)

#### `plot_caregiver_shares_by_age` (lines 875-953)
- **Line 914**: `sex = SEX` (local variable)
- **Lines 916-918**:
  ```python
  if "sex" in df_emp:
      df_emp = df_emp.loc[df_emp["sex"] == sex].copy()
  if "sex" in df_sim:
      df_sim = df_sim.loc[df_sim["sex"] == sex].copy()
  ```
- **Action**: Filters both empirical and simulated data to women only (if sex column exists)

#### `plot_caregiver_shares_by_age_bins` (lines 954-1178)
- **No explicit filtering**: Assumes data is already filtered
- **Note**: Uses moments data, not raw microdata

#### `plot_simulated_care_demand_by_age` (lines 1272-1342)
- **Line 1291**: `df_sim = df_sim.loc[df_sim["sex"] == SEX].copy()`
- **Action**: Filters simulated data to women only (if sex column exists)

#### `plot_transitions_by_age` (lines 1665-1813)
- **Line 1697**: `sex = SEX` (local variable)
- **Lines 1734, 1737**:
  ```python
  emp_sub = emp[(emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)]
  sim_sub = sim[(sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)]
  ```
- **Action**: Filters both empirical and simulated data to women only, by education

#### `plot_transition_counts_by_age` (lines 1815-2034)
- **Line 1856**: `sex = SEX` (local variable)
- **Lines 1901, 1904**:
  ```python
  emp_sub = emp[(emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)]
  sim_sub = sim[(sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)]
  ```
- **Action**: Filters both empirical and simulated data to women only, by education

#### `plot_transitions_by_age_bins` (lines 2036-2228)
- **Line 2079**: `sex = SEX` (local variable)
- **Lines 2153, 2156**:
  ```python
  emp_sub = emp[(emp["sex"] == sex) & (emp["education"] == i) & (emp["age"] <= age_max)]
  sim_sub = sim[(sim["sex"] == sex) & (sim["education"] == i) & (sim["age"] <= age_max)]
  ```
- **Action**: Filters both empirical and simulated data to women only, by education

### 3. In `task_create_soep_moments.py` helper functions

#### `create_df_non_caregivers` (lines 2130-2167)
- **Line 2163**: `& (df_full["sex"] == 1)`
- **Action**: Filters to women only (hardcoded `1`, not using `SEX` constant)
- **Note**: This is used to create empirical data subsets

#### `create_df_with_caregivers` (lines 2169-2205)
- **Line 2202**: `& (df_full["sex"] == 1)`
- **Action**: Filters to women only (hardcoded `1`, not using `SEX` constant)
- **Note**: This is used to create empirical data subsets

#### `create_df_caregivers` (lines 2207-2242)
- **Line 2239**: `& (df_caregivers_full["sex"] == 1)`
- **Action**: Filters to women only (hardcoded `1`, not using `SEX` constant)
- **Note**: This is used to create empirical data subsets

#### `create_df_wealth` (lines 2397-2432)
- **No explicit filtering**: Relies on upstream filtering
- **Note**: Uses `load_and_scale_correct_data` which may filter elsewhere

## Key Observations

1. **Simulated data**: Always set to `sex = SEX` at line 241 in the task function, so all simulated data is for women.

2. **Empirical data**: Filtered in multiple places:
   - Upstream in `create_df_non_caregivers`, `create_df_with_caregivers`, `create_df_caregivers` (hardcoded `sex == 1`)
   - Within plotting functions when creating subsets by education

3. **Inconsistency**: Helper functions in `task_create_soep_moments.py` use hardcoded `sex == 1` instead of the `SEX` constant, which could lead to issues if `SEX` value changes.

4. **Pattern**: Most plotting functions filter empirical data by `sex == SEX` (or `sex == sex` where `sex = SEX`), but assume simulated data is already filtered.

## Recommendations

1. **Standardize**: Replace hardcoded `sex == 1` in `create_df_non_caregivers`, `create_df_with_caregivers`, and `create_df_caregivers` with `sex == SEX` for consistency.

2. **Documentation**: Add comments in plotting functions noting that simulated data is assumed to be pre-filtered to women only.

3. **Verification**: Consider adding assertions in plotting functions to verify simulated data has `sex == SEX` to catch any upstream issues.
