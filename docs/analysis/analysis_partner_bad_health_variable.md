# Analysis: Creating a Partner Bad Health Variable in SOEP Data Pipeline

## Executive Summary

This document provides a systematic analysis of how to create a `partner_bad_health` variable in the SOEP data pipeline. The variable will be used later to drop observations where a partner is in bad health, but is not implemented yet.

---

## 1. Data Pipeline Overview

### 1.1 Main Data Flow

The SOEP data processing follows this pipeline:

```
task_load_and_merge_estimation_sample (merge module)
  ↓
  Creates: soep_estimation_data_raw.csv
  ↓
task_create_caregivers_sample (structural estimation sample module)
  ↓
  Creates: soep_structural_caregivers_sample.csv
```

### 1.2 Key Files

- **Merge Module**: `src/caregiving/data_management/soep/merge_data/task_load_and_merge_structural_samples.py`
  - Function: `task_load_and_merge_estimation_sample` (lines 45-193)
  - Merges multiple SOEP modules: pgen, ppathl, pl, hl, pequiv, bioparen, biobirth

- **Variable Creation Module**: `src/caregiving/data_management/soep/task_create_structural_estimation_sample.py`
  - Function: `task_create_caregivers_sample` (lines 200-367)
  - Processes raw merged data and creates derived variables

- **Variable Definitions**: `src/caregiving/data_management/soep/variables.py`
  - Contains helper functions for creating variables

---

## 2. Current Partner Information Handling

### 2.1 Partner Identifier: `parid`

- **Source**: Loaded from `ppathl` module in `task_load_and_merge_estimation_sample` (line 83)
- **Definition**: Partner's person identifier (PID)
- **Value Coding**:
  - Positive values: Valid partner PID
  - Negative values (< 0): No partner

### 2.2 Partner Merging Process

**Function**: `merge_couples()` in `variables.py` (lines 405-428)

**Process**:
1. Resets index to get `pid` and `syear` as columns
2. Creates `df_partners` copy of the dataframe
3. Sets negative `parid` values to `np.nan` for merging
4. Merges using:
   - `left_on=["hid", "syear", "parid"]`
   - `right_on=["hid", "syear", "pid"]`
   - `suffixes=("", "_p")`
5. Partner variables get `_p` suffix automatically

**Key Point**: After `merge_couples()`, all partner variables are available with `_p` suffix.

### 2.3 Current Partner Variables

**From `merge_couples()`**: Any column in the original dataframe gets a partner version with `_p` suffix.

**Currently Used**:
- `work_status_p`: Partner's working status (used in `create_partner_state`)
- Variables referenced in `task_create_event_study_sample` documentation include:
  - `age_p`, `sex_p`, `choice_p`, `education_p`, `health_p`, `working_hours_p`, etc.

### 2.4 Partner State Creation

**Function**: `create_partner_state()` in `variables.py` (lines 376-402)

**Process**:
1. Calls `create_working_status(df)` to create `work_status_p`
2. Calls `merge_couples(df)` to merge partner information
3. Creates `partner_state` variable:
   - `0`: No partner (`parid < 0`)
   - `1`: Working-age partner (`work_status_p == 1`)
   - `2`: Retired partner (`work_status_p == 0`)

**Location in Pipeline**: Called in `task_create_caregivers_sample` at line 217

---

## 3. Health Variable Structure

### 3.1 Health Data Source

**Module**: `pequiv` (SOEP-IS equivalent variables)

**Raw Variables Loaded** (in `task_load_and_merge_estimation_sample`, line 146):
- `m11126`: Self-Rated Health Status
- `m11124`: Disability Status of Individual

### 3.2 Health Variable Creation

**Function**: `create_health_var_good_bad()` in `variables.py` (lines 538-569)

**Raw Variable Coding**:

**m11126** (Self-Rated Health Status):
- `1`: Very good
- `2`: Good
- `3`: Satisfactory
- `4`: Bad
- `5`: Very bad
- `< 0`: Missing/not applicable (converted to `np.nan`)

**m11124** (Disability Status):
- `0`: No disability
- `1`: Has disability
- `< 0`: Missing (converted to `np.nan`)

**Derived Health Variable Logic**:
```python
# Bad health = 0
health = 0 if (m11126 in [4, 5]) OR (m11124 == 1)

# Good health = 1
health = 1 if (m11126 in [1, 2, 3]) AND (m11124 == 0)

# Otherwise: np.nan (missing)
```

### 3.3 Health Variable Usage in Pipeline

**Location**: `task_create_caregivers_sample`, line 271
```python
df = create_health_var_good_bad(df, drop_missing=False)
```

**Key Point**: Health is created AFTER `create_partner_state()` is called (line 217), which means partner health information should already be available as `health_p` via `merge_couples()`.

---

## 4. Creating Partner Bad Health Variable

### 4.1 Current Situation

After `merge_couples()` is called (inside `create_partner_state`), the dataframe contains:
- `m11126_p`: Partner's self-rated health status
- `m11124_p`: Partner's disability status
- `health_p`: Partner's health variable (if `create_health_var_good_bad` is applied to the merged data)

**However**: `create_health_var_good_bad` is currently called AFTER `create_partner_state`, which means `health_p` may not exist yet when partner merging happens.

### 4.2 Proposed Solution: Create Partner Health Variable

**Option 1: Create `partner_bad_health` from raw variables (`m11126_p`, `m11124_p`)**

**Implementation Location**: After `create_partner_state()` in `task_create_caregivers_sample`

**Logic**:
```python
def create_partner_bad_health(df, drop_missing=False):
    """
    Create partner_bad_health indicator variable.

    1 = partner is in bad health
    0 = partner is in good health or no partner
    np.nan = partner exists but health information is missing
    """
    # Initialize: default to 0 (no partner or partner in good health)
    df["partner_bad_health"] = 0

    # Case 1: No partner (parid < 0) → keep 0
    # Already handled by initialization

    # Case 2: Partner exists but has bad health
    # Bad health = (m11126_p in [4, 5]) OR (m11124_p == 1)
    mask_bad_health = (
        df["m11126_p"].isin([4, 5]) | (df["m11124_p"] == 1)
    )
    df.loc[mask_bad_health, "partner_bad_health"] = 1

    # Case 3: Partner exists, health information missing → set to np.nan
    # Only for observations with partner (parid >= 0)
    has_partner = df["parid"] >= 0
    health_missing = (
        (df["m11126_p"].isna()) | (df["m11126_p"] < 0) |
        (df["m11124_p"].isna()) | (df["m11124_p"] < 0)
    )
    mask_missing = has_partner & health_missing & ~mask_bad_health
    df.loc[mask_missing, "partner_bad_health"] = np.nan

    if drop_missing:
        initial_count = len(df)
        df = df[df["partner_bad_health"].notna()]
        print(f"{len(df)} observations left after dropping missing partner health.")

    return df
```

**Option 2: Create from `health_p` variable (if it exists)**

If `health_p` is already created by applying `create_health_var_good_bad` before `merge_couples()`, then:

```python
def create_partner_bad_health_from_health_p(df):
    """
    Create partner_bad_health from health_p variable.

    1 = partner is in bad health (health_p == 0)
    0 = partner is in good health (health_p == 1) or no partner
    np.nan = partner exists but health_p is missing
    """
    df["partner_bad_health"] = 0
    df.loc[df["health_p"] == 0, "partner_bad_health"] = 1
    df.loc[df["health_p"].isna() & (df["parid"] >= 0), "partner_bad_health"] = np.nan
    return df
```

### 4.3 Recommended Approach

**Recommended**: Option 1 (create from raw variables)

**Reasons**:
1. More explicit control over the logic
2. Does not depend on order of operations (whether `create_health_var_good_bad` is called before or after `merge_couples`)
3. Matches the logic in `create_health_var_good_bad` exactly
4. Easier to debug and verify

---

## 5. Implementation Location

### 5.1 Where to Add the Function

**File**: `src/caregiving/data_management/soep/variables.py`

**Location**: After `create_health_var_good_bad()` function (around line 570)

**Reason**: Groups health-related functions together.

### 5.2 Where to Call the Function

**File**: `src/caregiving/data_management/soep/task_create_structural_estimation_sample.py`

**Location**: In `task_create_caregivers_sample()`, after line 271 (after `create_health_var_good_bad`)

**Suggested Order**:
```python
df = create_partner_state(df, filter_missing=True)  # Line 217: merges partner data
# ... other variable creation ...
df = create_health_var_good_bad(df, drop_missing=False)  # Line 271: creates health
df = create_partner_bad_health(df, drop_missing=False)  # NEW: creates partner_bad_health
df = create_nursing_home(df)  # Line 272
```

**Important**: Must be called AFTER `create_partner_state()` because that's when `merge_couples()` brings in partner variables with `_p` suffix.

### 5.3 Data Type in `type_dict`

**File**: `task_create_structural_estimation_sample.py`, lines 296-330

**Add to `type_dict`**:
```python
"partner_bad_health": "float32",  # can be NA
```

---

## 6. Verification and Testing

### 6.1 Expected Values

- `partner_bad_health == 0`:
  - No partner (`parid < 0`)
  - Partner exists and is in good health (`m11126_p in [1,2,3]` AND `m11124_p == 0`)

- `partner_bad_health == 1`:
  - Partner exists and is in bad health (`m11126_p in [4,5]` OR `m11124_p == 1`)

- `partner_bad_health == np.nan`:
  - Partner exists (`parid >= 0`) but health information is missing

### 6.2 Cross-Checks

1. **Consistency with `partner_state`**:
   - If `partner_state == 0` (no partner), then `partner_bad_health` should be `0`
   - If `partner_state in [1, 2]` (has partner), then `partner_bad_health` can be `0`, `1`, or `np.nan`

2. **Consistency with own health logic**:
   - The bad health definition should match `create_health_var_good_bad`:
     - Bad: `(m11126 in [4,5]) OR (m11124 == 1)`
     - Good: `(m11126 in [1,2,3]) AND (m11124 == 0)`

3. **Missing data handling**:
   - Count of `partner_bad_health == np.nan` should be <= count of `partner_state in [1,2]`

---

## 7. Future Use: Dropping Observations

### 7.1 Filtering Logic (Not Implemented Yet)

When ready to drop observations with partner in bad health:

```python
# In task_create_caregivers_sample or later filtering step
initial_count = len(df)
df = df[df["partner_bad_health"] != 1]  # Keep only partner_bad_health == 0 or np.nan
# OR
df = df[(df["partner_bad_health"] == 0) | (df["partner_bad_health"].isna())]
dropped_count = initial_count - len(df)
print(f"Dropped {dropped_count} observations with partner in bad health.")
```

### 7.2 Considerations

- **Missing values**: Decide whether to drop observations where `partner_bad_health == np.nan`
  - Option A: Drop (conservative, ensures all remaining have valid partner health info)
  - Option B: Keep (maximizes sample size, but includes unknown partner health)

- **Timing**: Filter can be applied:
  - Immediately after creating the variable
  - At the end of the processing pipeline
  - In a separate filtering step

---

## 8. Dependencies and Order of Operations

### 8.1 Critical Dependencies

```
create_partner_state()
  → merge_couples()  [brings in m11126_p, m11124_p]
  → create_health_var_good_bad()  [creates health]
  → create_partner_bad_health()  [NEW: uses m11126_p, m11124_p]
```

### 8.2 Data Availability Check

**Before creating `partner_bad_health`**, verify that these columns exist:
- `parid`: Partner identifier (from ppathl)
- `m11126_p`: Partner's self-rated health (from pequiv, merged via `merge_couples`)
- `m11124_p`: Partner's disability status (from pequiv, merged via `merge_couples`)

**Verification Code**:
```python
required_cols = ["parid", "m11126_p", "m11124_p"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for partner health: {missing_cols}")
```

---

## 9. Summary and Next Steps

### 9.1 Summary

1. Partner information is merged via `merge_couples()` inside `create_partner_state()`
2. Raw health variables (`m11126`, `m11124`) from `pequiv` are loaded in the merge module
3. After `merge_couples()`, partner variables have `_p` suffix (`m11126_p`, `m11124_p`)
4. A `partner_bad_health` variable can be created using the same logic as `create_health_var_good_bad`, applied to `m11126_p` and `m11124_p`

### 9.2 Implementation Steps (When Ready)

1. **Create function** `create_partner_bad_health()` in `variables.py`
   - Use Option 1 approach (from raw variables)
   - Add to health-related section

2. **Import function** in `task_create_structural_estimation_sample.py`

3. **Call function** in `task_create_caregivers_sample()` after line 271

4. **Add to `type_dict`**: `"partner_bad_health": "float32"`

5. **Test**:
   - Verify variable exists and has expected values
   - Check consistency with `partner_state`
   - Verify missing value handling

6. **Future filtering** (when ready):
   - Add filtering logic to drop `partner_bad_health == 1`
   - Decide on handling of `np.nan` values

### 9.3 Open Questions

- Should we drop observations with `partner_bad_health == np.nan`?
- At what stage in the pipeline should filtering occur?
- Should we create a similar variable for the main estimation sample (`task_create_main_estimation_sample`)?

---

## Appendix: Related Code Locations

### A.1 Merge Module
- File: `src/caregiving/data_management/soep/merge_data/task_load_and_merge_structural_samples.py`
- Function: `task_load_and_merge_estimation_sample` (lines 45-193)
- Health variables loaded: `m11126`, `m11124` from `pequiv` (line 146)

### A.2 Variable Creation Module
- File: `src/caregiving/data_management/soep/task_create_structural_estimation_sample.py`
- Function: `task_create_caregivers_sample` (lines 200-367)
- Partner state created: Line 217
- Health created: Line 271

### A.3 Variable Functions
- File: `src/caregiving/data_management/soep/variables.py`
- `merge_couples()`: Lines 405-428
- `create_partner_state()`: Lines 376-402
- `create_health_var_good_bad()`: Lines 538-569
