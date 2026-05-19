# Analysis: Safety of Changing `start_age_caregiving` and `start_age_parents` in specs.yaml

## Summary

**Both `start_age_caregiving` and `start_age_parents` are SAFE to change**, but with some important considerations.

## ✅ `start_age_caregiving` - SAFE to Change

### How it's used:

1. **In `derive_specs.py` (line 22)**:
   ```python
   specs["start_period_caregiving"] = specs["start_age_caregiving"] - specs["start_age"]
   ```
   ✅ Correctly calculates period offset from `start_age`

2. **In state space functions** (`state_space.py`, `state_space_job_retention.py`, etc.):
   - Used to check if `age <= start_age_caregiving` or `age < start_age_caregiving`
   - Age is calculated as `age = start_age + period`
   ✅ This works correctly - it's a simple age comparison

3. **In caregiving transition** (`caregiving_transition.py`):
   - Uses `start_period_caregiving` from specs (which is correctly calculated)
   - Uses `end_period_caregiving = end_age_caregiving - start_age`
   ✅ Correctly uses period-based indexing

4. **In plotting/moments**:
   - Used for filtering data by age ranges
   ✅ Safe - just filtering/plotting

### Conclusion for `start_age_caregiving`:
✅ **SAFE to change** - All usage is correct and dynamic.

---

## ⚠️ `start_age_parents` - SAFE but with Important Notes

### How it's used:

1. **In `derive_specs.py` (lines 15-20)**:
   ```python
   specs["agent_to_parent_mat_age_offset"] = specs["start_age_parents"] - specs["start_age"]
   specs["parent_to_survival_mat_age_offset"] = specs["start_age_parents"] - specs["survival_min_age"]
   ```
   ✅ Correctly calculates age offsets dynamically

2. **In `caregiving_specs.py` - ADL Transition Matrices**:
   - **Lines 36, 89, 176, 216, 288, 363, 596**: Uses `start_age = specs["start_age_parents"]`
   - Then calculates: `age = start_age + period` (where period is 0-indexed)
   - Uses this age to index into transition matrices from data
   ✅ **This is CORRECT** - ADL transition matrices are indexed by parent age, starting at `start_age_parents`

3. **In `adl_transition.py` - Mother Age Calculation**:
   ```python
   mother_age = (
       period
       - model_specs["agent_to_parent_mat_age_offset"]
       + model_specs["mother_age_diff"][education]
       + PARENT_AGE_OFFSET
   )
   ```
   ✅ **This is CORRECT** - Uses the dynamically calculated `agent_to_parent_mat_age_offset`

4. **In death transition** (`adl_transition.py`):
   ```python
   age_index = mother_age + model_specs["parent_to_survival_mat_age_offset"]
   ```
   ✅ **This is CORRECT** - Uses the dynamically calculated offset

### Important Considerations:

1. **ADL Transition Matrix Data**:
   - The ADL transition matrices are loaded from CSV files that contain data indexed by parent age
   - These matrices are created with `start_age = specs["start_age_parents"]` (line 36 in `caregiving_specs.py`)
   - The matrices are indexed as `[sex, period, health, adl]` where `period` corresponds to parent age starting at `start_age_parents`
   - ✅ **This is correct** - as long as the data files contain the right age range

2. **Mother Age Calculation**:
   - The code correctly converts agent period to parent age using:
     - `agent_to_parent_mat_age_offset` (calculated as `start_age_parents - start_age`)
     - `mother_age_diff[education]` (age difference between agent and mother)
     - `PARENT_AGE_OFFSET = 3` (hardcoded constant)
   - ⚠️ **Note**: `PARENT_AGE_OFFSET = 3` is hardcoded - verify this is correct for your use case

3. **Survival/Death Matrices**:
   - Death transition matrices are indexed by `age_index = age - survival_min_age`
   - The code correctly uses `parent_to_survival_mat_age_offset` to convert parent age to the correct index
   - ✅ **This is correct**

### Potential Issues:

1. **Data Files**:
   - ⚠️ **IMPORTANT**: If you change `start_age_parents`, make sure your ADL transition matrix CSV files contain data for the new age range
   - The matrices are loaded assuming they start at `start_age_parents` and go to `end_age`

2. **Hardcoded `PARENT_AGE_OFFSET = 3`**:
   - This constant in `adl_transition.py` is used in mother age calculation
   - Verify this is correct for your model

### Conclusion for `start_age_parents`:
✅ **SAFE to change**, but:
1. Ensure ADL transition matrix data files cover the new age range
2. Verify `PARENT_AGE_OFFSET = 3` is correct
3. The indexing logic is correct and will work with different starting ages

---

## Summary Table

| Parameter | Safe to Change? | Notes |
|-----------|----------------|-------|
| `start_age_caregiving` | ✅ YES | All usage is correct and dynamic |
| `start_age_parents` | ✅ YES (with caveats) | Correctly implemented, but verify data files cover new age range |

## Recommendations

1. ✅ **`start_age_caregiving`**: Safe to change without restrictions
2. ⚠️ **`start_age_parents`**: Safe to change, but:
   - Verify ADL transition matrix CSV files contain data for the new age range
   - Verify `PARENT_AGE_OFFSET = 3` is correct
   - Test that mother age calculations work correctly with new value
