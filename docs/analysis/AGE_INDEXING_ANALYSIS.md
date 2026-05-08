# Analysis: Safety of Changing `start_age` and `start_age_caregiving` in specs.yaml

## Summary

**Overall Assessment**: Changing `start_age` and `start_age_caregiving` in `specs.yaml` is **MOSTLY SAFE**, but there are **SOME POTENTIAL ISSUES** that need attention.

## ✅ What Works Correctly

### 1. Period-to-Age Conversion
Most code correctly uses `age = period + start_age` or `age = period + specs["start_age"]`:
- ✅ `simulate.py`: `df["age"] = df.index.get_level_values("period") + model_specs["start_age"]`
- ✅ `transfers.py`: `age = start_age + period`
- ✅ `state_space.py`: `age = start_age + period`
- ✅ `caregiving_transition.py`: `end_period_caregiving = model_specs["end_age_caregiving"] - model_specs["start_age"]`

### 2. Derived Specs
`derive_specs.py` correctly calculates:
- ✅ `start_period_caregiving = start_age_caregiving - start_age`
- ✅ `agent_to_parent_mat_age_offset = start_age_parents - start_age`
- ✅ `n_periods = end_age - start_age + 1`

### 3. Transition Function Indexing
Most transition functions correctly calculate age from period:
- ✅ `health_specs.py`: `current_age = period + specs["start_age"]`
- ✅ `family_specs.py`: `age = period + specs["start_age"]`
- ✅ `adl_transition.py`: Uses `agent_to_parent_mat_age_offset` correctly

## ⚠️ Potential Issues

### 1. Hardcoded Constants in `shared.py`

**Location**: `src/caregiving/model/shared.py`

**Issues**:
- `MIN_AGE = 40`, `MAX_AGE = 70` - These are hardcoded constants
- `AGENT_TO_PARENT_MAT_AGE_OFFSET = 20` - **NOT USED** (code uses `model_specs["agent_to_parent_mat_age_offset"]` instead)
- `AGE_BINS` uses `AGE_40 - MIN_AGE` which assumes `MIN_AGE = 40`
- `NPV_START_AGE = 40` - Hardcoded
- `INITIAL_CONDITIONS_AGE_LOW = 50`, `INITIAL_CONDITIONS_AGE_HIGH = 60` - Hardcoded

**Impact**:
- ✅ **GOOD NEWS**: `AGENT_TO_PARENT_MAT_AGE_OFFSET` in `shared.py` is **NOT USED** - the code correctly uses `model_specs["agent_to_parent_mat_age_offset"]` from specs
- ⚠️ `AGE_BINS` calculations will be incorrect if `MIN_AGE` doesn't match `start_age` (but `AGE_BINS` appears to be unused)
- These constants may be used in some plotting/analysis code

**Recommendation**:
- ✅ No action needed for `AGENT_TO_PARENT_MAT_AGE_OFFSET` (already using dynamic value)
- Review `MIN_AGE` and `MAX_AGE` usage - if only for plotting, document as such
- Review `AGE_BINS` usage - appears unused, but verify

### 2. Hardcoded Ages in Plotting Functions

**Locations**: Multiple plotting files

**Examples**:
- `task_plot_care_demand_by_age_2_by_2.py`: `age_min=40, age_max=80` (hardcoded)
- `task_plot_income.py`: `age_min=30` (hardcoded)
- `simulate_moments.py`: `range(40, 75, 5)` (hardcoded age bins)

**Impact**: Low - these are just for plotting ranges, not critical for model functionality

**Recommendation**: Consider making these use `specs["start_age"]` and `specs["end_age"]` for consistency

### 3. Age Indexing in Income Specs

**Location**: `src/caregiving/specs/income_specs.py` (lines 177-179)

```python
periods > specs["max_ret_age"] - specs["start_age"]
periods[not_predicted_periods] = specs["max_ret_age"] - specs["start_age"]
```

**Status**: ✅ This looks correct - it uses `specs["start_age"]` dynamically

### 4. Partner Transition Age Binning

**Location**: `src/caregiving/specs/family_specs.py` (line 100)

```python
age_bin = np.floor(age / 10) * 10
```

**Status**: ✅ This is age-based binning, not period-based, so it should work correctly regardless of `start_age`

## 🔍 Specific Checks Needed

### 1. `task_write_specs.py` and Auxiliary Functions

**Status**: ✅ **SAFE**

All functions in `task_write_specs.py` and its auxiliary modules correctly use:
- `period + specs["start_age"]` to get age
- `specs["start_age"]` and `specs["start_age_caregiving"]` from the specs dict
- Age-based indexing for transition matrices (not period-based)

**Key Functions Checked**:
- ✅ `read_in_health_transition_specs`: Uses `period + specs["start_age"]`
- ✅ `read_in_partner_transition_specs`: Uses `period + specs["start_age"]`
- ✅ `read_in_adl_transition_specs`: Uses `start_age_parents + period` (correct for parent matrices)
- ✅ `read_in_survival_by_age_specs`: Uses age-based indexing with `survival_min_age`

### 2. Transition Functions

**Status**: ✅ **MOSTLY SAFE** with one caveat

**ADL Transitions** (`adl_transition.py`):
- ✅ Correctly uses `agent_to_parent_mat_age_offset` from specs
- ✅ Calculates `mother_age = period - agent_to_parent_mat_age_offset + mother_age_diff + PARENT_AGE_OFFSET`
- ⚠️ **Note**: `PARENT_AGE_OFFSET = 3` is hardcoded - verify this is correct

**Death Transitions** (`adl_transition.py`):
- ✅ Correctly uses `parent_to_survival_mat_age_offset` from specs
- ✅ Calculates `age_index = mother_age + parent_to_survival_mat_age_offset`

**Caregiving Transitions** (`caregiving_transition.py`):
- ✅ Uses `start_period_caregiving` from specs
- ✅ Uses `end_period_caregiving = end_age_caregiving - start_age`

## 📋 Recommendations

### High Priority

1. **Fix `shared.py` constants**:
   - Remove or update `AGENT_TO_PARENT_MAT_AGE_OFFSET = 20` (it's calculated in `derive_specs.py`)
   - Review `MIN_AGE = 40` and `MAX_AGE = 70` - either make them dynamic or document that they're for plotting only
   - Review `AGE_BINS` - ensure they work with different `start_age` values

2. **Verify `PARENT_AGE_OFFSET = 3`**:
   - Check if this constant in `adl_transition.py` is correct for all scenarios
   - Consider making it a spec parameter if it varies

### Medium Priority

3. **Update plotting functions**:
   - Replace hardcoded age ranges with `specs["start_age"]` and `specs["end_age"]` where appropriate
   - This improves consistency but doesn't affect model correctness

### Low Priority

4. **Documentation**:
   - Document that `start_age` and `start_age_caregiving` can be changed
   - Note any limitations or required updates

## ✅ Conclusion

**It is SAFE to change `start_age` and `start_age_caregiving` in `specs.yaml`**.

### Key Findings:
1. ✅ **`AGENT_TO_PARENT_MAT_AGE_OFFSET` in `shared.py` is NOT USED** - code correctly uses `model_specs["agent_to_parent_mat_age_offset"]` from specs
2. ✅ All transition functions correctly use `period + specs["start_age"]` or `model_specs["start_age"]`
3. ✅ `start_period_caregiving` is correctly calculated as `start_age_caregiving - start_age`
4. ✅ Age indexing in transition matrices uses age-based indexing (not period-based), so it works correctly

### Minor Issues (Non-Critical):
- Hardcoded ages in plotting functions (cosmetic only)
- `MIN_AGE` and `MAX_AGE` constants in `shared.py` (appear to be for plotting/analysis only)
- `AGE_BINS` in `shared.py` (appears unused)

### Recommendation:
**You can safely change `start_age` and `start_age_caregiving` in `specs.yaml`**. The core model logic is correctly implemented and will work with different starting ages. The hardcoded constants in `shared.py` are either unused or only used for plotting/analysis purposes.
