# Period Indexing Analysis: Does period 0 = start_age?

## Answer: **YES, but it depends on which specs**

## Summary

If you change `start_age` to 40, then **agent-related specs** will have `period 0 = age 40`, but **parent-related specs** will still use `start_age_parents` (50).

---

## ✅ Agent-Related Specs (Use `specs["start_age"]`)

These specs use `period + specs["start_age"]` to get age, so **period 0 = start_age**:

### 1. **Health Transition Specs** (`health_specs.py`)
- **Line 60**: `current_age = period + specs["start_age"]`
- **Line 128**: `current_age = period + specs["start_age"]`
- ✅ **If `start_age = 40`, then period 0 = age 40**

### 2. **Partner Transition Specs** (`family_specs.py`)
- **Line 98**: `age = period + specs["start_age"]`
- ✅ **If `start_age = 40`, then period 0 = age 40**

### 3. **Children Specs** (`family_specs.py`)
- **Line 128**: `age = period + specs["start_age"]`
- ✅ **If `start_age = 40`, then period 0 = age 40**

### 4. **Experience Specs** (`experience_specs.py`)
- **Line 17**: `min_SRA_period = specs["min_SRA"] - specs["start_age"]`
- **Line 25**: Uses `specs["start_age"]` for period calculations
- ✅ **If `start_age = 40`, then period 0 = age 40**

### 5. **Income Specs** (`income_specs.py`)
- **Line 177**: Uses `specs["start_age"]` for period calculations
- ✅ **If `start_age = 40`, then period 0 = age 40**

### 6. **Exogenous Care Supply** (`caregiving_specs.py`)
- **Line 424**: `start_age = specs["start_age"]` (for exogenous care supply)
- ✅ **If `start_age = 40`, then period 0 = age 40**

---

## ⚠️ Parent-Related Specs (Use `specs["start_age_parents"]`)

These specs use `specs["start_age_parents"]` instead, so **period 0 = start_age_parents (50)**:

### 1. **ADL Transition Specs** (`caregiving_specs.py`)
- **Line 36**: `start_age = specs["start_age_parents"]`
- **Line 50**: `age = start_age + period` (where `start_age = start_age_parents`)
- ⚠️ **Period 0 = start_age_parents (50), NOT start_age**

### 2. **ADL State Transition Specs** (`caregiving_specs.py`)
- **Line 176**: `start_age = specs["start_age_parents"]`
- **Line 189**: `age = start_age + period`
- ⚠️ **Period 0 = start_age_parents (50), NOT start_age**

### 3. **ADL Light/Intensive Transition Specs** (`caregiving_specs.py`)
- **Line 288**: `start_age = specs["start_age_parents"]`
- **Line 313**: `age = start_age + period`
- ⚠️ **Period 0 = start_age_parents (50), NOT start_age**

### 4. **Weighted ADL Transitions** (`caregiving_specs.py`)
- **Line 596**: `start_age = specs["start_age_parents"]`
- ⚠️ **Period 0 = start_age_parents (50), NOT start_age**

---

## Example

If you set `start_age = 40` in `specs.yaml`:

### Agent Specs (period 0 = age 40):
- Health transition matrix: `health_trans_mat[sex, edu, period=0, ...]` → age 40
- Partner transition matrix: `partner_trans_mat[edu, period=0, ...]` → age 40
- Children specs: `children_by_state[..., period=0]` → age 40

### Parent Specs (period 0 = age 50):
- ADL transition matrix: `adl_trans_mat[sex, period=0, ...]` → age 50 (start_age_parents)
- ADL state transition: `adl_state_trans_mat[sex, period=0, ...]` → age 50

---

## Conclusion

**For agent-related specs**: ✅ **YES** - If you change `start_age` to 40, period 0 will refer to age 40.

**For parent-related specs**: ⚠️ **NO** - These use `start_age_parents` (50), so period 0 = age 50 regardless of `start_age`.

This is **correct behavior** because:
- Agent transition matrices (health, partner, children) are indexed by agent age (starting at `start_age`)
- Parent transition matrices (ADL) are indexed by parent age (starting at `start_age_parents`)

The code correctly uses the appropriate starting age for each type of transition matrix.
