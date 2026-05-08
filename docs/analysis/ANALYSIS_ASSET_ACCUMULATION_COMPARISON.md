# Asset Accumulation Comparison: No Inheritance vs. No Care Demand Models

## Executive Summary

This document provides a meticulous comparison of asset accumulation mechanisms between two counterfactual models:
1. **No Inheritance Model** (`budget_equation_no_inheritance.py`): Baseline model with inheritance removed
2. **No Care Demand Model** (`budget_equation_no_care_demand.py`): Counterfactual model without care demand processes

The analysis traces every sub-function and component affecting asset accumulation.

### ⚠️ CRITICAL FINDING

**The no-inheritance model has HIGHER asset accumulation than the no-care-demand model** despite not receiving inheritance. The primary reason is:

**Experience Accumulation Bonus for Intensive Caregivers**: In the no-inheritance model, part-time workers providing intensive informal care receive **full experience credit (1.0)** instead of reduced credit (0.5). This leads to:
- Higher experience accumulation over working life
- Higher wages (experience affects wage function)
- Higher retirement income (experience affects pension calculation)
- **Higher lifetime earnings and asset accumulation**

In contrast, the no-care-demand model cannot access this bonus (care demand process removed), so part-time workers always accumulate experience at the reduced rate (0.5), leading to lower lifetime earnings and lower assets.

**The experience effect dominates the inheritance effect** because experience affects income in every period, while inheritance is a one-time transfer.

---

## 1. Budget Constraint Function Signatures

### 1.1 No Inheritance Model
```python
def budget_constraint(
    period,
    education,
    lagged_choice,
    experience,
    partner_state,
    care_demand,          # ← Present
    mother_dead,
    asset_end_of_previous_period,
    income_shock_previous_period,
    model_specs,
)
```

**Key Differences:**
- Includes `care_demand` parameter (3-state: NO_CARE_DEMAND, LIGHT, INTENSIVE)
- Uses 16-choice space (4 labor states × 4 care arrangements)
- `mother_dead` present but inheritance logic removed

### 1.2 No Care Demand Model
```python
def budget_constraint(
    period,
    education,
    lagged_choice,
    experience,
    partner_state,
    mother_dead,          # ← Present (for inheritance)
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,               # ← Additional parameter
    model_specs,
)
```

**Key Differences:**
- **No `care_demand` parameter** (care demand process removed)
- Uses 4-choice space (retirement, unemployed, part-time, full-time)
- `mother_dead` retained for inheritance calculation
- Additional `params` parameter (unused in budget constraint but may be used elsewhere)

---

## 2. Asset Accumulation Formula

### 2.1 Core Formula (Both Models)

Both models use the same fundamental asset accumulation equation:

```
assets_begin_of_period = assets_scaled + total_income_plus_interest
```

Where:
- `assets_scaled = asset_end_of_previous_period * wealth_unit`
- `total_income_plus_interest = total_income + interest + [other components]`

### 2.2 No Inheritance Model

```python
total_income_plus_interest = total_income + interest
```

**Components:**
- `total_income`: Net household income + child benefits (max with unemployment benefits)
- `interest`: `interest_rate * assets_scaled`
- **No inheritance**: `bequest_from_parent = 0` (commented out)
- **No care benefits/costs**: `care_benefits_and_costs = 0` (set to zero)

### 2.3 No Care Demand Model

```python
total_income_plus_interest = total_income + interest + bequest_from_parent
```

**Components:**
- `total_income`: Net household income + child benefits (max with unemployment benefits)
- `interest`: `interest_rate * assets_scaled`
- **Inheritance included**: `bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount`
- **No care benefits/costs**: Not calculated (care demand process removed)

---

## 3. Income Components Comparison

### 3.1 Labor Income

#### No Inheritance Model
- **Function**: `calc_labor_income_after_ssc` from `wages.py`
- **Choice space**: 16 choices (uses `is_part_time` and `is_full_time` from `shared.py`)
- **Logic**: Determines part-time vs. full-time from lagged_choice using baseline choice arrays
- **Hours calculation**: Uses `av_annual_hours_pt` and `av_annual_hours_ft` from model_specs

#### No Care Demand Model
- **Function**: `calc_labor_income_after_ssc` from `wages_no_care_demand.py`
- **Choice space**: 4 choices (uses `is_part_time` and `is_full_time` from `shared_no_care_demand.py`)
- **Logic**: Direct mapping (choice 2 = part-time, choice 3 = full-time)
- **Hours calculation**: Same `av_annual_hours_pt` and `av_annual_hours_ft` from model_specs

**Impact on Asset Accumulation**: **IDENTICAL** - Both use the same wage calculation logic and hours parameters. The only difference is the choice space interpretation.

### 3.2 Retirement Income

**Both Models:**
- **Function**: `calc_pensions_after_ssc` (identical)
- **Input**: `experience_years` (calculated identically via `construct_experience_years`)
- **Logic**: Pension points based on experience, converted to gross retirement income, then net after SSC

**Impact on Asset Accumulation**: **IDENTICAL**

### 3.3 Partner Income

**Both Models:**
- **Function**: `calc_partner_income_after_ssc` (identical)
- **Logic**: Partner income depends on `partner_state` (0=no partner, 1=working, 2=retired)
- **Returns**: `(partner_income_after_ssc, gross_partner_income, gross_partner_pension)`

**Impact on Asset Accumulation**: **IDENTICAL**

### 3.4 Unemployment Benefits

**Both Models:**
- **Function**: `calc_unemployment_benefits` (identical)
- **Logic**: Means-tested based on assets, education, sex, partner status, period
- **Returns**: `(household_unemployment_benefits, own_unemployment_benefits)`

**Impact on Asset Accumulation**: **IDENTICAL**

### 3.5 Child Benefits

**Both Models:**
- **Function**: `calc_child_benefits` (identical)
- **Logic**: Depends on education, sex, partner status, period
- **Returns**: Annual child benefits amount

**Impact on Asset Accumulation**: **IDENTICAL**

### 3.6 Total Income Calculation

#### No Inheritance Model
```python
total_income = jnp.maximum(
    total_net_household_income + child_benefits,
    household_unemployment_benefits,
)
```

#### No Care Demand Model
```python
total_income = jnp.maximum(
    total_net_household_income + child_benefits,
    household_unemployment_benefits,
)
```

**Impact on Asset Accumulation**: **IDENTICAL**

---

## 4. Care Benefits and Costs

### 4.1 No Inheritance Model

```python
care_benefits_and_costs = jnp.zeros_like(child_benefits)  # Set to zero
```

**Rationale**: Inheritance is removed, but care demand process still exists. However, care benefits/costs are explicitly set to zero in the budget constraint.

**Function Available but Not Used**: `calc_care_benefits_and_costs` exists in `transfers.py` but is **not called** in the no-inheritance budget constraint.

**Impact on Asset Accumulation**:
- Care benefits: **0** (no additional income)
- Care costs: **0** (no deductions)
- **Net effect**: Neutral (no impact on assets)

### 4.2 No Care Demand Model

**No care benefits/costs calculation**: The function `calc_care_benefits_and_costs` is not called at all because:
- `care_demand` parameter is removed from function signature
- Care demand process is eliminated from the model
- No care-related transfers exist

**Impact on Asset Accumulation**:
- Care benefits: **0** (no calculation)
- Care costs: **0** (no calculation)
- **Net effect**: Neutral (no impact on assets)

**Comparison**: Both models have **zero care benefits/costs**, but for different reasons:
- **No Inheritance**: Care demand exists but benefits/costs set to zero
- **No Care Demand**: Care demand process completely removed

---

## 5. Inheritance Calculations

### 5.1 No Inheritance Model

**Inheritance Logic**: **COMPLETELY REMOVED**

```python
# Inheritance-related code is commented out:
# "bequest_from_parent": bequest_from_parent / model_specs["wealth_unit"],
# "gets_inheritance": gets_inheritance,
```

**Functions Not Called**:
- `calc_inheritance_amount` - Not called
- `draw_inheritance_outcome` - Not called

**Impact on Asset Accumulation**:
- `bequest_from_parent = 0` always
- No inheritance income added to `total_income_plus_interest`
- **Result**: Lower asset accumulation compared to baseline (inheritance removed)

### 5.2 No Care Demand Model

**Inheritance Logic**: **FULLY IMPLEMENTED**

```python
# Only compute inheritance if mother recently died this period (state 1)
mother_died_recently = mother_dead == PARENT_RECENTLY_DEAD
inheritance_amount = calc_inheritance_amount_no_care_demand(
    period=period,
    education=education,
    model_specs=model_specs,
)
gets_inheritance = draw_inheritance_outcome_no_care_demand(
    period=period,
    education=education,
    asset_end_of_previous_period=asset_end_of_previous_period,
    model_specs=model_specs,
)
bequest_from_parent = mother_died_recently * gets_inheritance * inheritance_amount
```

**Functions Called**:
- `calc_inheritance_amount_no_care_demand` from `transfers_no_care_demand.py`
- `draw_inheritance_outcome_no_care_demand` from `transfers_no_care_demand.py`

**Key Differences from Baseline Inheritance Functions**:

#### Inheritance Amount Calculation

**Baseline** (`calc_inheritance_amount`):
- Uses `lagged_choice` to determine care type (no_care, light_care, intensive_care)
- Looks up inheritance amount from matrix: `inheritance_amount_mat[sex, period, education, care_type_idx]`
- Care type index: 0=no_care, 1=light_care, 2=intensive_care

**No Care Demand** (`calc_inheritance_amount_no_care_demand`):
- **No `lagged_choice` parameter** (care choices don't exist)
- Always uses `care_type_idx = 0` (no_care column)
- Looks up inheritance amount from matrix: `inheritance_amount_mat[sex, period, education, 0]`
- **Result**: Uses "no care" inheritance amounts regardless of previous choices

#### Inheritance Probability/Draw

**Baseline** (`draw_inheritance_outcome`):
- Uses `lagged_choice` to determine care type (binary: any_care vs. no_care)
- Looks up probability: `inheritance_prob_mat[sex, period, education, care_type_idx]`
- Care type index: 0=no_care, 1=any_care (light or intensive)
- Seed includes `lagged_choice * 7` in calculation

**No Care Demand** (`draw_inheritance_outcome_no_care_demand`):
- **No `lagged_choice` parameter**
- Always uses `care_type_idx = 0` (no_care column)
- Looks up probability: `inheritance_prob_mat[sex, period, education, 0]`
- Seed calculation: `base_seed + period * 1000 + education * 200 + asset_end_of_previous_period * 3`
- **Result**: Uses "no care" inheritance probabilities

**Impact on Asset Accumulation**:
- `bequest_from_parent` can be positive when `mother_died_recently == 1` and `gets_inheritance == 1`
- Inheritance amount added to `total_income_plus_interest`
- **Result**: Higher asset accumulation compared to no-inheritance model (inheritance included)

---

## 6. Interest Calculation

**Both Models**: **IDENTICAL**

```python
interest_rate = model_specs["interest_rate"]
interest = interest_rate * assets_scaled
```

**Impact on Asset Accumulation**: **IDENTICAL** - Both models apply the same interest rate to scaled assets.

---

## 7. Government Budget Components

### 7.1 No Inheritance Model

```python
calc_government_budget_components(
    household_income_tax_total=income_tax_total,
    was_worker=was_worker,
    was_retired=was_retired,
    gross_labor_income=gross_labor_income,
    gross_retirement_income=gross_retirement_income,
    partner_state=partner_state,
    gross_partner_income=gross_partner_income,
    gross_partner_pension=gross_partner_pension,
    child_benefits=child_benefits,
    care_benefits_and_costs=jnp.zeros_like(child_benefits),  # ← Zero
    household_unemployment_benefits=household_unemployment_benefits,
    model_specs=model_specs,
)
```

**Key Point**: `care_benefits_and_costs` is explicitly set to zero.

### 7.2 No Care Demand Model

```python
calc_government_budget_components(
    household_income_tax_total=income_tax_total,
    was_worker=was_worker,
    was_retired=was_retired,
    gross_labor_income=gross_labor_income,
    gross_retirement_income=gross_retirement_income,
    partner_state=partner_state,
    gross_partner_income=gross_partner_income,
    gross_partner_pension=gross_partner_pension,
    child_benefits=child_benefits,
    care_benefits_and_costs=jnp.zeros_like(child_benefits),  # ← Zero
    household_unemployment_benefits=household_unemployment_benefits,
    model_specs=model_specs,
)
```

**Key Point**: `care_benefits_and_costs` is also set to zero (no care demand process).

**Impact on Asset Accumulation**: **IDENTICAL** - Government budget calculations are the same in both models. These calculations are for tracking purposes only and do not affect individual asset accumulation directly.

---

## 8. Model Specification Differences

### 8.1 State Space

#### No Inheritance Model (`task_specify_model_no_inheritance.py`)

**Stochastic States**:
- `partner_state`: 0, 1, 2 (no partner, working, retired)
- `health`: 0, 1, 2 (bad, good, dead)
- `job_offer`: 0, 1 (no offer, offer)
- `mother_dead`: 0, 1, 2 (alive, recently dead, longer dead)
- `mother_adl`: ADL states (light/intensive care needs)
- `care_demand`: 0, 1, 2 (no care, light, intensive)

**Deterministic States**:
- `caregiving_type`: 0, 1 (not caregiver, caregiver)
- `education`: 0, 1 (low, high)
- `already_retired`: 0, 1

**Choices**: 16 choices (4 labor × 4 care arrangements)

#### No Care Demand Model (`task_specify_model_no_care_demand.py`)

**Stochastic States**:
- `partner_state`: 0, 1, 2 (no partner, working, retired)
- `health`: 0, 1, 2 (bad, good, dead)
- `job_offer`: 0, 1 (no offer, offer)
- `mother_dead`: 0, 1, 2 (alive, recently dead, longer dead) ← **Retained for inheritance**
- **No `mother_adl`**: Removed
- **No `care_demand`**: Removed

**Deterministic States**:
- `caregiving_type`: 0, 1 (kept but not used in choice space)
- `education`: 0, 1 (low, high)
- `already_retired`: 0, 1

**Choices**: 4 choices (retirement, unemployed, part-time, full-time)

### 8.2 Stochastic State Transitions

#### No Inheritance Model

```python
{
    "job_offer": job_offer_process_transition,
    "partner_state": partner_transition,
    "health": health_transition,
    "mother_adl": limitations_with_adl_transition,
    "care_demand": care_demand_transition_adl_light_intensive,
    "mother_dead": death_transition,
}
```

**Note**: `inheritance_transition` is **not included** (inheritance removed).

#### No Care Demand Model

```python
{
    "job_offer": job_offer_process_transition,
    "partner_state": partner_transition,
    "health": health_transition,
    "mother_dead": death_transition,  # ← Retained for inheritance
    # No mother_adl or care_demand transitions
}
```

**Impact on Asset Accumulation**:
- **No Inheritance**: Care demand transitions affect choice availability but not direct asset flows (care benefits/costs = 0)
- **No Care Demand**: No care-related state transitions, simpler choice set

---

## 9. Auxiliary Variables (aux dictionary)

### 9.1 No Inheritance Model

**Included**:
- `net_hh_income`
- `hh_net_income_wo_interest`
- `interest`
- `joint_gross_labor_income`
- `joint_gross_retirement_income`
- `gross_partner_income`
- `gross_partner_pension`
- `gross_labor_income`
- `gross_retirement_income`
- `income_shock_previous_period`
- `income_shock_for_labor`
- `own_income_after_ssc`
- `child_benefits`
- `household_unemployment_benefits`
- Government budget components (tax, SSC, expenditures, net budget)

**Excluded** (commented out):
- `bequest_from_parent`
- `gets_inheritance`

### 9.2 No Care Demand Model

**Included**:
- All variables from no-inheritance model
- **`bequest_from_parent`**: Included
- **`gets_inheritance`**: Included

**Excluded**:
- `care_benefits_and_costs` (not calculated)

**Impact on Asset Accumulation**: Auxiliary variables are for tracking/simulation output only. They do not affect the asset accumulation calculation itself.

---

## 10. Summary: Asset Accumulation Differences

### 10.1 Identical Components

The following components are **identical** in both models:
1. **Labor income calculation** (same wage functions, same hours)
2. **Retirement income** (same pension calculation)
3. **Partner income** (same partner income logic)
4. **Unemployment benefits** (same means-testing)
5. **Child benefits** (same calculation)
6. **Interest calculation** (same interest rate)
7. **Total income** (same max with unemployment benefits)
8. **Government budget components** (same calculations, both have zero care benefits/costs)

### 10.2 Key Differences

| Component | No Inheritance Model | No Care Demand Model | Impact on Assets |
|-----------|---------------------|---------------------|------------------|
| **Experience Accumulation** | ✅ **Full credit (1.0) for part-time intensive caregivers** | ❌ Reduced credit (0.5) for all part-time workers | **No Inheritance: MUCH HIGHER** (primary driver) |
| **Inheritance** | ❌ Removed (`bequest_from_parent = 0`) | ✅ Included (uses no_care column) | No Care Demand: **Higher** (but smaller effect than experience) |
| **Care Benefits/Costs** | 0 (explicitly set) | 0 (not calculated) | **Identical** (both zero) |
| **Choice Space** | 16 choices (can access intensive care) | 4 choices (cannot access intensive care) | **Critical**: Affects ability to get experience bonus |
| **Care Demand State** | Present (3 states) | Removed | **Critical**: Required for intensive care experience bonus |

### 10.3 Net Effect on Asset Accumulation

**⚠️ REVISED UNDERSTANDING**: The initial analysis incorrectly assumed no-care-demand would have higher assets. The **opposite is true** due to experience accumulation differences.

**No Inheritance Model**:
```
assets_begin = assets_scaled + total_income + interest
```
- **Higher assets** compared to no-care-demand model (despite no inheritance)
- **Reason**: Part-time intensive caregivers get full experience credit (1.0) → higher wages → higher lifetime earnings
- Same as baseline except inheritance removed, but still benefits from intensive care experience bonus

**No Care Demand Model**:
```
assets_begin = assets_scaled + total_income + interest + bequest_from_parent
```
- **Lower assets** compared to no-inheritance model (despite receiving inheritance)
- **Reason**: Part-time workers always get reduced experience credit (0.5) → lower wages → lower lifetime earnings
- Cannot access intensive care experience bonus (care demand process removed)
- Inheritance included but uses "no care" column (may be lower than baseline intensive care inheritance)

### 10.4 Inheritance Amount Differences

**Critical Difference in Inheritance Calculation**:

- **Baseline Model**: Inheritance amount/probability depends on care type provided (no_care, light_care, intensive_care)
- **No Care Demand Model**: Inheritance amount/probability always uses "no_care" column (index 0)

**Implication**: Even though no-care-demand model includes inheritance, the amounts/probabilities may differ from what a baseline agent would receive if they provided care, because:
- Baseline caregivers might receive higher inheritance (if care increases inheritance probability/amount)
- No-care-demand model always uses "no care" inheritance rates

---

## 11. Code-Level Differences Summary

### 11.1 Function Calls Comparison

| Function | No Inheritance | No Care Demand | Notes |
|---------|---------------|----------------|-------|
| `calc_labor_income_after_ssc` | ✅ `wages.py` | ✅ `wages_no_care_demand.py` | Different modules, same logic |
| `calc_pensions_after_ssc` | ✅ | ✅ | Identical |
| `calc_partner_income_after_ssc` | ✅ | ✅ | Identical |
| `calc_unemployment_benefits` | ✅ | ✅ | Identical |
| `calc_child_benefits` | ✅ | ✅ | Identical |
| `calc_net_household_income` | ✅ | ✅ | Identical |
| `calc_care_benefits_and_costs` | ❌ Not called | ❌ Not called | Both zero |
| `calc_inheritance_amount` | ❌ Not called | ❌ Not called | No inheritance version used |
| `calc_inheritance_amount_no_care_demand` | ❌ Not called | ✅ Called | No-care-demand specific |
| `draw_inheritance_outcome` | ❌ Not called | ❌ Not called | No inheritance version used |
| `draw_inheritance_outcome_no_care_demand` | ❌ Not called | ✅ Called | No-care-demand specific |
| `calc_government_budget_components` | ✅ | ✅ | Identical (both pass zero care benefits) |

### 11.2 Import Differences

**No Inheritance Model**:
- Imports from `shared.py` (16-choice space helpers)
- Imports from `wages.py` (baseline wage functions)
- Imports from `transfers.py` (but doesn't use inheritance functions)

**No Care Demand Model**:
- Imports from `shared_no_care_demand.py` (4-choice space helpers)
- Imports from `wages_no_care_demand.py` (no-care-demand wage functions)
- Imports from `transfers_no_care_demand.py` (inheritance functions for no-care-demand)

---

## 12. Conclusion

### 12.1 Primary Difference (REVISED)

The **fundamental difference** in asset accumulation between the two models is:

1. **No Inheritance Model**: Removes inheritance BUT retains intensive care experience bonus → **HIGHER assets**
2. **No Care Demand Model**: Includes inheritance BUT cannot access intensive care experience bonus → **LOWER assets**

**The experience accumulation difference dominates the inheritance difference.**

### 12.2 Secondary Differences

1. **Experience Accumulation**: **CRITICAL** - Part-time intensive caregivers get full experience credit (1.0) in no-inheritance vs. reduced credit (0.5) in no-care-demand
2. **Choice Space**: 16 vs. 4 choices (affects ability to access intensive care experience bonus)
3. **Care Demand State**: Present vs. absent (affects experience bonus eligibility)
4. **Inheritance**: Removed vs. included (but effect is smaller than experience effect)
5. **Care Benefits/Costs**: Both zero, but for different reasons (explicit zero vs. not calculated)

### 12.3 Asset Accumulation Formula Comparison

**No Inheritance**:
```
assets_begin = assets_scaled + (total_income + interest)
```
Where `total_income` includes higher wages due to intensive care experience bonus

**No Care Demand**:
```
assets_begin = assets_scaled + (total_income + interest + bequest_from_parent)
```
Where `total_income` includes lower wages due to no intensive care experience bonus, and `bequest_from_parent` uses "no care" inheritance rates

**Net Result**: The higher `total_income` in no-inheritance (from experience bonus) exceeds the `bequest_from_parent` in no-care-demand, leading to higher assets in no-inheritance model.

---

## Appendix: Function Call Trees

### No Inheritance Model Budget Constraint Call Tree

```
budget_constraint()
├── construct_experience_years()
├── calc_partner_income_after_ssc()
├── calc_pensions_after_ssc()
├── calc_unemployment_benefits()
├── calc_labor_income_after_ssc() [wages.py]
│   ├── calculate_gross_labor_income()
│   │   ├── is_part_time() [shared.py]
│   │   ├── is_full_time() [shared.py]
│   │   └── calc_hourly_wage()
│   └── calc_after_ssc_income_worker()
├── is_working() [shared.py]
├── is_retired() [shared.py]
├── calc_net_household_income()
├── calc_child_benefits()
├── calc_government_budget_components()
│   ├── calc_pension_unempl_contr()
│   └── calc_health_ltc_contr()
└── [NO INHERITANCE CALCULATIONS]
```

### No Care Demand Model Budget Constraint Call Tree

```
budget_constraint()
├── construct_experience_years()
├── calc_partner_income_after_ssc()
├── calc_pensions_after_ssc()
├── calc_unemployment_benefits()
├── calc_labor_income_after_ssc() [wages_no_care_demand.py]
│   ├── calculate_gross_labor_income()
│   │   ├── is_part_time() [shared_no_care_demand.py]
│   │   ├── is_full_time() [shared_no_care_demand.py]
│   │   └── calc_hourly_wage()
│   └── calc_after_ssc_income_worker()
├── is_working() [shared_no_care_demand.py]
├── is_retired() [shared_no_care_demand.py]
├── calc_net_household_income()
├── calc_child_benefits()
├── calc_inheritance_amount_no_care_demand() [transfers_no_care_demand.py]
├── draw_inheritance_outcome_no_care_demand() [transfers_no_care_demand.py]
│   └── jax.random.uniform() [for inheritance draw]
└── calc_government_budget_components()
    ├── calc_pension_unempl_contr()
    └── calc_health_ltc_contr()
```

---

## 13. CRITICAL FINDING: Experience Accumulation Difference

### 13.1 The Key Difference

**This is the primary reason why the no-inheritance model has HIGHER asset accumulation than the no-care-demand model.**

### 13.2 Experience Update Formula Comparison

#### No Inheritance Model (`experience_baseline_model.py`)

```python
# Line 69-72
intensive_care = is_intensive_informal_care(lagged_choice)
exp_update = is_full_time(lagged_choice) + is_part_time(lagged_choice) * (
    model_specs["exp_increase_part_time"] * (1 - intensive_care) + intensive_care
)
```

**Experience Update Logic:**
- **Full-time workers**: `exp_update = 1.0` (always full credit)
- **Part-time workers WITHOUT intensive care**: `exp_update = exp_increase_part_time` (typically 0.5)
- **Part-time workers WITH intensive care**: `exp_update = 1.0` (FULL CREDIT - intensive care bonus!)

**Mathematical Breakdown:**
- If `intensive_care = 0`: `exp_update = exp_increase_part_time * 1 + 0 = exp_increase_part_time`
- If `intensive_care = 1`: `exp_update = exp_increase_part_time * 0 + 1 = 1.0`

#### No Care Demand Model (`experience_no_care_demand.py`)

```python
# Line 53-55
exp_update = is_full_time(lagged_choice) + is_part_time(lagged_choice) * (
    model_specs["exp_increase_part_time"]
)
```

**Experience Update Logic:**
- **Full-time workers**: `exp_update = 1.0` (always full credit)
- **Part-time workers**: `exp_update = exp_increase_part_time` (typically 0.5) - **ALWAYS, no bonus**

**Mathematical Breakdown:**
- Part-time workers always get: `exp_update = exp_increase_part_time` (no intensive care bonus possible)

### 13.3 Impact on Asset Accumulation

#### Direct Effects

1. **Experience Stock**:
   - **No Inheritance**: Part-time intensive caregivers accumulate experience at 1.0 per period
   - **No Care Demand**: Part-time workers accumulate experience at 0.5 per period
   - **Result**: No-inheritance model accumulates experience **2x faster** for part-time intensive caregivers

2. **Wage Growth**:
   - Wages depend on experience: `hourly_wage = calc_hourly_wage(..., experience_years, ...)`
   - Higher experience → Higher wages
   - **Result**: No-inheritance model has higher wages for part-time intensive caregivers

3. **Retirement Income**:
   - Pensions depend on experience: `calc_pensions_after_ssc(pension_points=experience_years, ...)`
   - Higher experience → Higher pension points → Higher retirement income
   - **Result**: No-inheritance model has higher retirement income

#### Cumulative Effects Over Lifetime

**Example Calculation** (assuming `exp_increase_part_time = 0.5`):

**Scenario**: Agent works part-time for 10 periods while providing intensive care

**No Inheritance Model**:
- Experience accumulated: `10 periods × 1.0 = 10.0 years`
- Higher wages during working years
- Higher pension in retirement

**No Care Demand Model**:
- Experience accumulated: `10 periods × 0.5 = 5.0 years`
- Lower wages during working years
- Lower pension in retirement

**Net Effect**: The no-inheritance model accumulates **twice as much experience** for part-time intensive caregivers, leading to:
- Higher lifetime labor income
- Higher retirement income
- **Higher asset accumulation**

### 13.4 Why This Matters More Than Inheritance

**Inheritance Effect**:
- Inheritance is a **one-time transfer** when mother dies
- Only affects agents whose mothers die during simulation
- Amount depends on period, education, and (in baseline) care type

**Experience Effect**:
- Experience accumulation is **continuous** throughout working life
- Affects **all part-time intensive caregivers** in every period
- Compounds over time (higher experience → higher wages → more savings → more interest)
- Affects both working income AND retirement income

**Conclusion**: The experience accumulation difference is likely **much larger** than the inheritance difference for agents who provide intensive care part-time, explaining why no-inheritance has higher assets despite not receiving inheritance.

### 13.5 Is This a Bug?

**This is NOT a bug** - it's an intentional policy feature in the baseline model:

**Policy Rationale** (from code comment in `experience_baseline_model.py` line 67-68):
> "Full pension point (1.0) for part-time workers providing intensive informal care"

This reflects a real-world policy where part-time workers providing intensive care receive **full pension credit** (experience credit) for their part-time work, recognizing the caregiving burden.

**In the No Care Demand Counterfactual**:
- There is no intensive care (care demand process removed)
- Therefore, the intensive care bonus cannot apply
- Part-time workers always get reduced experience credit (0.5)

**This is correct behavior** for the counterfactual, but it means:
- The no-care-demand model cannot benefit from the intensive care experience bonus
- Part-time workers in no-care-demand accumulate less experience
- This leads to lower lifetime earnings and lower assets

### 13.6 State Space Function Assignment

**No Inheritance Model**:
- Uses `state_space.py` → `create_state_space_functions()`
- Calls `get_next_period_experience` from `experience_baseline_model.py`
- **Includes intensive care bonus**

**No Care Demand Model**:
- Uses `state_space_no_care_demand.py` → `create_state_space_functions()`
- Calls `get_next_period_experience` from `experience_no_care_demand.py`
- **No intensive care bonus** (care demand removed)

### 13.7 Quantifying the Effect

To estimate the magnitude:

1. **Experience Difference per Period**:
   - Part-time intensive caregiver: `1.0 - 0.5 = 0.5 years` per period

2. **Over 10 periods of part-time intensive care**:
   - Experience difference: `10 × 0.5 = 5.0 years`

3. **Wage Impact**:
   - Wage function: `wage = f(experience)`
   - 5 years more experience → significant wage increase (depends on wage function parameters)

4. **Pension Impact**:
   - Pension: `pension = f(experience_years)`
   - 5 years more experience → significant pension increase

5. **Lifetime Income Impact**:
   - Higher wages × working years + Higher pension × retirement years
   - This can easily exceed typical inheritance amounts (which are one-time transfers)

### 13.8 Summary: Why No-Inheritance Has Higher Assets

**Primary Reason**: **Experience Accumulation Bonus for Intensive Caregivers**

1. **No Inheritance Model**:
   - Part-time intensive caregivers get **full experience credit** (1.0) instead of reduced (0.5)
   - Higher experience → Higher wages → Higher lifetime earnings
   - Higher experience → Higher pensions → Higher retirement income
   - **Result**: Higher asset accumulation despite no inheritance

2. **No Care Demand Model**:
   - Part-time workers always get **reduced experience credit** (0.5)
   - Lower experience → Lower wages → Lower lifetime earnings
   - Lower experience → Lower pensions → Lower retirement income
   - **Result**: Lower asset accumulation despite receiving inheritance

**The experience effect dominates the inheritance effect** because:
- Experience affects income in **every period** of working life
- Experience affects retirement income for **all retirement periods**
- Inheritance is a **one-time transfer** that only occurs when mother dies
- The intensive care experience bonus applies to **all part-time intensive caregivers**

---

## 14. Additional Considerations

### 14.1 Labor Supply Patterns

The experience accumulation difference may also affect **labor supply decisions**:

- **No Inheritance Model**: Part-time intensive caregivers have incentive to continue working part-time (get full experience credit)
- **No Care Demand Model**: Part-time workers have less incentive to work part-time (only get 0.5 experience credit)

This could lead to:
- Different labor supply patterns
- Different lifetime work histories
- Further compounding the asset accumulation difference

### 14.2 Choice Set Differences

**No Inheritance Model**: 16 choices (4 labor × 4 care arrangements)
- Agents can choose part-time + intensive care
- This choice gets full experience credit

**No Care Demand Model**: 4 choices (retirement, unemployed, part-time, full-time)
- Agents can choose part-time, but no intensive care option exists
- Part-time always gets reduced experience credit

**Implication**: The choice set difference means agents in the no-care-demand model **cannot** access the intensive care experience bonus, even if they wanted to.

### 14.3 Inheritance Amount Comparison

Even though no-care-demand includes inheritance, the amounts may be lower:

- **Baseline Model**: Inheritance amount depends on care type (no_care, light_care, intensive_care)
- **No Care Demand Model**: Inheritance amount always uses "no_care" column (index 0)

If intensive caregivers receive higher inheritance in the baseline, then:
- No-care-demand model gets lower inheritance (always "no_care" rates)
- This further reduces the inheritance advantage of no-care-demand model

### 14.4 Potential Bug Check

**Question**: Should the no-care-demand model use the same experience accumulation as baseline for part-time workers, or is the reduced experience correct?

**Answer**: The reduced experience is **correct** for the counterfactual because:
1. The intensive care bonus is a policy feature for caregivers
2. In no-care-demand, there is no caregiving (care demand removed)
3. Therefore, the bonus should not apply

**However**, this creates an **apples-to-oranges comparison**:
- No-inheritance model: Baseline with inheritance removed (still has caregiving, still gets intensive care bonus)
- No-care-demand model: Counterfactual without care demand (no caregiving, no intensive care bonus)

**For a fair comparison**, you might want:
- No-inheritance model: Baseline with inheritance removed (current implementation)
- No-care-demand model: Baseline with care demand removed BUT with intensive care bonus still available (if part-time workers could somehow provide intensive care, which they can't in this counterfactual)

But this is a **conceptual issue**, not a bug. The models are correctly implementing their respective counterfactuals.

### 14.5 Verification Checklist

To verify this analysis, check:

1. **Experience Accumulation**:
   - [ ] Confirm `exp_increase_part_time = 0.5` in model_specs
   - [ ] Verify part-time intensive caregivers in no-inheritance get `exp_update = 1.0`
   - [ ] Verify part-time workers in no-care-demand get `exp_update = 0.5`

2. **Wage Function**:
   - [ ] Check if `calc_hourly_wage` uses experience as input
   - [ ] Verify that higher experience leads to higher wages
   - [ ] Calculate wage difference for 5 years experience difference

3. **Pension Function**:
   - [ ] Check if `calc_pensions_after_ssc` uses experience as input
   - [ ] Verify that higher experience leads to higher pensions
   - [ ] Calculate pension difference for 5 years experience difference

4. **Simulation Results**:
   - [ ] Compare average experience levels between models
   - [ ] Compare average wages between models
   - [ ] Compare average pensions between models
   - [ ] Compare average assets between models
   - [ ] Check share of part-time intensive caregivers in no-inheritance model

5. **Inheritance Comparison**:
   - [ ] Compare inheritance amounts: baseline intensive care vs. no-care-demand "no care"
   - [ ] Calculate average inheritance received per agent in no-care-demand
   - [ ] Compare to lifetime income difference from experience bonus

---

**End of Analysis**
