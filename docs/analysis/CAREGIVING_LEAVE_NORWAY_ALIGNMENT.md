# Caregiving Leave: Steps to Align with Norwegian Implementation

This document describes the code changes required to align the **full caregiving leave** (and optionally the **normal 65% leave**) policy with the Norwegian design: **gross** benefit, **taxable**, **no SSC** on the benefit, and **care leave replaces unemployment** when on leave and not working. No code has been modified yet; this is the implementation plan only.

---

## 1. Target: Norwegian-Style Rules

| Aspect | Current implementation | Target (Norway-style) |
|--------|------------------------|------------------------|
| **Benefit level** | Net top-up to previous net wage (minus unemployment when unemp.) | 100% of **previous gross** wage when not working; gross gap when FT→PT |
| **SSC on benefit** | None (benefit is “after-SSC”) | None (unchanged) |
| **Income tax** | Yes (benefit included in taxable income) | Yes (unchanged) |
| **Unemployment** | Top-up added; then `total_income = max(household_net, unemployment)` | Care leave benefit **replaces** unemployment when on leave and not working (no max with unemployment) |

Reference: Norwegian Tax Administration (Skatteetaten) — care benefits from the National Insurance scheme are reported as **gross**, subject to **withholding tax**, and **not** a basis for employer’s National Insurance contributions.

---

## 2. Files to Modify

- `src/caregiving/model/wealth_and_budget/caregiving_leave_top_up.py`
- `src/caregiving/model/wealth_and_budget/budget_equation_full_caregiving_leave_with_job_retention.py`
- `src/caregiving/model/wealth_and_budget/budget_equation_caregiving_leave_with_job_retention.py` (if aligning normal leave)
- `src/caregiving/model/wealth_and_budget/government_budget_caregiving_leave_with_job_retention.py` (documentation / naming only, no formula change)
- Specs (e.g. `src/caregiving/specs.yaml` or wherever caregiving leave bounds/rates live) — documentation only
- Tests / fiscal table tasks (sanity checks after implementation)

---

## 3. Step-by-Step Code Changes

### Step 1: Full leave — compute benefit in **gross** terms

**File:** `src/caregiving/model/wealth_and_budget/caregiving_leave_top_up.py`
**Function:** `calc_full_caregiving_leave_top_up` (approx. lines 17–124).

**Current behavior:**

- Benefit is a **net** top-up: gap to **net** (after-SSC) wage.
- When unemployed: `topup = max(net_wage - household_unemployment_benefits, 0)`.
- When FT→PT: `topup = max(net_ft_income - labor_income_after_ssc, 0)`.
- Uses `net_ft_income`, `net_pt_income`, `labor_income_after_ssc`, `household_unemployment_benefits`.

**Changes to make:**

1. **Previously FT, now unemployed**
   - **Current:** `topup_prior_ft_unemp = max(net_ft_income - household_unemployment_benefits, 0) * mask_prior_ft_unemp`
   - **New:** `benefit = gross_ft_income_min_checked * mask_prior_ft_unemp` (100% of previous **gross** FT wage; do **not** subtract unemployment).

2. **Previously FT, now PT**
   - **Current:** `topup_prior_ft_pt = max(net_ft_income - labor_income_after_ssc, 0) * mask_prior_ft_pt`
   - **New:** `benefit = max(0, gross_ft_income_min_checked - gross_pt_income_min_checked) * mask_prior_ft_pt` (gross gap; need `gross_pt_income_min_checked` in scope — it already is).

3. **Previously PT, now unemployed**
   - **Current:** `topup_prior_pt = max(net_pt_income - household_unemployment_benefits, 0) * mask_prior_pt_unemp`
   - **New:** `benefit = gross_pt_income_min_checked * mask_prior_pt_unemp` (100% of previous **gross** PT wage; do **not** subtract unemployment).

4. **Previously no job, now unemployed**
   - **Current:** `topup_prior_none = 0.0 * mask_prior_none_unemp`
   - **New:** Leave as zero (no wage replacement).

5. **Remove** all use of `household_unemployment_benefits` and `net_ft_income` / `net_pt_income` for the **amount** of the benefit. Keep `labor_income_after_ssc` only if still needed for another purpose (e.g. FT→PT gap in gross terms does not need it). You can keep `calc_after_ssc_income_worker` and net variables in the function only if needed elsewhere (e.g. for bounds or future use); for the Norwegian alignment, the **returned** benefit should be the sum of **gross** amounts above.

6. **Return:** Same single annual amount (now gross). The variable name can stay `wage_replacement_annual` or be renamed to `caregiving_leave_benefit_gross` for clarity.

7. **Docstring:** Update to state that the benefit is **100% of previous gross wage** (when not working) or **gross wage gap** (when FT→PT), Norwegian-style; taxable; not subject to SSC; and that when unemployed on leave it **replaces** unemployment (handled in the budget equation).

**Signature:** The function can keep its current signature. `household_unemployment_benefits` and `labor_income_after_ssc` may become unused for the full-leave formula; they can be kept for API compatibility or removed if no longer needed.

---

### Step 2: Full leave — treat benefit as gross in the budget (no SSC on benefit)

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_full_caregiving_leave_with_job_retention.py`

**Current (approx. lines 105–130):**

- `caregiving_leave_top_up = calc_full_caregiving_leave_top_up(...)`
- `own_income_after_ssc = was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc + caregiving_leave_top_up`
- That `own_income` is passed to `calc_net_household_income` (income tax).

**Changes to make:**

1. **No formula change** for how `own_income_after_ssc` is built: the care leave amount is still added as one term. After Step 1, that term is now a **gross** amount (no SSC deducted on it), which is correct for Norway.
2. **Optional:** Rename the variable to `caregiving_leave_benefit_gross` in this file when it is first assigned and when it is used in `own_income_after_ssc` and in the government budget call, to make the “gross, no SSC” semantics explicit in the code.
3. **Comment:** Add a short comment above the aggregation that the care leave benefit is a **gross** amount (no SSC), added to taxable income and subject only to income tax.

No change to the call to `calc_net_household_income` or to the tax logic; the only change is the **definition** of the benefit (now gross) from Step 1.

---

### Step 3: Full leave — care leave replaces unemployment (no stacking)

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_full_caregiving_leave_with_job_retention.py`

**Current (approx. lines 155–159):**

```python
total_income = jnp.maximum(
    total_net_household_income + child_benefits,
    household_unemployment_benefits,
)
```

So whenever `household_unemployment_benefits` is higher than `total_net_household_income + child_benefits`, the household gets unemployment instead.

**Changes to make:**

1. When the agent is on **care leave** and receives a **positive** leave benefit, do **not** take the maximum with unemployment. So:
   - If `caregiving_leave_top_up > 0` (or, more precisely, if the agent is in informal care and not working and receives the gross benefit), then set:
     - `total_income = total_net_household_income + child_benefits`
   - Else (no care leave benefit):
     - `total_income = jnp.maximum(total_net_household_income + child_benefits, household_unemployment_benefits)`

2. Implementation option (recommended): define a scalar or mask `receives_care_leave_benefit = (caregiving_leave_top_up > 0)`. Then:
   - `total_income = jnp.where(
       receives_care_leave_benefit,
       total_net_household_income + child_benefits,
       jnp.maximum(total_net_household_income + child_benefits, household_unemployment_benefits),
   )`

3. **Comment:** Add a one-line comment that when the household receives the care leave benefit, unemployment is not applied (Norwegian-style: benefit replaces unemployment).

---

### Step 4: Government budget (full leave)

**File:** `src/caregiving/model/wealth_and_budget/government_budget_caregiving_leave_with_job_retention.py`

**Current:** `government_expenditures` includes `caregiving_leave_top_up` (around line 109).

**Changes to make:**

1. **No formula change:** Continue to add the same variable (the care leave amount) to `government_expenditures`. After Step 1, that amount is the **gross** benefit, which is correct (government pays the gross).
2. **Docstring:** In the function docstring, state that `caregiving_leave_top_up` (or `caregiving_leave_benefit_gross`) is the **gross** care leave benefit paid by the government (Norwegian-style; taxable to recipient; not subject to SSC).

---

### Step 5 (Optional): Normal leave (65%) — same Norwegian logic

**File:** `src/caregiving/model/wealth_and_budget/caregiving_leave_top_up.py`
**Function:** `calc_caregiving_leave_top_up` (approx. lines 128–264).

**Current behavior:**

- Benefit = 65% of **net** wage (with bounds), minus unemployment or minus current PT income.
- Uses `net_ft_income`, `net_pt_income`, `household_unemployment_benefits`, `labor_income_after_ssc`.

**Changes to make (parallel to full leave):**

1. **Define benefit in gross terms:** e.g. `benefit_gross = clip(replacement_rate * gross_wage, lower_bound_annual, upper_bound_annual)` for the relevant previous job (FT or PT). Use **gross** wages and **gross** bounds if bounds are intended to be gross (otherwise keep bounds as-is but apply to gross wage).
2. **Previously FT, now unemployed:** benefit = clipped 65% of **gross** FT wage (no subtraction of unemployment).
3. **Previously FT, now PT:** benefit = max(0, clipped 65% of gross FT wage minus **gross** PT income) or equivalent gross-based gap.
4. **Previously PT, now unemployed:** benefit = clipped 65% of **gross** PT wage (no subtraction of unemployment).
5. **Previously none, now unemployed:** 0.
6. **Return** this gross amount.

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_caregiving_leave_with_job_retention.py`

- Apply the same **total_income** logic as in Step 3: when `caregiving_leave_top_up > 0`, do not take max with `household_unemployment_benefits`.
- Optionally add the same comment and variable naming as in Steps 2–3.

---

### Step 6: Specs and documentation

**Files:** Specs (e.g. `src/caregiving/specs.yaml`) and any parameter docs.

**Changes to make:**

1. If there are parameters such as `caregiving_leave_benefit_lower_bound`, `caregiving_leave_benefit_upper_bound`, or `caregiving_leave_benefit_replacement_rate`, add a short note that under the Norwegian-style design they refer to **gross** wages (and that the benefit is taxable and not subject to SSC).
2. In `caregiving_leave_top_up.py`, update the module docstring and the docstrings of `calc_full_caregiving_leave_top_up` and (if changed) `calc_caregiving_leave_top_up` to state:
   - Benefit is defined as a **gross** amount (100% or 65% of previous gross wage / gross gap).
   - It is **taxable** (included in household income for income tax).
   - It is **not** subject to social security contributions.
   - When the person is on leave and not working, the care leave benefit **replaces** unemployment (no stacking; implemented in the budget equation via the `total_income` rule).

---

### Step 7: Tests and sanity checks

**No new test file is strictly required;** the following checks are recommended after implementing Steps 1–4 (and 5 if applicable).

1. **Fiscal / government budget**
   - In a scenario where an agent is on **full** leave (e.g. previously FT, now unemployed):
     - The care leave payment in the simulated data (or in a single-period test) equals **gross** FT wage (with min-wage floor), not net.
     - Government expenditure includes this gross amount.
     - Household pays income tax on (partner income + this gross amount) (and possibly other income).
   - Run the fiscal table task (e.g. `task_fiscal.py`) and spot-check that totals and “avg cost per caregiver” move in a plausible direction (higher gross benefit, higher tax revenue).

2. **No stacking with unemployment**
   - For an agent on care leave and unemployed with positive gross benefit, verify that `total_income` (or consumption) is **not** set to `household_unemployment_benefits` when that would be higher; it should be `total_net_household_income + child_benefits` (and similar components as in the budget equation).

3. **Unit / regression tests**
   - If there are tests that assert on `calc_full_caregiving_leave_top_up` or on the budget equation (e.g. in `tests/test_budget_constraint.py` or similar), update expected values: benefit should be gross, and (if tested) the unemployment-max logic should be conditional on care leave.
   - Optionally add a small test: for a given (education, experience, job_before_caregiving, lagged_choice=informal_care, unemployed), the returned benefit equals the expected gross FT or PT wage (with min check).

---

## 4. Summary Checklist

| Step | Description | File(s) |
|------|-------------|--------|
| 1 | Full leave: compute benefit as **gross** (100% previous gross when unemp.; gross gap when FT→PT); do not subtract unemployment | `caregiving_leave_top_up.py` — `calc_full_caregiving_leave_top_up` |
| 2 | Full leave: treat benefit as gross in budget (no SSC); optional rename/comment | `budget_equation_full_caregiving_leave_with_job_retention.py` |
| 3 | Full leave: when care leave benefit > 0, do not take max with unemployment | `budget_equation_full_caregiving_leave_with_job_retention.py` — `total_income` |
| 4 | Government expenditure and docstring (gross benefit) | `government_budget_caregiving_leave_with_job_retention.py` |
| 5 | (Optional) Normal leave: same gross / no-SSC / replaces-unemployment logic | `caregiving_leave_top_up.py` — `calc_caregiving_leave_top_up`; `budget_equation_caregiving_leave_with_job_retention.py` |
| 6 | Specs and docstrings (gross, taxable, no SSC, replaces unemployment) | Specs; `caregiving_leave_top_up.py` |
| 7 | Tests and sanity checks (fiscal, no stacking, unit tests) | Tests; fiscal tasks |

---

## 5. Order of Implementation

Recommended order:

1. **Step 1** (full leave benefit formula in gross terms).
2. **Steps 2 and 3** (budget equation: same aggregation, then `total_income` conditional on care leave).
3. **Step 4** (government budget docstring).
4. **Step 7** (run fiscal table and sanity checks).
5. **Step 5** (normal leave) if desired.
6. **Step 6** (specs and docstrings).
7. **Step 7** (any new or updated unit tests).

This keeps the full-leave policy consistent first, then extends the same logic to the normal leave and documentation.
