# Elterngeld alignment: implementation outline

This document outlines **step-by-step** how to align the 65% caregiving leave with German Elterngeld in the codebase. No code changes are made until you give the go.

**Target behaviour (German Elterngeld):**
- Benefit is **tax-free** but subject to **Progressionsvorbehalt** (§32b EStG): it is not part of taxable income, but is added only to compute the *average tax rate*, which is then applied to taxable income.
- When the person receives the benefit, it **replaces** wage-replacement benefits (e.g. unemployment) for that period; the household does not get both.

**Current behaviour (65% leave):**
- Benefit is **fully taxable** (included in `own_income_after_ssc`, then `calc_net_household_income` taxes it).
- Household income is `max(total_net_household_income + child_benefits, household_unemployment_benefits)`, so unemployment can be received on top of (or instead of) the leave benefit.

---

## Part A: Progressionsvorbehalt (tax treatment)

**Goal:** Treat the 65% caregiving leave benefit as **not** part of taxable income, but include it **only** when computing the average tax rate, then apply that rate to taxable income.

### Step A1: Tax module — support progression income

**File:** `src/caregiving/model/wealth_and_budget/tax_and_ssc.py`

- Add an optional parameter to `calc_net_household_income`, e.g. `progression_income=0.0` (or `None`; if `None`, treat as 0).
- **When `progression_income > 0`:**
  - **Taxable** (base for tax): `family_income = own_income + partner_income` (so the caller must **not** include the benefit in `own_income` when using this path).
  - **Rate base** (for average rate): `family_income + progression_income`.
  - Per single-equivalent: `taxable_single = family_income / split_factor`, `rate_base_single = (family_income + progression_income) / split_factor`.
  - Compute `tax_on_rate_base = calc_inc_tax_for_single_income(rate_base_single, model_specs)` (existing function).
  - Average rate: `avg_rate = tax_on_rate_base / rate_base_single` (handle division by zero when `rate_base_single == 0`).
  - Tax on taxable: `tax_single = avg_rate * taxable_single`.
  - Household tax: `income_tax = tax_single * split_factor`.
  - **Return:** `(family_income - income_tax + progression_income, income_tax, income_tax_single)` so that disposable income includes the (untaxed) benefit. For `income_tax_single`, either compute with progression on own income only, or keep as tax that would apply to own income only (document choice).
- **When `progression_income == 0`:** Keep current behaviour unchanged (tax on `family_income`; return `family_income - income_tax`, etc.).
- Add a short docstring note that `progression_income` implements a Progressionsvorbehalt (benefit not in tax base, but used to compute average tax rate).

### Step A2: 65% budget equation — exclude benefit from tax base and pass progression

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_caregiving_leave_with_job_retention.py`

- Compute **taxable** own income **without** the caregiving leave top-up:
  - `own_income_for_tax = was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc` (no `+ caregiving_leave_top_up`).
- Call `calc_net_household_income(own_income=own_income_for_tax, partner_income=partner_income_after_ssc, has_partner_int=has_partner_int, model_specs=model_specs, progression_income=caregiving_leave_top_up)`.
- Use the returned `total_net_household_income` as is (it already includes the benefit on the disposable side when progression is used). No need to add `caregiving_leave_top_up` again.
- Ensure any downstream use of `income_tax_total` / `income_tax_single` still receives the values returned by this call (government budget, etc.).

### Step A3: Government budget and tests

- **Government budget** (`government_budget_caregiving_leave_with_job_retention.py`): No change to the interface; it already receives `household_income_tax_total` from the budget equation. The new tax logic will simply yield a (generally) lower income tax when progression is used.
- **Tests:** Add or update tests that:
  - Call `calc_net_household_income` with `progression_income > 0` and check that (i) tax is lower than when the same amount is included in `own_income` and (ii) disposable is `family_income - tax + progression_income`.
  - Optionally check that the 65% budget equation returns the same total disposable when using progression as when (conceptually) adding the untaxed benefit to net income after tax.

### Step A4: Docstrings and specs

- In `budget_equation_caregiving_leave_with_job_retention.py`, add a short comment that the 65% caregiving leave benefit is treated with Progressionsvorbehalt (not in tax base, only in rate).
- In `caregiving_leave_top_up.py`, in the docstring for `calc_caregiving_leave_top_up`, state that the benefit is tax-free but subject to Progressionsvorbehalt (implementation in budget equation + tax module).
- If there is a central specs or policy doc, add a one-line note that the 65% leave benefit uses Progressionsvorbehalt.

---

## Part B: Benefit replaces unemployment

**Goal:** When the person receives the 65% caregiving leave benefit (`caregiving_leave_top_up > 0`), total household income is **not** the maximum of (net income + child benefits) and unemployment; the benefit replaces unemployment for that period.

### Step B1: Total income when on 65% leave

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_caregiving_leave_with_job_retention.py`

- Replace the current:
  - `total_income = jnp.maximum(total_net_household_income + child_benefits, household_unemployment_benefits)`
- With logic mirroring the full-leave budget equation:
  - Define `receives_care_leave_benefit = caregiving_leave_top_up > 0`.
  - `total_income = jnp.where(receives_care_leave_benefit, total_net_household_income + child_benefits, jnp.maximum(total_net_household_income + child_benefits, household_unemployment_benefits))`.
- So when the household receives the 65% leave benefit, they get `total_net_household_income + child_benefits` only (no max with unemployment).

### Step B2: (Optional) Benefit formula: full 65% vs top-up over unemployment

**Current:** The 65% benefit is implemented as a *top-up*: e.g. when unemployed, `top_up = max(0, 65%_net - household_unemployment_benefits)`, so effectively the household gets `unemployment + top_up = max(unemployment, 65%_net)` before the change in Step B1.

**After Step B1 only:** You keep this formula; when on leave you simply do not take the max with unemployment. So the household gets `total_net_household_income + child_benefits` (which already includes the top-up in taxable/base income depending on Part A). This is consistent with “benefit replaces unemployment” in the sense that you do not add unemployment on top.

**Optional alternative (full 65% when on leave):** If you want the *amount* of the benefit to be the full 65% (bounded) and not “65% minus unemployment”:
- In `caregiving_leave_top_up.py`, for the unemployed cases (prior PT unemployed, prior FT unemployed), optionally compute the benefit as the clipped 65% amount **without** subtracting `household_unemployment_benefits` (e.g. pass a flag or use a separate code path). Then in the budget equation, when on leave, total income is still without max with unemployment. This would mirror the full (Norwegian-style) leave, where the benefit is a full wage replacement and does not subtract unemployment.
- Document in the outline or in code which policy you implement: “top-up over unemployment but no max with unemployment when on leave” vs “full 65% benefit when on leave, no unemployment.”

### Step B3: Government budget (unemployment expenditure)

- When the household is on 65% leave and benefit replaces unemployment, the government should **not** count unemployment benefits for that household (or the budget logic already reflects that no unemployment is paid when they receive the leave benefit). Confirm in `government_budget_caregiving_leave_with_job_retention.py` how `household_unemployment_benefits` is used: if it is the *potential* unemployment and expenditures are computed elsewhere, you may need to zero out or reduce unemployment expenditure for households with `caregiving_leave_top_up > 0`. Outline: add a step to check and, if needed, pass a flag or adjust so that when `caregiving_leave_top_up > 0`, unemployment expenditure for that household is not counted (or is reduced accordingly).

### Step B4: Docstrings

- In `budget_equation_caregiving_leave_with_job_retention.py`, add a short comment that when the household receives the 65% caregiving leave benefit, total income does not take the maximum with unemployment (benefit replaces unemployment).
- In `caregiving_leave_top_up.py`, in the docstring for `calc_caregiving_leave_top_up`, state that when the benefit is received, it replaces unemployment (handled in the budget equation).

---

## Part C: Optional — 67% / 100% rate tiers (German schedule)

If you later want to align with the German Elterngeld rate schedule (e.g. 67% for lower incomes, 65% default, 100% for a period):

- **Specs:** Add parameters (e.g. in `specs.yaml` or `model_specs`) for replacement rates by tier and income thresholds.
- **`caregiving_leave_top_up.py`:** In `calc_caregiving_leave_top_up`, compute the applicable rate from previous period income (e.g. net FT/PT income) and the thresholds, then apply the same bounds (300–1800 monthly) and the same “minus unemployment” or “full amount” logic as chosen in Part B.
- No change to the tax or “replaces unemployment” logic beyond what is in Part A and B.

---

## Order of implementation

1. **Part A** (Progressionsvorbehalt): A1 → A2 → A3 → A4.
2. **Part B** (replaces unemployment): B1 → B2 (if desired) → B3 → B4.
3. **Part C** only if you introduce rate tiers later.

---

## Files to touch (summary)

| Part | File | Changes |
|------|------|--------|
| A1   | `tax_and_ssc.py` | Optional `progression_income` in `calc_net_household_income`; rate-base tax and return disposable including progression income. |
| A2   | `budget_equation_caregiving_leave_with_job_retention.py` | Taxable income without top-up; call tax with `progression_income=caregiving_leave_top_up`. |
| A3   | `government_budget_caregiving_leave_with_job_retention.py` | No interface change; optional test that tax is lower. |
| A4   | Docstrings in budget equation, `caregiving_leave_top_up.py`, and specs | Document Progressionsvorbehalt. |
| B1   | `budget_equation_caregiving_leave_with_job_retention.py` | `total_income` conditional on `receives_care_leave_benefit`. |
| B2   | `caregiving_leave_top_up.py` | Optional: full 65% without subtracting unemployment. |
| B3   | `government_budget_caregiving_leave_with_job_retention.py` | Ensure unemployment expenditure is not counted when household receives leave benefit, if applicable. |
| B4   | Docstrings | Document “replaces unemployment”. |

Once you give the go, implementation can follow this outline step by step.
