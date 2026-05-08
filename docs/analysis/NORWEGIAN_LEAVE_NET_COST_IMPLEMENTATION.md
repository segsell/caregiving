# Norwegian (full) caregiving leave: government net cost — implementation outline

**Goal:** Compute the **net cost to the government** of the full (Norwegian-style) caregiving leave benefit. Because the benefit is **taxable**, part of the gross expenditure is clawed back via income tax. Net cost = gross benefit paid out − income tax attributable to that benefit.

**No code changes are made in this document; it only specifies where and how to modify or extend the code.**

---

## 1. Definitions

- **Gross cost (gross top-up):** The full caregiving leave benefit amount paid to the household (`caregiving_leave_top_up` from `calc_full_caregiving_leave_top_up`). This is already computed and used as expenditure in the government budget.
- **Income tax attributable to the benefit:** The difference between (i) income tax actually paid when the benefit is included in taxable income and (ii) income tax that would be paid if the benefit were excluded from taxable income (all else unchanged). So:
  - `tax_with_benefit` = current `income_tax_total` (benefit in `own_income_after_ssc`).
  - `tax_without_benefit` = income tax with `own_income` = `was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc` (no benefit).
  - `tax_attributable_to_benefit` = `tax_with_benefit − tax_without_benefit`.
- **Net cost (net top-up):**
  `net_cost_full_leave = full_caregiving_leave_benefit − tax_attributable_to_benefit`.
  This is the net fiscal cost of the leave policy for that household (expenditure minus extra tax collected because the benefit is taxable).

---

## 2. Where to compute the counterfactual tax and net cost

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_full_caregiving_leave_with_job_retention.py`

**Current flow (simplified):**
- `own_income_after_ssc` = worker/retired income + `caregiving_leave_top_up`.
- `calc_net_household_income(own_income=own_income_after_ssc, ...)` → `total_net_household_income`, `income_tax_total`, `income_tax_single`.

**Modifications:**

1. **Compute taxable income without the benefit (counterfactual for tax only).**
   - After forming `own_income_after_ssc` (with benefit), add:
     - `own_income_for_tax_without_benefit = was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc`
     (same as current `own_income_after_ssc` but **without** `+ caregiving_leave_top_up`).

2. **Call the tax function a second time for the counterfactual.**
   - Call `calc_net_household_income(own_income=own_income_for_tax_without_benefit, partner_income=partner_income_after_ssc, has_partner_int=has_partner_int, model_specs=model_specs)` and unpack only the income tax total (e.g. `_, tax_without_benefit, _ = ...`).
   - No need to use the returned disposable income; we only need the tax.

3. **Compute tax attributable and net cost.**
   - `tax_attributable_to_benefit = income_tax_total - tax_without_benefit`
     (both scalar or array, depending on how the budget is called).
   - `full_leave_net_cost = caregiving_leave_top_up - tax_attributable_to_benefit`
     (net cost to government for the full leave benefit).

4. **Keep household disposable income unchanged.**
   - Continue using the **first** tax call (with benefit in income) for `total_net_household_income`, `income_tax_total`, and all downstream consumption/wealth. The second call is only for measuring tax attributable to the benefit.

**Placement in the file:** Immediately after the existing `calc_net_household_income(...)` block that produces `total_net_household_income`, `income_tax_total`, `income_tax_single`, add the counterfactual tax call and the two formulas above. Use the same `partner_income_after_ssc`, `has_partner_int`, and `model_specs` as in the first call.

---

## 3. Where to use the net cost: government budget function

**File:** `src/caregiving/model/wealth_and_budget/government_budget_caregiving_leave_with_job_retention.py`
**Function:** `calc_government_budget_components_full_caregiving_leave_with_job_retention`

**Current behaviour:**
- `government_expenditures` includes `full_caregiving_leave_benefit` (gross).
- `total_tax_revenue` includes `household_income_tax_total` (which already contains tax on the benefit).
- So `net_government_budget = total_tax_revenue - government_expenditures` is correct at the **aggregate** level, but the **reported expenditure** for the leave policy is gross, not net.

**Options (choose one and implement consistently):**

**Option A — Report net expenditure for full leave (change definition of “expenditure” for this policy):**
- Add a new parameter: `full_leave_net_cost` (or pass `tax_attributable_to_benefit` and compute net inside the function).
- In the expenditure calculation, use **net** cost for the full leave component instead of gross:
  `government_expenditures = child_benefits + care_benefits + unemployment_paid + full_leave_net_cost`
  (and keep `total_tax_revenue` unchanged: still `household_income_tax_total + own_ssc + partner_ssc`).
- **Consequence:** `net_government_budget` is unchanged (algebraically the same), but “government expenditures” becomes a **net** concept for the full leave line (gross benefit minus tax attributable).
- **Docstring and parameter list:** Document that the full-leave component is net cost; either rename the parameter to `full_leave_net_cost` or add a second parameter and document that when both are passed, expenditure uses the net.

**Option B — Keep expenditure gross; add net cost as an extra return or for reporting only:**
- Keep `government_expenditures` as is (including gross `full_caregiving_leave_benefit`).
- Add an optional parameter `full_leave_net_cost` (or `tax_attributable_to_benefit`). If provided, compute and return an additional quantity, e.g. `full_leave_net_cost` or `full_leave_gross_expenditure` and `full_leave_net_expenditure`, so that callers (or aggregation scripts) can report net cost without changing the definition of total government expenditures.

**Recommendation for a clean “net cost” concept:** Prefer **Option A**: define the full-leave **expenditure** in the government budget as the net cost (gross benefit − tax attributable), and pass `full_leave_net_cost` from the budget equation. Then both the budget equation and the government budget function have a single, consistent definition of “net top-up” for the Norwegian leave.

---

## 4. Budget equation: passing net cost into the government budget

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_full_caregiving_leave_with_job_retention.py`

- After computing `full_leave_net_cost` (see section 2), pass it into the government budget call instead of (or in addition to) the gross benefit, depending on the chosen option:
  - **If Option A:** Call `calc_government_budget_components_full_caregiving_leave_with_job_retention(..., full_caregiving_leave_benefit=caregiving_leave_top_up, full_leave_net_cost=full_leave_net_cost, model_specs=model_specs)` and adapt the government budget function signature and docstring to accept and use `full_leave_net_cost` for the expenditure term (see section 3).
  - **If Option B:** Keep passing `full_caregiving_leave_benefit=caregiving_leave_top_up`; optionally pass `full_leave_net_cost` or `tax_attributable_to_benefit` only if the government budget (or downstream) is extended to return/report it.

---

## 5. Auxiliary dict and downstream use

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_full_caregiving_leave_with_job_retention.py`

- Add to the `aux` dict (so simulation/estimation can aggregate or plot):
  - `tax_attributable_to_full_leave` = `tax_attributable_to_benefit / model_specs["wealth_unit"]` (if you want to expose tax attributable in the same units as other aux).
  - `full_leave_net_cost` (or `full_leave_net_cost` scaled by `wealth_unit` for consistency with other aux entries, e.g. `full_leave_net_cost / model_specs["wealth_unit"]`).
- Optionally keep `caregiving_leave_top_up` in aux as the gross amount for transparency.

**Downstream:** Any script or table that aggregates “cost of full caregiving leave” (e.g. over agents or periods) should use the new net cost variable from aux (or from the government budget return values, if you expose them there) so that reported “government cost of Norwegian leave” is the net cost after income tax.

---

## 6. Government budget function: signature and docstring (full leave)

**File:** `src/caregiving/model/wealth_and_budget/government_budget_caregiving_leave_with_job_retention.py`
**Function:** `calc_government_budget_components_full_caregiving_leave_with_job_retention`

- **If Option A (net expenditure):**
  - Add parameter: `full_leave_net_cost` (required when computing full-leave expenditure as net).
  - Keep `full_caregiving_leave_benefit` in the signature for optional use (e.g. docstring or checks) or drop it and use only `full_leave_net_cost` in the expenditure formula.
  - In the expenditure block, set the full-leave term to `full_leave_net_cost` instead of `full_caregiving_leave_benefit`.
  - Docstring: state that the full leave component in government expenditures is the **net** cost (gross benefit minus income tax attributable to the benefit), and that the budget equation is responsible for computing and passing `full_leave_net_cost`.

- **If Option B (gross expenditure, report net separately):**
  - Add optional parameter(s), e.g. `full_leave_net_cost=None` or `tax_attributable_to_full_leave=None`.
  - If provided, include in the return tuple (e.g. extend the return to include `full_leave_net_cost`) or document that callers can compute net from gross and tax if they have access to both. Expenditure remains gross.

---

## 7. Tests and consistency checks

- **Unit-style check (e.g. in tests or a small script):** For a household with positive `caregiving_leave_top_up`, verify that `tax_without_benefit <= income_tax_total` and that `tax_attributable_to_benefit = income_tax_total - tax_without_benefit` is non-negative and ≤ `caregiving_leave_top_up`, so that `full_leave_net_cost` is in [0, caregiving_leave_top_up].
- **Aggregation sanity check:** Sum of `full_leave_net_cost` over agents/periods should be less than (or equal to) sum of gross benefit, with equality only if the marginal tax on the benefit is zero everywhere.

---

## 8. Summary of files to touch

| File | Change |
|------|--------|
| `budget_equation_full_caregiving_leave_with_job_retention.py` | (i) Compute `own_income_for_tax_without_benefit`; (ii) second call to `calc_net_household_income` to get `tax_without_benefit`; (iii) `tax_attributable_to_benefit` and `full_leave_net_cost`; (iv) pass `full_leave_net_cost` (and possibly gross) to government budget; (v) add to `aux`: e.g. `tax_attributable_to_full_leave`, `full_leave_net_cost` (scaled). |
| `government_budget_caregiving_leave_with_job_retention.py` (full-leave function) | (i) Add parameter `full_leave_net_cost` (and possibly keep or drop `full_caregiving_leave_benefit` depending on option); (ii) use net cost in `government_expenditures` (Option A) or add return/reporting (Option B); (iii) update docstring. |
| Tests / aggregation scripts | Add or adapt tests for tax counterfactual and net cost; ensure any “cost of full leave” reporting uses net cost. |

---

## 9. No change to 65% (normal) leave

The 65% (German-style) leave benefit is tax-free (Progressionsvorbehalt only); there is no “tax attributable” to subtract. So:

- **No** second tax call or net-cost computation in `budget_equation_caregiving_leave_with_job_retention.py`.
- **No** change to `calc_government_budget_components_caregiving_leave_with_job_retention` (65% function).

Net cost logic applies **only** to the full (Norwegian-style) caregiving leave in the full-leave budget equation and the full-leave government budget function.

---

## 10. Comparison with alternative (ChatGPT) approach

A similar design was proposed elsewhere (ChatGPT). The **logic** for the tax-only net cost is the same; the **implementation** and one **optional extension** differ as follows.

### 10.1 Logic alignment (tax-only)

Both use the same definitions:

- **Y₀** = taxable household income without benefit (own income without top-up + partner income).
- **Y₁** = Y₀ + B (with benefit).
- **ΔT** = T(Y₁) − T(Y₀) = tax with benefit − tax without benefit.
- **Net cost (tax-only)** = B − ΔT.

So there is **no logic difference** for the core net cost formula.

### 10.2 Where to compute ΔT and net cost

| Aspect | This document | ChatGPT |
|--------|----------------|---------|
| **Location of the two tax calls** | **Budget equation** (`budget_equation_full_caregiving_leave_with_job_retention.py`). All income components (labor, retirement, partner, benefit) are already there; second call to `calc_net_household_income` with `own_income` excluding benefit. | **Government budget function** (`calc_government_budget_components_full_caregiving_leave_with_job_retention`). Compute and store there; add extra return values. |
| **Implication** | Government budget keeps a small interface: it receives pre-computed `full_leave_net_cost` (and optionally gross). No need to pass `partner_income_after_ssc`, `own_income_no_benefit`, `has_partner_int`, or `model_specs` into the government budget for a second tax call. | To do the two tax calls inside the government budget, that function must either (i) receive the inputs for `calc_net_household_income` (e.g. `own_income_no_benefit`, `partner_income_after_ssc`, `has_partner_int`, `model_specs`) and call the tax function twice, or (ii) receive pre-computed `tax0`/`tax1`/`delta_tax` from the budget equation (in which case the “computation” in the government budget is trivial). |

**Recommendation:** Keeping the two tax calls in the **budget equation** avoids enlarging the government budget signature and keeps tax logic in one place (budget equation already has all after-SSC incomes). The government budget then just consumes `full_leave_net_cost` for the expenditure line and/or reporting.

### 10.3 Transfer crowd-out (ΔU) — optional extension

- **This document:** Net cost is defined as **B − ΔT** only. The fact that “benefit replaces unemployment” is already reflected in the **expenditure** rule: when `full_caregiving_leave_benefit > 0`, `unemployment_paid = 0` in the government budget. So the aggregate budget is consistent; we do not add an explicit “ΔU” term to the net cost formula.
- **ChatGPT:** Adds an explicit **transfer crowd-out** term:
  **Net cost (full)** = B − ΔT − ΔU,
  where ΔU = reduction in other transfers (e.g. unemployment) because of the benefit. With an income floor:
  `transfer = max(0, floor - income)`,
  compute `transfer0` (without benefit) and `transfer1` (with benefit); then `delta_transfer_savings = transfer0 - transfer1` and subtract from net cost.

**When to include ΔU:**

- If the goal is a **single headline “net cost of the leave”** that includes both tax claw-back and the saving from not paying unemployment to that household, then adding **ΔU** is the right concept: **net_cost_full = B − ΔT − ΔU**.
- In our model, `total_income = max(total_net_household_income + child_benefits, household_unemployment_benefits)`, so the implied transfer is
  `transfer = max(0, household_unemployment_benefits - (total_net_household_income + child_benefits))`.
  Compute this with `disposable0 + child_benefits` (without benefit) and `disposable1 + child_benefits` (with benefit); then `delta_transfer_savings = transfer0 - transfer1` (positive when the government pays less unemployment because of the benefit).

**Implementation (if you add ΔU):** In the **budget equation**, after the two tax calls you already have `disposable0` (from counterfactual call) and `total_net_household_income` (disposable1) and `child_benefits`. So:

- `income0 = disposable0 + child_benefits`, `income1 = total_net_household_income + child_benefits` (or disposable1 + child_benefits if you keep it).
- `transfer0 = jnp.maximum(0.0, household_unemployment_benefits - income0)`, `transfer1 = jnp.maximum(0.0, household_unemployment_benefits - income1)`.
- `delta_transfer_savings = transfer0 - transfer1`.
- `full_leave_net_cost_incl_transfer = caregiving_leave_top_up - tax_attributable_to_benefit - delta_transfer_savings`.

You can then report both: **net cost (tax-only)** = B − ΔT and **net cost (incl. transfer savings)** = B − ΔT − ΔU. The current document only mandates B − ΔT; adding ΔU is an optional extension.

### 10.4 Return values / “single source of truth”

- **This document:** Budget equation computes ΔT and net cost, puts them in `aux`, and passes `full_leave_net_cost` to the government budget. Government budget does not return delta_tax or net_cost; it just uses net cost in the expenditure term (Option A) or keeps gross expenditure (Option B).
- **ChatGPT:** Prefers the government budget to be the “single source of truth” and to **return** `delta_tax_from_care_leave`, `net_cost_care_leave`, and optionally `transfer_savings`, `net_cost_care_leave_including_savings`. The budget equation would then pass these through to `aux`.

Functionally equivalent: either the budget equation or the government budget can compute and return these; the other just passes or stores. Choosing the budget equation for computation keeps the government budget’s interface and dependencies smaller.

---

## 11. Additional changes to match ChatGPT suggestions (ΔU + optional returns)

To incorporate the **transfer crowd-out (ΔU)** and the **extra return values** from the ChatGPT design, the following concrete changes are needed. The model already uses an **income floor** for unemployment (`unemployment_transfer_paid = max(0, floor - income)`), so ΔU is the **saving** from paying less of that transfer when the household receives the leave benefit.

### 11.1 Budget equation: capture disposable without benefit and compute ΔU

**File:** `src/caregiving/model/wealth_and_budget/budget_equation_full_caregiving_leave_with_job_retention.py`

1. **Capture disposable income from the counterfactual tax call.**
   Currently the second call is `_, total_tax_without_benefit, _ = calc_net_household_income(...)`. Change to unpack the first return (disposable) as well:
   - `disposable_without_benefit, total_tax_without_benefit, _ = calc_net_household_income(own_income=own_income_for_tax_without_benefit, ...)`.

2. **Compute transfer with and without benefit (after `child_benefits` and `household_net_income_before_floor` / `unemployment_transfer_paid` exist).**
   - `income_with_benefit` = `total_net_household_income + child_benefits` (current income before floor; same as `household_net_income_before_floor` if that name exists).
   - `income_without_benefit` = `disposable_without_benefit + child_benefits`.
   - `transfer_without_benefit` = `jnp.maximum(0.0, household_unemployment_benefits - income_without_benefit)`.
   - `transfer_with_benefit` = current `unemployment_transfer_paid` (already = `max(0, floor - income_with_benefit)`).
   - `delta_transfer_savings` = `transfer_without_benefit - transfer_with_benefit` (≥ 0 when the benefit raises income and thus reduces the transfer).

3. **Net cost including transfer savings (ChatGPT "full" net cost).**
   - `full_leave_net_cost_incl_transfer` = `caregiving_leave_top_up - tax_attributable_to_benefit - delta_transfer_savings`.

4. **Aux.**
   Add to `aux` (scaled by `wealth_unit`):
   `"delta_transfer_savings"`, `"full_leave_net_cost_incl_transfer"`.

5. **Optional: pass to government budget.**
   If the government budget should return or use the "full" net cost, add a parameter (e.g. `full_leave_net_cost_incl_transfer`) and pass it from the budget equation. The government budget would not use it in the expenditure total (expenditure stays gross leave + actual `unemployment_transfer_paid`), but could return it for reporting.

### 11.2 Government budget: optional extra return values (ChatGPT "single source of truth")

**File:** `src/caregiving/model/wealth_and_budget/government_budget_caregiving_leave_with_job_retention.py`
**Function:** `calc_government_budget_components_full_caregiving_leave_with_job_retention`

1. **Optional parameters.**
   Add optional parameters (or require them if the budget equation always passes them):
   `tax_attributable_to_full_leave=None`, `delta_transfer_savings=None`, `full_leave_net_cost_incl_transfer=None`.

2. **Return tuple.**
   Extend the return to include, when provided:
   `tax_attributable_to_full_leave`, `full_leave_net_cost` (tax-only), `delta_transfer_savings`, `full_leave_net_cost_incl_transfer`.
   Callers (e.g. the budget equation) would then put these into `aux` if desired, so the government module is the single place that "returns" these metrics even when they are computed in the budget equation and passed in.

3. **No change to expenditure logic.**
   Expenditure stays: `child_benefits + care_benefits + unemployment_transfer_paid + full_caregiving_leave_benefit`. Net cost (tax-only or incl. transfer) is for **reporting** only, not for subtracting from expenditure (to avoid double counting; see section 10 and the fix applied earlier).

### 11.3 Summary checklist

| Item | Where | Change |
|------|--------|--------|
| Capture disposable without benefit | Budget equation | Unpack first return from second `calc_net_household_income` call. |
| transfer0, transfer1, ΔU | Budget equation | `transfer_without_benefit`, `transfer_with_benefit` (= current `unemployment_transfer_paid`), `delta_transfer_savings`. |
| Net cost incl. transfer | Budget equation | `full_leave_net_cost_incl_transfer = B - ΔT - delta_transfer_savings`. |
| Aux | Budget equation | Add `delta_transfer_savings`, `full_leave_net_cost_incl_transfer` (scaled). |
| Optional returns | Government budget | Add optional params and extend return tuple for reporting (no change to expenditure formula). |
