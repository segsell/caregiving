# Fiscal Costs Table: Architecture, Budget Constraints, and Extension Proposals

## 1. What `task_create_fiscal_costs` Does

**File:** `src/caregiving/tables/publication/task_fiscal.py`
**Output:** `bld/tables/publication/fiscal_costs_caregiving_policies.tex`

This task produces a LaTeX table comparing three caregiving policy regimes:

| Row | Policy |
|-----|--------|
| 1 | **Baseline**: informal care cash benefits (Pflegegeld) — flat monthly benefit for informal caregivers |
| 2 | **Normal caregiving leave** (65%, German Elterngeld-style) with job retention |
| 3 | **Full caregiving leave** (100%, Norwegian-style) with job retention |

### Columns currently in the table

| # | Column | Unit | Description |
|---|--------|------|-------------|
| 1 | Policy | — | Label |
| 2 | Total cost (currency) | EUR | Total fiscal cost of the policy across all agents and periods (age ≤ end_age_caregiving) |
| 3 | N unique caregivers | count | Number of distinct agents who ever choose informal care in the caregiving window |
| 4 | Avg. caregiving years (cond. on caregivers) | years | Total caregiver-periods / N unique caregivers |
| 5 | Avg cost per caregiver (currency) | EUR | Total cost / N unique caregivers |
| 6 | Avg. monthly cost per caregiver per caregiving month | EUR | (Avg cost per caregiver) / 12 / (Avg caregiving years) |
| 7 | Avg. income tax per caregiver | EUR | Sum(income_tax × wealth_unit) over lagged_choice ∈ INFORMAL_CARE / N caregivers |
| 8 | Avg. income tax (single) per caregiver | EUR | Same, for income_tax_single |
| 9 | Avg. joint gross labor income per caregiver | EUR | Own + partner gross labor income |
| 10 | Avg. joint gross retirement income per caregiver | EUR | Own + partner gross pension |
| 11 | Avg. household unemployment benefits per caregiver | EUR | Unemployment floor benefit |
| 12 | Avg. total tax revenue per caregiver | EUR | income_tax + own_ssc + partner_ssc |
| 13 | Avg. net government budget per caregiver | EUR | total_tax_revenue − government_expenditures |

### How costs are computed

- **Baseline cost:** Sum of `max(care_benefits_and_costs, 0) × wealth_unit` over rows where `lagged_choice ∈ INFORMAL_CARE`. This is the **gross** cash benefit paid out by the government for informal care.
- **Leave cost:** Uses `compute_net_caregiving_leave_top_up_cost()`, which computes:
  ```
  net_cost = gross_top_up − (tax_with_top_up − tax_without_top_up)
  ```
  i.e., the gross leave benefit minus the income tax revenue the government recovers because the top-up raises the tax base. This is an **approximation** using single income tax (not household), so it's exact for singles and approximate for couples.

### Conditioning

All aggregation conditions on `lagged_choice ∈ INFORMAL_CARE` for the numerator (costs and outcome sums) and `choice ∈ INFORMAL_CARE` for the denominator (N unique caregivers and caregiver-periods). This conditions on *actual caregiving* rather than just care demand.

---

## 2. Budget Constraints of the Two Leave Policies

### 2.1 Normal Caregiving Leave (65%, German Elterngeld-Style)

**File:** `budget_equation_caregiving_leave_with_job_retention.py`

#### Benefit computation (`calc_caregiving_leave_top_up`)

Eligible: informal caregivers who are not retired. Benefit depends on `job_before_caregiving` (state tracking pre-caregiving employment):

| Prior job | Current state | Benefit |
|-----------|--------------|---------|
| None (0) | Any | 0 |
| PT (1) | Unemployed + caregiver | `clip(0.65 × net_PT_wage, lower_bound × 12, upper_bound × 12)` |
| PT (1) | PT or FT | 0 |
| FT (2) | Unemployed + caregiver | `clip(0.65 × net_FT_wage, lower_bound × 12, upper_bound × 12)` |
| FT (2) | PT + caregiver | `max(clip(0.65 × net_FT_wage, lb, ub) − labor_income_after_ssc, 0)` |
| FT (2) | FT | 0 |

Net wages are computed from the wage equation (hourly_wage × hours) after SSC deduction.

#### Tax treatment: Progressionsvorbehalt (§32b EStG)

The 65% benefit is **tax-free** but subject to Progressionsvorbehalt:
1. Compute the average tax rate as if the benefit were included in taxable income: `avg_rate = tax(taxable + benefit) / (taxable + benefit)`
2. Apply that average rate to taxable income only (excluding the benefit)
3. The benefit raises the marginal tax rate on other income without being taxed itself
4. Disposable income = taxable − tax + benefit

#### Key intermediate quantities

- `own_income_for_tax` = was_worker × labor_income_after_ssc + was_retired × retirement_income_after_ssc (excludes benefit)
- `tax_increase_from_progression` = income_tax_with_progression − income_tax_without_progression
- `normal_leave_net_cost` = caregiving_leave_top_up − tax_increase_from_progression (net fiscal cost)

#### Unemployment interaction

Unemployment is an income floor:
```
total_income = max(household_net_income + child_benefits, unemployment_benefits)
unemployment_transfer_paid = max(0, unemployment_floor − household_net_income − child_benefits)
```

#### Returned aux dict

| Key | Description |
|-----|-------------|
| `net_hh_income` | Total household income including interest and bequests |
| `hh_net_income_wo_interest` | Total income before interest |
| `interest` | Interest on assets |
| `joint_gross_labor_income` | Own + partner gross labor income |
| `joint_gross_retirement_income` | Own + partner gross pension |
| `gross_partner_income` | Partner gross labor income |
| `gross_partner_pension` | Partner gross pension |
| `gross_labor_income` | Own gross labor income |
| `gross_retirement_income` | Own gross pension |
| `bequest_from_parent` | Inheritance received |
| `gets_inheritance` | Binary: received inheritance |
| `caregiving_leave_top_up` | Gross leave benefit (65% replacement) |
| `own_income_after_ssc` | Own income including benefit (for reporting) |
| `child_benefits` | Child benefits |
| `formal_care_costs` | Formal care co-payments (negative) |
| `household_unemployment_benefits` | Unemployment floor amount |
| `unemployment_transfer_paid` | Actual transfer paid (max(0, floor − income)) |
| `tax_increase_from_progression` | Extra tax from Progressionsvorbehalt |
| `normal_leave_net_cost` | Net cost = gross top-up − tax increase |
| `income_tax` | Household income tax |
| `income_tax_single` | Income tax on own income only |
| `own_ssc` | Own social security contributions |
| `partner_ssc` | Partner social security contributions |
| `total_tax_revenue` | income_tax + own_ssc + partner_ssc |
| `government_expenditures` | child_benefits + care_benefits + unemp_transfer + leave_top_up |
| `net_government_budget` | total_tax_revenue − government_expenditures |

---

### 2.2 Full Caregiving Leave (100%, Norwegian-Style)

**File:** `budget_equation_full_caregiving_leave_with_job_retention.py`

#### Benefit computation (`calc_full_caregiving_leave_top_up`)

Eligible: informal caregivers who are not retired. Benefit = 100% wage replacement of earnings loss, capped at 6G (Norwegian social insurance ceiling):

| Prior job | Current state | Benefit |
|-----------|--------------|---------|
| None (0) | Any | 0 |
| PT (1) | Unemployed + caregiver | 100% × min(gross_PT_wage, 6G_cap) |
| PT (1) | PT or FT | 0 |
| FT (2) | Unemployed + caregiver | 100% × min(gross_FT_wage, 6G_cap) |
| FT (2) | PT + caregiver | max(capped_gross_FT − capped_gross_PT, 0) |
| FT (2) | FT | 0 |

Key difference from normal leave: benefit is based on **gross** wages (not net), replacement rate is 100% (not 65%), and it's capped at 6G instead of bounded.

#### Tax treatment: Taxable (not SSC)

The full leave benefit is **taxable income** (not tax-free):
1. `own_income_after_ssc` = labor_income + pension_income + caregiving_leave_top_up
2. Tax is computed on the full amount (including benefit)
3. The benefit is NOT subject to SSC

This means the government pays out the gross benefit but recovers income tax on it.

#### Key intermediate quantities

- `tax_attributable_to_benefit` = tax_with_benefit − tax_without_benefit
- `full_leave_net_cost` = caregiving_leave_top_up − tax_attributable_to_benefit
- `delta_transfer_savings` = unemployment_transfer_without_benefit − unemployment_transfer_with_benefit (crowd-out of unemployment)
- `full_leave_net_cost_incl_transfer` = gross_top_up − tax_on_benefit − transfer_savings

#### Returned aux dict

Same as normal leave, except replaces the Progressionsvorbehalt-specific keys with:

| Key | Description |
|-----|-------------|
| `tax_attributable_to_full_leave` | Income tax increase attributable to the benefit |
| `full_leave_net_cost` | Net cost = gross top-up − tax on benefit |
| `delta_transfer_savings` | Reduction in unemployment transfers (crowd-out) |
| `full_leave_net_cost_incl_transfer` | Net cost including transfer savings: gross − tax − ΔU |

---

### 2.3 Government Budget Components

Both leave policies and the baseline share the same structure for government budget:

```
Revenue  = income_tax + own_ssc + partner_ssc
Expenditure = child_benefits + max(care_benefits, 0) + unemployment_transfer + leave_benefit
Net budget = Revenue − Expenditure
```

The key difference: in the baseline, `leave_benefit = 0` and `care_benefits_and_costs` includes the informal care cash benefit. In the leave policies, `leave_benefit = caregiving_leave_top_up` and `care_benefits_and_costs = formal_care_costs` only (the cash benefit for informal care is replaced by the leave benefit).

---

## 3. Economically Interesting Extensions to the Fiscal Comparison

### 3.1 Tier 1 — High priority (core policy evaluation metrics)

#### (1) Net fiscal cost of the leave benefit (direct measure)
**Currently missing from the table but already computed in aux.**

- Normal leave: `normal_leave_net_cost` = gross_top_up − tax_increase_from_Progressionsvorbehalt
- Full leave: `full_leave_net_cost` = gross_top_up − tax_attributable_to_benefit
- Full leave incl. transfer: `full_leave_net_cost_incl_transfer` = gross − tax − ΔU

These are the **true net fiscal costs** of each leave policy. The current table uses `compute_net_caregiving_leave_top_up_cost()` which is an approximation. Reporting the model-internal net cost directly would be more accurate and allow decomposition.

**Columns to add:**
- Avg. net leave cost per caregiver (direct from aux)
- Avg. net leave cost incl. transfer savings per caregiver (full leave only; for normal leave, compute analogously)

#### (2) Decomposition of net fiscal cost
Break the net cost into its components to understand the financing structure:

| Component | Description |
|-----------|-------------|
| Gross benefit paid | `caregiving_leave_top_up` |
| Tax claw-back | `tax_increase_from_progression` (normal) or `tax_attributable_to_full_leave` (full) |
| Transfer savings (ΔU) | Reduction in unemployment transfer from income being above the floor |
| Net cost | Gross − tax claw-back − ΔU |

This shows what fraction of the gross cost the government recovers through taxation and reduced transfers.

#### (3) Difference-in-difference: policy vs baseline net government budget
The net_government_budget is already in the table per caregiver, but the **difference** (leave − baseline) per caregiver is the most policy-relevant number: how much worse/better is the government's fiscal position under the leave policy?

**Columns to add:**
- Δ net government budget per caregiver (leave − baseline)
- Δ total tax revenue per caregiver (leave − baseline)
- Δ government expenditures per caregiver (leave − baseline)

#### (4) Per-capita cost (over ALL agents, not just caregivers)
The current table conditions everything on caregivers. But the population-level fiscal burden is:
```
total_cost / N_total_agents
```
This answers: "What does this policy cost per person in the economy?"

**Column to add:**
- Total cost per capita (total cost / total agents)

### 3.2 Tier 2 — Medium priority (behavioral and distributional insights)

#### (5) Behavioral labor supply response
The leave policies change behavior: agents may supply more or less care, and their employment patterns change. Currently this is only partially captured through `joint_gross_labor_income`.

**Columns to add:**
- Share of agents ever providing informal care (= N caregivers / N total agents)
- Avg. employment rate during caregiving periods (share of caregiver-periods where choice ∈ WORK)
- Avg. full-time share during caregiving periods
- Avg. gross own labor income per caregiver (already in aux but not in table)

These capture whether the leave induces more caregiving (extensive margin) and how it changes employment composition during caregiving spells (intensive margin).

#### (6) Formal care substitution
If more generous leave induces more informal care, formal care utilization should fall. This is a key benefit of the policy (formal care is expensive).

**Columns to add:**
- Avg. formal care costs per caregiver
- Total formal care expenditure
- Δ formal care expenditure (leave − baseline)

#### (7) Revenue decomposition
Break total_tax_revenue into its components to understand where revenue changes come from:

**Columns to add (or separate sub-table):**
- Avg. income tax per caregiver (already in table)
- Avg. own SSC per caregiver (in aux, not in table)
- Avg. partner SSC per caregiver (in aux, not in table)
- Avg. government expenditures per caregiver (in aux, not in table)

#### (8) Unemployment interaction
How much does the leave benefit crowd out unemployment transfers?

**Columns to add:**
- Avg. unemployment transfer paid per caregiver
- Δ unemployment transfer (leave − baseline) — the transfer savings
- For full leave: `delta_transfer_savings` is already computed

#### (9) Cost by education group
Different education groups have different wages, so the leave benefit (which is wage-linked) varies. This is a distributional question.

**Separate table or panel:**
- All metrics split by education level (low/high)

### 3.3 Tier 3 — Lower priority but economically interesting

#### (10) Long-run pension effects
If leave preserves job attachment (via job retention), it preserves experience accumulation, which affects future pension income. Compare retirement income across policies.

**Column to add:**
- Avg. gross retirement income per ever-caregiver (over full lifecycle, not just caregiving periods)

#### (11) Net income effect on caregivers
What does the caregiver receive net?

**Columns to add:**
- Avg. own_income_after_ssc per caregiver (already in aux; includes benefit for normal leave, also includes for full leave)
- Avg. net household income per caregiver

#### (12) Cost-effectiveness ratios
These require combining cost data with outcome data:
- Cost per additional caregiving year induced (relative to baseline)
- Cost per EUR of informal care provided (requires valuing informal care)
- Cost per avoided formal care period

#### (13) Lifecycle vs caregiving-period costs
The current table sums over caregiving periods only (lagged_choice ∈ INFORMAL_CARE). But the policy also affects non-caregiving periods (e.g., through preserved job attachment, different pension accumulation). A lifecycle comparison (sum over all periods for agents who ever provide care) would capture these spillovers.

#### (14) Welfare measure
If utility values or value functions are accessible from the model solution, report:
- Avg. expected lifetime utility for ever-caregivers under each policy
- Compensating variation: how much would a caregiver need to be paid under baseline to be indifferent to the leave policy?

---

## 4. Ranking of All Metrics (Existing + Proposed)

Priority ranking for inclusion in a comprehensive fiscal comparison table:

| Rank | Metric | Status | Why |
|------|--------|--------|-----|
| 1 | **Net fiscal cost of leave** (direct from aux) | NEW | The single most important policy number |
| 2 | **Decomposition: gross benefit, tax claw-back, transfer savings** | NEW | Shows the financing mechanism |
| 3 | **Δ net government budget (leave − baseline)** | NEW | Policy impact on overall fiscal position |
| 4 | Total cost (currency) | EXISTING | Aggregate cost |
| 5 | N unique caregivers | EXISTING | Size of beneficiary pool |
| 6 | Avg cost per caregiver | EXISTING | Per-beneficiary cost |
| 7 | **Share of agents ever caregiving** | NEW | Extensive margin behavioral response |
| 8 | Avg. caregiving years | EXISTING | Duration |
| 9 | **Per-capita cost (all agents)** | NEW | Population-level fiscal burden |
| 10 | Avg. net government budget per caregiver | EXISTING | Fiscal position per caregiver |
| 11 | Avg. total tax revenue per caregiver | EXISTING | Revenue side |
| 12 | **Avg. employment rate during caregiving** | NEW | Labor supply response |
| 13 | Avg. joint gross labor income per caregiver | EXISTING | Income level |
| 14 | **Δ formal care expenditure** | NEW | Substitution effect |
| 15 | **Avg. unemployment transfer paid** | NEW | Transfer interaction |
| 16 | Avg. monthly cost per caregiving month | EXISTING | Monthly flow cost |
| 17 | **Cost by education group** | NEW | Distributional equity |
| 18 | Avg. income tax per caregiver | EXISTING | Tax detail |
| 19 | Avg. income tax (single) per caregiver | EXISTING | Tax detail (singles) |
| 20 | **Avg. own SSC per caregiver** | NEW (in aux) | Revenue decomposition |
| 21 | **Avg. partner SSC per caregiver** | NEW (in aux) | Revenue decomposition |
| 22 | Avg. joint gross retirement income | EXISTING | Pension level |
| 23 | Avg. household unemployment benefits | EXISTING | Transfer level |
| 24 | **Avg. own_income_after_ssc per caregiver** | NEW (in aux) | Net income to caregiver |
| 25 | **Lifecycle cost comparison** | NEW | Beyond-caregiving-spell effects |
| 26 | **Cost-effectiveness ratios** | NEW | Efficiency metrics |
| 27 | **Compensating variation / welfare** | NEW | Ultimate welfare comparison |

---

## 5. Proposed Additional Aux Objects from Budget Constraints

### 5.1 Currently missing from BOTH leave budget constraints

| Proposed key | Value | Rationale |
|-------------|-------|-----------|
| `labor_income_after_ssc` | was_worker × calc_labor_income_after_ssc result | Pure labor income net of SSC, without the leave benefit mixed in. Currently `own_income_after_ssc` includes the benefit in the normal leave case and both labor + benefit in the full leave case. Splitting allows decomposing income sources. |
| `retirement_income_after_ssc` | was_retired × calc_pensions_after_ssc result | Own pension income net of SSC. Currently folded into `own_income_after_ssc`. Useful for lifecycle pension analysis. |
| `partner_income_after_ssc` | calc_partner_income_after_ssc result | Partner's net income contribution. Useful for understanding household-level effects and who actually benefits from the policy (individual vs household). |
| `total_net_household_income` | calc_net_household_income disposable result | The household disposable income after tax but before child benefits and the unemployment floor. Shows the "pure" tax-and-transfer outcome. |
| `household_net_income_before_floor` | total_net_household_income + child_benefits | Income before the max() with unemployment benefits. Shows who is constrained by the unemployment floor. |

**Note:** `experience_years` is directly available in simulated DataFrames (column `experience`). `was_worker` and `was_retired` can be derived from `choice` / `lagged_choice` columns in the simulated data using the `is_working()` and `is_retired()` helpers. Similarly `own_gross_income` can be derived as `was_worker * gross_labor_income + was_retired * gross_retirement_income` from existing aux. These do NOT need to be added to aux.

### 5.2 Missing from normal leave only

| Proposed key | Value | Rationale |
|-------------|-------|-----------|
| `own_income_for_tax` | was_worker × labor_income_after_ssc + was_retired × retirement_income_after_ssc | The taxable own income EXCLUDING the 65% benefit. Already computed internally but not returned. Shows the base on which Progressionsvorbehalt operates. |
| `tax_without_progression` | calc_net_household_income(own_income_for_tax, ..., progression_income=0) | Tax that would be due without Progressionsvorbehalt. Already computed internally but not returned. Together with `tax_increase_from_progression` (already returned) allows full decomposition. |

### 5.3 Missing from full leave only

| Proposed key | Value | Rationale |
|-------------|-------|-----------|
| `disposable_without_benefit` | calc_net_household_income(own_income_without_benefit, ...) disposable | Disposable income if the leave benefit were zero. Already computed internally but not returned. Shows counterfactual income level. |
| `total_tax_without_benefit` | calc_net_household_income(own_income_without_benefit, ...) tax | Tax without the leave benefit. Already computed internally but not returned. Enables exact decomposition of tax_attributable_to_benefit. |
| `transfer_without_benefit` | max(0, unemployment_floor − income_without_benefit) | Unemployment transfer that would be paid without the leave benefit. Already computed internally but not returned. Enables exact delta_transfer_savings verification. |

### 5.4 Missing from baseline

| Proposed key | Value | Rationale |
|-------------|-------|-----------|
| `unemployment_transfer_paid` | max(0, unemployment_floor − total_net_household_income − child_benefits) | The baseline uses `household_unemployment_benefits` as the full floor, but doesn't separately track the actual transfer paid (as the leave policies do). Adding this would allow consistent comparison of unemployment transfers across all three policies. Currently the baseline government budget function uses `household_unemployment_benefits` directly rather than the actual top-up, which overstates unemployment expenditure when income is above zero. |
| `labor_income_after_ssc` | Same as leave policies | Not currently returned in baseline aux. |
| `retirement_income_after_ssc` | Same as leave policies | Not currently returned in baseline aux. |
| `experience_years` | Same as leave policies | Not currently returned in baseline aux. |

### 5.5 Priority for implementation

**Immediate (required for Tier 1 table extensions):**
1. Add `normal_leave_net_cost` and `full_leave_net_cost` / `full_leave_net_cost_incl_transfer` to the OUTCOME_COLUMNS used by the fiscal table (they're already in aux — just need to be added to the table aggregation).
2. Add `government_expenditures` to the table (already in aux).

**Short-term (for Tier 2 metrics):**
3. Add `labor_income_after_ssc`, `retirement_income_after_ssc`, `partner_income_after_ssc` to all three budget constraints.
4. Add `unemployment_transfer_paid` to baseline budget constraint (and government budget).
5. Add `experience_years` to all three budget constraints.

**Medium-term (for decomposition and distributional analysis):**
6. Add `own_income_for_tax` and `tax_without_progression` to normal leave aux.
7. Add `disposable_without_benefit` and `transfer_without_benefit` to full leave aux.
8. Add `was_worker`, `was_retired`, `own_gross_income` to all three aux dicts.

---

## 6. Summary: What to Do Next

### Phase 0: Quick wins (table-level changes only) — IMPLEMENTED

1. **DONE** — Added `normal_leave_net_cost`, `full_leave_net_cost`, `full_leave_net_cost_incl_transfer`, `tax_increase_from_progression`, `tax_attributable_to_full_leave`, `delta_transfer_savings`, `caregiving_leave_top_up`, `care_benefits_and_costs`, `formal_care_costs`, `unemployment_transfer_paid`, `own_ssc`, `partner_ssc`, `government_expenditures`, `own_income_after_ssc` to OUTCOME_COLUMNS.
2. **DONE** — Added `government_expenditures` to outcome columns.
3. **DONE** — Δ rows appended: "Delta Normal − Baseline" and "Delta Full − Baseline".
4. **DONE** — Per-capita cost column added (total cost / N total agents).
5. **DONE** — Share ever caregiving column added (N caregivers / N total agents).
6. **DONE** — Total gross benefit and Total net cost (from model aux) columns added.
7. **DONE** — N total agents column added.

### Phase 1: RAM optimization and life-cycle scope — IMPLEMENTED

8. **DONE** — Simulation DataFrames are now subsetted to `_REQUIRED_COLUMNS` immediately after loading via `_load_sim_df()`, reducing memory footprint substantially (from ~50+ columns to ~30).
9. **DONE** — Life-cycle scope extended from `end_age_caregiving` (70) to `end_age` (100). The caption now reads "life cycle ages 30–100". This captures pension income effects that persist well beyond the caregiving window. Key rationale: `max_ret_age` is 67, meaning retirement income accrues for 33 additional periods (ages 67–100). Truncating at age 70 missed most of the pension income stream.
10. **DONE** — Runtime assertion verifying `age == start_age + period` for all three DataFrames (confirms period and age evolve identically).

### Phase 2: Government formal care cost (social planner perspective) — IMPLEMENTED

11. **DONE** — Added government/social-planner formal care cost computation. This is *distinct* from the agent's co-payment (`formal_care_costs` in the budget constraint). The agent's cost is the nursing home insurance contribution; the government cost is the full cost of providing formal care.

#### Design and implementation

**Constants (module-level, easily adjustable):**

| Variable | Value | Unit | Description |
|----------|-------|------|-------------|
| `GOV_COST_NURSING_HOME` | 1,500 | EUR/month | Government cost per nursing home resident |
| `SHARE_PURE_FORMAL_HOME_CARE` | 0.1 | fraction | Share of "formal care" that is home care (not nursing home) |
| `PURE_FORMAL_HOME_CARE_COST` | 1,000 | EUR/month | Government cost per home care recipient |
| `GOV_ANNUAL_FORMAL_CARE_COST` | 17,400 | EUR/year | Weighted annual cost (derived, not set manually) |

**Weighted annual cost formula:**
```
GOV_ANNUAL_FORMAL_CARE_COST = (1 - 0.1) × 1500 × 12  +  0.1 × 1000 × 12
                             = 16,200 + 1,200
                             = 17,400 EUR/year
```

**Conditioning on `lagged_choice`:** Formal care costs are computed for agent-period pairs where `lagged_choice ∈ FORMAL_CARE` (choices [4,5,6,7]). This is correct because the budget constraint computes all income and costs in period t based on the decision made in t-1 (the lagged choice). The `formal_care_costs` variable in the budget constraint follows this same convention:
```python
formal_care = is_formal_care(lagged_choice)
annual_formal_care_costs_agent = -model_specs["formal_care_costs"][period] * formal_care * 12
```

**Why `lagged_choice` not `choice`:** In the budget constraint, `lagged_choice` is d_{t-1}. Income, taxes, benefits, and care costs in period t are all determined by what the agent did in t-1. The wage is from working in t-1, the pension is from being retired in t-1, and the formal care cost is from using formal care in t-1. Using `choice` (d_t) would be off by one period.

**Distinction from agent's formal care cost:**
- Agent's cost (`formal_care_costs` in aux): The insurance co-payment, computed as `-model_specs["formal_care_costs"][period] × is_formal_care(lagged_choice) × 12`. This varies by age (period-indexed array from estimated regression).
- Government's cost (`_total_gov_formal_care_cost`): The social planner's full cost of providing nursing home or home care, fixed per formal-care period. This is the resource cost to society, not the transfer within the insurance system.

**Table column:** "Gov. formal care cost (total)" — total government cost of formal care across all agent-periods, for each policy scenario. The Δ columns show how much the government saves on formal care when informal care is subsidized.

### Phase 3: Budget constraint aux additions — COMPLETED (user-side)

The user has confirmed that all short-term aux additions from §5 have been implemented in the budget constraint files and simulation data has been regenerated. The following are now available in simulated DataFrames:

- `labor_income_after_ssc` — all three policies
- `retirement_income_after_ssc` — all three policies
- `partner_income_after_ssc` — all three policies
- `unemployment_transfer_paid` — all three policies (baseline was missing before)
- `own_income_for_tax` / `own_income_for_tax_without_benefit` — normal / full leave
- `total_net_household_income` — all three policies
- `household_net_income_before_floor` — all three policies
- `tax_without_progression` — normal leave
- `disposable_without_benefit`, `total_tax_without_benefit`, `transfer_without_benefit` — full leave

### Phase 4: New task functions — IN PROGRESS

#### 4a. Education-stratified fiscal table
Split all metrics by education level (low/high). Requires adding an `education` column to `_REQUIRED_COLUMNS` and grouping the computation.

**Implementation approach:**
- Add `education` to `_REQUIRED_COLUMNS`.
- Create `task_create_fiscal_costs_by_education()` that calls the same helper functions but filters `df[df["education"] == edu]` before computing.
- Output: separate LaTeX table or panel with education groups as sub-columns.

#### 4b. Age-profile fiscal table
Show fiscal metrics by age group (e.g., 30-39, 40-49, 50-59, 60-69, 70-79, 80+). This reveals how costs and revenues shift across the life cycle.

**Implementation approach:**
- Create `task_create_fiscal_costs_by_age_group()` that bins ages and computes per-group.
- Particularly relevant for showing that retirement income effects extend beyond the caregiving window (ages 70+).

#### 4c. Cost-effectiveness ratios
Combine fiscal costs with behavioral outcomes:
- Cost per additional caregiving year induced (vs. baseline)
- Cost per avoided formal care period
- Cost per EUR of informal care value provided

**Implementation approach:**
- Requires valuation of informal care (e.g., replacement cost method).
- Compute Δ caregiving years × cost per additional year.

#### 4d. Lifecycle vs caregiving-period comparison
Currently all cost aggregations condition on `lagged_choice ∈ INFORMAL_CARE` (numerator) or `choice ∈ INFORMAL_CARE` (denominator). This captures only caregiving periods. A lifecycle comparison would sum over all periods for agents who *ever* provide care, capturing spillovers like preserved job attachment and pension accumulation.

**Implementation approach:**
- Identify ever-caregivers: `agents_who_ever_care = df[df["choice"].isin(care_choices)]["agent"].unique()`
- Filter full lifecycle for these agents: `df[df["agent"].isin(agents_who_ever_care)]`
- Sum outcomes over their full lifecycle, not just caregiving periods.
