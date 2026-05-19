# Caregiving Leave with Job Retention Model

This document describes how the **caregiving leave with job retention** counterfactual model is implemented and outlines what would be needed to implement a **limited** variant (caregiving leave restricted to at most 3 years, not necessarily consecutive).

---

## Part 1: Current Implementation

### 1.1 Entry point and wiring

- **Module:** `src/caregiving/model/task_specify_model_caregiving_leave_with_job_retention.py`
- **Pytask task:** `task_specify_model_caregiving_leave_with_job_retention` (marker: `caregiving_leave_with_job_retention_model`)
- **Outputs:**
  - `model_config_caregiving_leave_with_job_retention.pkl`
  - `model_caregiving_leave_with_job_retention.pkl`

The task loads derived specs from `specs_full.pkl`, builds a `model_config` (state space, choices, grids), and calls `dcegm.setup_model()` with:

- **State space:** `create_state_space_functions()` from `state_space_caregiving_leave_with_job_retention`
- **Stochastic transitions:** including `job_offer_process_transition_leave_with_job_retention` for the job-offer process
- **Budget:** `budget_constraint` from `budget_equation_caregiving_leave_with_job_retention`
- **Experience:** `get_next_period_experience_caregiving_leave` from `experience_caregiving_leave_model`

So the caregiving-leave-with-job-retention logic is assembled here; the actual behavior lives in the state space, job transition, budget, and experience modules.

---

### 1.2 State space

**Module:** `src/caregiving/model/state_space_caregiving_leave_with_job_retention.py`

- **Purpose:** Same as the job-retention state space, but in a separate module so caregiving-leave-specific rules can be added without touching the original job-retention code.

**Deterministic states (relevant to leave):**

- **`job_before_caregiving`** (values 0, 1, 2):
  - **0:** No job when caregiving started (or not currently caregiving; state is reset to 0 when not in informal care).
  - **1:** Part-time job when caregiving started.
  - **2:** Full-time job when caregiving started.

  This is the only extra deterministic state compared to a baseline without job retention. It is used to:
  - Grant job retention (no separation, guaranteed job offer when on “leave”) only to those who had a job when they started caregiving.
  - Define “on caregiving leave” for experience and budget (e.g. unemployed or part-time while caregiving, with prior job).

**State space functions:**

- **`state_specific_choice_set`:** `state_specific_choice_set_with_caregiving` (shared with baseline/caregiving models).
- **`next_period_deterministic_state`:** `next_period_deterministic_state_with_job_retention`
  - Updates `period`, `lagged_choice`, `already_retired`, and **`job_before_caregiving`**.
  - When the agent **starts** a caregiving spell (`just_started_care`), `job_before_caregiving` is set from last period’s work status (0 / 1=PT / 2=FT).
  - When **not** in informal care, `job_before_caregiving` is reset to 0 so a future spell is re-evaluated.
  - When **continuing** in care, `job_before_caregiving` is unchanged.
- **`next_period_experience`:** `get_next_period_experience_caregiving_leave` (see below).
- **`sparsity_condition`:** `sparsity_condition_with_job_retention`
  - Encodes admissibility of (period, states): retirement age, caregiving age window, caregiving_type, and consistency of `job_before_caregiving` (e.g. if not in informal care, `job_before_caregiving` must be 0).
  - Also handles proxy states (e.g. death, mother longer dead, out-of-window care) and sets `job_before_caregiving` to 0 where care is not relevant.

There is **no** cap on how many years an agent can be on caregiving leave; the only tracking is “did they have a job (and which type) when the current spell started.”

---

### 1.3 Job offer transition (leave-with-job-retention)

**Module:** `src/caregiving/model/stochastic_processes/job_transition_job_retention.py`
**Function:** `job_offer_process_transition_leave_with_job_retention(params, model_specs, education, period, choice, job_before_caregiving)`

- **Inputs:** Parameters, specs, education, period, **current choice** `choice`, and **deterministic state** `job_before_caregiving`.
- **Output:** Two-element probability vector for next period `job_offer` (0 = no offer, 1 = offer).

**Policy logic:**

1. **Job separation**
   - If the agent is in **informal care** and **had a job before caregiving** (`job_before_caregiving` ∈ {1, 2}): separation probability is set to **0**.
   - Otherwise: use baseline `job_sep_prob` from `model_specs["job_sep_probs"]`.

2. **“Caregiving leave” definition**
   - In care **and** (unemployed **or** part-time **or** full-time).
   - Retired caregivers are not treated as on leave for the job-offer adjustment (retirement already forces prob_job = 0).

3. **Job finding**
   - If on **caregiving leave** **and** had a job before caregiving: **job finding probability = 1.0** (guaranteed job offer when they exit leave).
   - Otherwise: baseline `job_finding_prob` (from `calc_job_finding_prob_women_linear`).

So the current model gives **unlimited** job retention and guaranteed re-employment for anyone on caregiving leave who had a job when the spell started; there is no notion of “years of leave used.”

---

### 1.4 Budget and government components

**Module:** `src/caregiving/model/wealth_and_budget/budget_equation_caregiving_leave_with_job_retention.py`

- **Budget constraint** uses:
  - Standard income components (pensions, unemployment, labor income, partner, etc.).
  - **Caregiving leave top-up:** `calc_caregiving_leave_top_up(..., job_before_caregiving=job_before_caregiving, ...)` so that leave benefits can depend on prior job type.
- Government budget components for this counterfactual: `calc_government_budget_components_caregiving_leave_with_job_retention` (from `government_budget_caregiving_leave_with_job_retention`).

There is no argument or state for “years of leave already used”; eligibility is based only on current caregiving status and `job_before_caregiving`.

---

### 1.5 Experience

**Module:** `src/caregiving/model/experience_caregiving_leave_model.py`
**Function:** `get_next_period_experience_caregiving_leave(period, lagged_choice, already_retired, partner_state, education, experience, job_before_caregiving, model_specs)`

**Definition of “on caregiving leave” (for experience only):**
Caregiver, not retired, and either **unemployed** or **part-time with prior full-time** (`job_before_caregiving == 2`). This is the same eligibility as the leave top-up (and distinct from the job-transition “leave,” which also includes full-time caregivers for guaranteed job offer).

- **Freeze:** In periods when the agent is “on caregiving leave,” experience **does not** follow the current choice (e.g. unemployed → 0, part-time → part-time credit). Instead the period’s credit equals what they would get if they had stayed in **job_before_caregiving**: 0 → 0, 1 (PT) → part-time credit, 2 (FT) → 1.0. So pension-relevant experience grows as if they had kept the pre-care job.
- **No cap:** There is no limit on how many years this rule applies; every year on leave (under the above definition) gets the same treatment.

**Who gets what experience:**

| Current choice while caregiving | Prior job (`job_before_caregiving`) | On leave (for experience)? | Experience credit this period |
|--------------------------------|-------------------------------------|----------------------------|------------------------------|
| Unemployed                     | any (0, 1, 2)                       | Yes                        | Frozen: 0 / PT credit / 1.0  |
| Part-time                      | Full-time (2)                       | Yes                        | Frozen: 1.0                  |
| Part-time                      | Part-time (1) or none (0)           | No                         | Baseline PT (or 1.0 if intensive care) |
| Full-time                      | any                                 | No                         | Baseline: 1.0                |
| Retired                        | —                                   | No                         | Baseline retirement logic    |

So **full-time workers can be caregiving**, but for experience they are **never** “on leave”: they get normal full-time credit (1.0). Only **unemployed** caregivers and **part-time-with-prior-FT** caregivers are “on leave” for experience and get the frozen pre-care path.

---

### 1.6 Summary of current design

| Component | Role |
|-----------|------|
| **task_specify_model_caregiving_leave_with_job_retention** | Assembles model: state space, transitions, budget, experience; registers `job_before_caregiving` in `model_config["deterministic_states"]` and uses `job_offer_process_transition_leave_with_job_retention`. |
| **state_space_caregiving_leave_with_job_retention** | Defines `job_before_caregiving` (0/1/2), its transition (set at spell start, reset when not in care), choice set, experience hook, and sparsity/proxies. |
| **job_transition_job_retention.job_offer_process_transition_leave_with_job_retention** | Zero separation and guaranteed job offer for caregivers on leave who had a job before caregiving; no time limit. |
| **budget_equation_caregiving_leave_with_job_retention** | Budget and leave top-up conditional on `job_before_caregiving`; no years-used limit. |
| **experience_caregiving_leave_model** | Experience freeze on caregiving leave using `job_before_caregiving`; no years-used limit. |

---

## Part 2: Limited Caregiving Leave (max 3 years, not necessarily consecutive)

**Goal (for later implementation):** A variant where caregiving leave with job retention is **limited to at most 3 years** over the lifecycle, and those years **do not need to be consecutive**. After 3 years of leave are “used,” the agent no longer gets the job-retention / guaranteed job-offer benefit (and possibly no leave top-up), but can still choose to provide care.

---

### 2.1 New deterministic state: years of leave used

- **Name (suggestion):** `years_caregiving_leave_used`
- **Values:** `0, 1, 2, 3`
- **Meaning:** Number of periods (years) the agent has already spent on “caregiving leave” (under the same definition as today: caregiver, not retired, and either unemployed or part-time with prior full-time).
- **Transition (conceptual):**
  - If **not** on caregiving leave this period: **no change** (state stays as is).
  - If **on** caregiving leave this period: **increment by 1**, but **cap at 3** (min(current + 1, 3)).
  - When **not** in informal care, we do **not** reset this state (unlike `job_before_caregiving`), because the 3-year cap is lifetime.

So we need a new state transition that:
- Takes `years_caregiving_leave_used`, `choice`, `lagged_choice`, `job_before_caregiving`, and (if needed) `already_retired`.
- Computes “on caregiving leave” the same way as in the job transition and experience modules.
- Returns `min(years_caregiving_leave_used + 1, 3)` when on leave, else `years_caregiving_leave_used`.

---

### 2.2 New state space module

- **Proposed file:** `src/caregiving/model/state_space_limited_caregiving_leave_with_job_retention.py`
- **Content (conceptual):**
  - **Copy / adapt** from `state_space_caregiving_leave_with_job_retention.py`:
    - Keep `state_specific_choice_set`, same sparsity structure, and same proxy logic where appropriate.
    - **Add** `years_caregiving_leave_used` to the deterministic state (values 0–3).
    - **New** `next_period_deterministic_state_limited_leave`: same as current `next_period_deterministic_state_with_job_retention`, **plus** update of `years_caregiving_leave_used` as above (increment on leave, cap at 3, no reset when not in care).
    - **Sparsity:** No need to exclude any (period, state) solely because of `years_caregiving_leave_used`; the cap only affects **policies** (job offer, benefits). Optionally one could restrict choices when `years_caregiving_leave_used == 3` (e.g. no “leave” option), but that can also be handled in the job transition and budget by treating “leave with benefit” as unavailable.
  - **Proxy states:** In all proxy state dicts, set `years_caregiving_leave_used` to a consistent value (e.g. 0 when care is not relevant, or carry through where it matters for child states).
  - Export `create_state_space_functions()` pointing to the new transition and sparsity functions.

This keeps the “limited” logic in one place and leaves the original caregiving-leave-with-job-retention model unchanged.

---

### 2.3 Job offer transition

- **Option A – New function in same module:**
  Add e.g. `job_offer_process_transition_limited_leave_with_job_retention(params, model_specs, education, period, choice, job_before_caregiving, years_caregiving_leave_used)` in `job_transition_job_retention.py`.
  - **Signature:** Same as current leave-with-job-retention, plus `years_caregiving_leave_used`.
  - **Logic:**
    - If `years_caregiving_leave_used >= 3`: do **not** apply the retention policy (use baseline separation and job-finding probabilities), even if currently on caregiving leave and `job_before_caregiving` ∈ {1, 2}.
    - If `years_caregiving_leave_used < 3`: same as current `job_offer_process_transition_leave_with_job_retention` (zero separation, guaranteed job offer when on leave and had job before caregiving).

- **Option B – New module:**
  If you prefer to keep the limited policy separate, a new small module (e.g. `job_transition_limited_caregiving_leave.py`) could wrap or duplicate the logic with the `years_caregiving_leave_used` check.

The transition function must receive `years_caregiving_leave_used` in the state vector so the model’s transition dispatcher passes it through; the task that builds the model must register this state and use the new transition.

---

### 2.4 Task and model config

- **New task (suggestion):** e.g. `task_specify_model_limited_caregiving_leave_with_job_retention` in a new file or alongside the current one (e.g. `task_specify_model_limited_caregiving_leave_with_job_retention.py`).
- **Model config:**
  - **deterministic_states:** Add `"years_caregiving_leave_used": np.arange(4, dtype=int)` (0, 1, 2, 3).
  - **State space:** Use `create_state_space_functions()` from `state_space_limited_caregiving_leave_with_job_retention`.
  - **Stochastic transitions:** Use the new job-offer transition that conditions on `years_caregiving_leave_used` (e.g. `job_offer_process_transition_limited_leave_with_job_retention`).
- **Outputs:** e.g. `model_config_limited_caregiving_leave_with_job_retention.pkl` and `model_limited_caregiving_leave_with_job_retention.pkl`.

---

### 2.5 Budget and top-up

- **Leave top-up:** If the policy is “no leave benefit after 3 years,” then `calc_caregiving_leave_top_up` (or a limited-leave-specific version) should take `years_caregiving_leave_used` and return 0 when `years_caregiving_leave_used >= 3` (and otherwise use the same rule as now, given `job_before_caregiving`).
- **Budget constraint:** The limited-leave budget function must accept `years_caregiving_leave_used` and pass it into the top-up (and any other leave-related transfers). So either:
  - a new `budget_equation_limited_caregiving_leave_with_job_retention.py`, or
  - a shared budget with an extra argument, used by the limited-leave task.
- **Government budget:** If leave benefits are capped at 3 years, the government budget for this counterfactual should use the same cap (e.g. a dedicated `calc_government_budget_components_limited_caregiving_leave_with_job_retention` or an extra argument in the existing one).

---

### 2.6 Experience

- **Current:** `get_next_period_experience_caregiving_leave` does not take any “years used” state; it only uses “on caregiving leave” and `job_before_caregiving`.
- **Limited variant:** Two possible approaches:
  - **Option A:** Keep the same experience function. Then experience is still “frozen” on leave even after 3 years; only job offer and (optionally) cash benefits are withdrawn.
  - **Option B:** Pass `years_caregiving_leave_used` into experience and **do not** freeze experience when `years_caregiving_leave_used >= 3` (treat as normal unemployment/part-time). That would require a new or extended experience function and possibly a new experience module for the limited model.

Document the chosen assumption (A or B) so that calibration and policy interpretation are clear.

---

### 2.7 Summary: what would need to be done

| Component | Change |
|-----------|--------|
| **State space** | New module `state_space_limited_caregiving_leave_with_job_retention.py`: add `years_caregiving_leave_used` ∈ {0,1,2,3}, transition that increments on leave (capped at 3) and does not reset when not in care; update sparsity/proxies to include the new state. |
| **Task / model config** | New task (and possibly new task file) that builds the model with the new state space, adds `years_caregiving_leave_used` to `deterministic_states`, and uses the new job-offer transition (and, if needed, new budget/experience). |
| **Job transition** | New transition function (e.g. in `job_transition_job_retention.py`) that takes `years_caregiving_leave_used` and applies retention + guaranteed job offer only when `years_caregiving_leave_used < 3`; otherwise use baseline job dynamics. |
| **Budget / top-up** | Budget and government budget for the limited counterfactual must take `years_caregiving_leave_used` and restrict leave top-up (and any other leave benefits) to at most 3 years. |
| **Experience** | Decide: either keep current rule (freeze even after 3 years) or add a limited-leave-specific experience update that stops freezing after 3 years. |

No code has been changed in this step; the above is a specification for a future implementation of the limited caregiving leave with job retention model.

---

## Part 3: Integrating Pflegegeld (baseline care benefits) with the caregiving leave top-up

**Goal:** Allow the caregiving leave with job retention model (current or limited variant) to include **both** (i) **Pflegegeld** from the baseline model and (ii) the **caregiving leave top-up** in the same budget. This can be done **independently** of the year restriction (i.e. for the current unlimited leave model) or **together** with the limited (max 3 years) variant.

### 3.1 Current situation

- **Baseline** (`budget_equation.py`): Uses `calc_care_benefits_and_costs(period, lagged_choice, model_specs)` from `transfers.py`. That function returns **Pflegegeld** (informal care cash benefits: light/intensive from `informal_care_cash_benefits_light` / `informal_care_cash_benefits_intensive`) **minus** formal care costs. The result is added to the household budget (e.g. in `total_income_plus_interest`) and passed to the government budget as `care_benefits_and_costs`.

- **Caregiving leave with job retention** (`budget_equation_caregiving_leave_with_job_retention.py`): Does **not** call `calc_care_benefits_and_costs`. It only uses `calc_caregiving_leave_top_up(...)` and applies **formal care costs only** (no Pflegegeld). The comment in the code states that the caregiving leave top-up *replaces* informal care cash benefits. So currently caregivers on leave get the leave top-up and face formal care costs, but do **not** receive Pflegegeld.

### 3.2 Desired integration

- **Add Pflegegeld alongside the leave top-up:** In the caregiving leave budget (and, if applicable, in the limited-leave budget), call `calc_care_benefits_and_costs(period, lagged_choice, model_specs)` in the same way as in the baseline, and:
  - Add the result to the household’s total resources (e.g. to the same place where `total_income_plus_interest` is built, i.e. add `care_benefits_and_costs` in addition to `annual_formal_care_costs_agent` / leave top-up logic).
  - Pass the full `care_benefits_and_costs` (Pflegegeld net of formal care costs) into the government budget component function (e.g. `calc_government_budget_components_caregiving_leave_with_job_retention` or the limited-leave counterpart) so that government revenue/expenditure accounts for both Pflegegeld and the leave top-up.

- **Interaction with formal care costs:** In the baseline, `calc_care_benefits_and_costs` already nets out formal care costs (benefits minus costs). The caregiving leave budget currently adds `annual_formal_care_costs_agent` (formal care costs only) to total resources (as a negative). When integrating Pflegegeld, one must avoid double-counting formal care costs: either (a) use the baseline’s single `care_benefits_and_costs` (which already includes minus formal care costs) and then add the leave top-up and any other leave-specific terms, or (b) add Pflegegeld (positive), subtract formal care costs once, and add leave top-up. The baseline’s `calc_care_benefits_and_costs` already does (benefits − formal costs); reusing it keeps the same convention.

- **Optional: cap Pflegegeld when leave is limited:** If implementing the limited (3-year) variant, one can optionally restrict Pflegegeld to the same 3 years of “leave used” (e.g. pass `years_caregiving_leave_used` into a wrapper that returns 0 when `years_caregiving_leave_used >= 3`). Alternatively, Pflegegeld can remain as in the baseline (no cap), with only the leave top-up and job-retention benefits capped at 3 years. The document does not prescribe one; document the chosen assumption.

### 3.3 Where to change (conceptual)

| Component | Change |
|-----------|--------|
| **Budget** | In `budget_equation_caregiving_leave_with_job_retention.py` (and, if added, in the limited-leave budget module): import and call `calc_care_benefits_and_costs(period, lagged_choice, model_specs)`; add the result to the expression that defines `total_income_plus_interest` (or equivalent); pass the resulting `care_benefits_and_costs` into the government budget function instead of (or in addition to) passing only formal care costs. |
| **Government budget** | Ensure `calc_government_budget_components_caregiving_leave_with_job_retention` (and the limited-leave version, if any) accepts and uses `care_benefits_and_costs` in the same way as the baseline government budget (e.g. for care-related expenditures/revenue). Leave top-up remains a separate argument/component. |

This integration is **independent** of Part 2 (year restriction): it can be implemented for the current unlimited leave model first, and then the same budget logic can be reused or extended for the limited-leave variant (with or without capping Pflegegeld at 3 years).

---

## Part 4: Full vs Partial Leave — policy detail, mapping to the model, and implementation

### 4.1 The advisory-board recommendation (Familienpflegezeit)

The independent advisory board recommends a **Familienpflegezeit** of up to **36 months** per care-dependent person, split into two types of leave:

| Leave type | Hours/week | Max duration (months) |
|---|---|---|
| **Full leave** (Vollständige Freistellung) | 0 hours | 6 months |
| **Partial leave** (Teilweise Freistellung) | ≥ 15 h (first 6 months: can be < 15 h) and ≤ 32 h | 30 months |

Additional rules:
- **Total cap:** 36 months (regardless of combination of full/partial).
- **End-of-life exception:** Up to 3 months additional full leave for end-of-life care (ignored below because the model does not separately model end-of-life care).
- **Ordering is flexible:** The caregiver can mix full and partial leave in any order, and can split the leave into up to 3 segments.
- **Wage replacement (Familienpflegegeld):** Paid during both full and partial leave, as long as hours are within the above limits.
- The board recommends a **32-hour** upper bound for partial leave (synchronised with the Elterngeld limit), replacing the previous 30-hour threshold.

### 4.2 Mapping from the model's discrete work levels to the policy

The model has three discrete work states (plus retirement):

| Model choice | Weekly hours | Policy category |
|---|---|---|
| **Unemployed** (0 hours) | 0 | **Full leave** — within the 0 h requirement |
| **Part-time** | 20 h (WEEKLY_HOURS_PART_TIME in shared.py) | **Partial leave** — above the 15 h minimum, below the 32 h cap |
| **Full-time** | 40 h (WEEKLY_HOURS_FULL_TIME in shared.py) | **NOT eligible** for Familienpflegegeld — exceeds 32 h |

The mapping is clean:
- **Unemployed caregiver → full leave.** 0 h/week satisfies the full-leave definition.
- **Part-time caregiver → partial leave.** 20 h/week is between 15 and 32, satisfying partial-leave requirements.
- **Full-time caregiver → not on leave** for Familienpflegegeld purposes. 40 h/week exceeds the 32 h cap, so no wage replacement.

### 4.3 How the current (unlimited) implementation already maps to full vs partial leave

The current code does **not** use the labels "full leave" and "partial leave," but it **implicitly** distinguishes them through the agent's work choice:

**`calc_caregiving_leave_top_up` in `caregiving_leave_top_up.py` (65% model):**

| Prior job (`job_before_caregiving`) | Current choice while caregiving | Label (implicit) | Benefit |
|---|---|---|---|
| FT (2) | Unemployed | **Full leave** | 65% of prior FT net wage (bounded) |
| FT (2) | Part-time | **Partial leave** | 65% of prior FT net wage (bounded) minus current PT income |
| PT (1) | Unemployed | **Full leave** | 65% of prior PT net wage (bounded) |
| PT (1) | Part-time | Not on leave | 0 (same hours as before) |
| None (0) | Any | Not eligible | 0 |
| Any | Full-time | Not on leave | 0 (exceeds 32 h threshold) |
| Any | Retired | Not eligible | 0 |

**`calc_full_caregiving_leave_top_up` (100% Norwegian-style model):**

Same structure but with 100% gross-wage replacement (capped at 6G) instead of 65% net replacement.

**Experience (`get_next_period_experience_caregiving_leave`):**

`on_caregiving_leave` is defined as: caregiver, not retired, AND (unemployed OR (part-time with prior FT)). This captures both full leave (unemployed) and partial leave (PT with prior FT). In both cases, experience is frozen at the pre-caregiving path.

**Job transition (`job_offer_process_transition_leave_with_job_retention`):**

`caregiving_leave` is defined as: caregiver AND (unemployed OR part-time OR full-time), i.e. all non-retired caregivers. If on caregiving leave AND had a prior job: zero separation + guaranteed job offer. This is broader than the benefit eligibility (includes FT caregivers for job retention, even though they get no wage replacement). This is arguably correct: the job-retention right protects the employment relationship even when the caregiver is currently full-time; only the **wage replacement** is restricted to ≤ 32 h.

**Summary:** The current implementation implicitly treats:
- **Unemployed** caregivers as on **full leave** (0 h, gets full wage replacement).
- **Part-time** caregivers with prior FT as on **partial leave** (20 h, gets gap benefit).
- **Full-time** caregivers as **not on leave** for benefit purposes (40 h > 32 h threshold, no wage replacement).

This is consistent with the advisory-board recommendation, **except** that there is currently no time limit on either type.

### 4.4 Implementation: limited leave with full vs partial distinction (3-year cap)

**Discretisation to annual periods:**

| Policy (months) | Model (years) | Rounding |
|---|---|---|
| Full leave: max 6 months | **Max 1 year** | Round up (6 → 12 months = 1 year) |
| Partial leave: max 30 months | **Max 2 years** | Round down (30 → 24 months = 2 years, to keep total = 3) |
| **Total: max 36 months** | **Max 3 years** | Exact |

#### 4.4.1 New deterministic states

Two new state variables are needed (on top of `job_before_caregiving`):

1. **`years_leave_used_total`** ∈ {0, 1, 2, 3}
   Total number of years the agent has been on any form of caregiving leave (full or partial), accumulated over the lifecycle. Incremented by 1 each year the agent is on leave; capped at 3. **Not** reset when the agent exits care (lifetime cap).

2. **`full_leave_year_used`** ∈ {0, 1}
   Binary indicator: has the agent used their 1 year of full leave (unemployed while caregiving with prior job)? Set to 1 the first time the agent spends a full year on full leave; stays at 1 forever. **Not** reset.

**Valid state combinations:**

| `years_leave_used_total` | `full_leave_year_used` | Valid? | Meaning |
|---|---|---|---|
| 0 | 0 | Yes | No leave used |
| 1 | 0 | Yes | 1 year of partial leave used |
| 1 | 1 | Yes | 1 year of full leave used |
| 2 | 0 | Yes | 2 years of partial leave used |
| 2 | 1 | Yes | 1 full + 1 partial |
| 3 | 0 | Yes | 3 years of partial leave, no full leave ever taken |
| 3 | 1 | Yes | 1 full + 2 partial = max |
| 0 | 1 | No | Cannot have used full leave with 0 total |

**Design choice — partial-leave sub-cap (resolved):**

There is **no** separate sub-cap on partial leave. The only sub-cap is on full leave (max 1 year). The **total 3-year cap** is the binding overall constraint. This means:
- An agent who **never** takes full leave can use up to **3 years** of partial leave.
- An agent who uses 1 year of full leave can use up to **2 years** of partial leave.

This is consistent with the advisory board's design: "the absolute total limit for receiving the Familienpflegegeld is 36 months." The 6-month (→ 1-year) full-leave allowance and the 30-month (→ 2-year discretised) partial-leave allowance are not independent quotas — they describe the **composition** of the 36-month total. An agent who forgoes full leave can use the entire 36 months (3 years) as partial leave.

Under Option B, the full-leave sub-cap (max 1 year) is the only sub-cap. All 7 valid combinations in the table above are reachable.

The state space expansion is: 4 values × 2 values = 8 combinations, minus 1 invalid (0, 1) = **7 valid states**. Sparsity can exclude (0, 1).

#### 4.4.2 State transitions

In `next_period_deterministic_state` (new limited-leave module), at the end of each period:

```
Define "on leave this period":
  on_full_leave  = caregiver AND not retired AND unemployed AND had_job_before_caregiving
  on_partial_leave = caregiver AND not retired AND part_time AND had_ft_job_before_caregiving

still_eligible_for_full = (years_leave_used_total < 3) AND (full_leave_year_used == 0)
still_eligible_for_partial = (years_leave_used_total < 3)

actually_on_full_leave  = on_full_leave AND still_eligible_for_full
actually_on_partial_leave = on_partial_leave AND still_eligible_for_partial AND NOT actually_on_full_leave

on_any_leave = actually_on_full_leave OR actually_on_partial_leave

Update:
  years_leave_used_total  = min(years_leave_used_total + on_any_leave, 3)
  full_leave_year_used    = max(full_leave_year_used, actually_on_full_leave)
```

Note: `actually_on_full_leave` and `actually_on_partial_leave` determine whether benefits are paid **and** whether the leave counter increments. If the agent is in the "full leave" work state (unemployed caregiver) but has already used their full-leave year, they are **not** counted as on full leave for benefits — but they **could** still count as on partial leave if `years_leave_used_total < 3` (debatable: an unemployed agent works 0 h, which is below the 15 h partial-leave minimum → they should **not** receive partial-leave benefits either). Under the strict reading, an agent who has exhausted full leave and is unemployed while caregiving gets **no wage replacement** and **no leave counted** — they are treated like a baseline unemployed caregiver. This is the recommended interpretation.

#### 4.4.3 Wage replacement (top-up) function

Extend `calc_caregiving_leave_top_up` (or create a limited-leave variant) to accept `years_leave_used_total` and `full_leave_year_used`:

```
eligible_for_full_leave_benefit  = on_full_leave AND (full_leave_year_used == 0) AND (years_leave_used_total < 3)
eligible_for_partial_leave_benefit = on_partial_leave AND (years_leave_used_total < 3)

If eligible_for_full_leave_benefit:
  benefit = bounded 65% of prior wage (same as current unemployed-caregiver case)
Elif eligible_for_partial_leave_benefit:
  benefit = bounded 65% of prior FT wage minus current PT income (same as current FT→PT case)
Else:
  benefit = 0
```

The **amount** of the benefit is unchanged from the current `calc_caregiving_leave_top_up`; only the **eligibility** is gated by the two new states.

#### 4.4.4 Job transition

Extend the job-offer transition to condition on `years_leave_used_total`:

- If `years_leave_used_total < 3`: current retention rules apply (zero separation, guaranteed job offer for caregivers with prior job).
- If `years_leave_used_total >= 3`: baseline job dynamics (no retention). The agent's leave entitlement is exhausted; they no longer receive the protected-employment benefit.

The full vs partial distinction is **not** needed for job retention — only the total cap matters. The rationale: the Familienpflegezeit protects the employment relationship for the full 36-month duration, regardless of how the months are split between full and partial leave.

#### 4.4.5 Experience

Same logic as in Part 2 (Section 2.6), but now gated by `years_leave_used_total` and `full_leave_year_used`:

- **If on full leave AND eligible** (full_leave_year_used == 0, total < 3): freeze experience at prior job level.
- **If on partial leave AND eligible** (total < 3): freeze experience at prior FT level.
- **If NOT eligible** (total ≥ 3, or full leave exhausted and unemployed): baseline experience rules (no freeze).

#### 4.4.6 Summary table

| Component | State inputs | Change vs current unlimited model |
|---|---|---|
| **State space** | `years_leave_used_total` ∈ {0..3}, `full_leave_year_used` ∈ {0,1} | Two new deterministic states; transitions increment on eligible leave, cap at 3/1 respectively; sparsity excludes (total=0, full=1). |
| **Wage replacement** | `years_leave_used_total`, `full_leave_year_used` | Gate full-leave benefit on `full_leave_year_used == 0` AND `total < 3`; gate partial-leave benefit on `total < 3`. Amounts unchanged. |
| **Job retention** | `years_leave_used_total` | Apply retention only when `total < 3`; baseline dynamics otherwise. Full/partial distinction not needed here. |
| **Experience** | `years_leave_used_total`, `full_leave_year_used` | Freeze only when eligible for leave (same gates as wage replacement). |
| **Budget / government** | Passes through the gated benefit | Same structure as current; only the benefit amount changes (can be 0 when exhausted). |

No code has been changed; this is a specification only.
