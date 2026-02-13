# Intra-family caregiving statistics: analysis and implementation plan

## 1. Request summary

Create a new descriptives module `task_intra_family_caregiving.py` with a task `task_intra_family_statistics()` that produces a **LaTeX table** using **SHARE data**, with the following statistics:

| # | Statistic | Data source | Conditioning |
|---|-----------|--------------|---------------|
| **1** | Share of **female caregivers** that care for **both parents simultaneously** (mother AND father) | SHARE estimation data | — |
| **2** | Among individuals (parent–child sample) where **partner is alive** (i.e. has spouse/partner): share of informal care **from spouse/partner** and share **from (at least one) child** | SHARE parent–child sample | Partner alive |
| **3** | Among families with **at least one informal caregiver** and **at least two (adult) children**: share with **multiple caregiving adult children** (i.e. ≥2 children provide care) | SHARE parent–child sample | ≥1 caregiver, ≥2 children |

For each of 1–3, produce numbers in **two variants**:

- **Informal care in general**: no restriction on daily intensity (any frequency of help).
- **Daily caregiving**: conditioning on / restricting to **daily** informal care where applicable.

---

## 2. Data sources and current variables

### 2.1 Estimation data (`task_create_estimation_data_set.py`)

- **Output**: `BLD / "data" / "share_estimation_data.csv"` (and weight variants).
- **Population**: **Females only** (filter `dat["gender"] == FEMALE` before save), so all rows are women.
- **Relevant variables** (from `check_share_informal_care_to_mother_father` and related):
  - **`care_to_mother`**: 1 if respondent gives informal care to mother (within or outside HH), 0 otherwise.
  - **`care_to_father`**: 1 if respondent gives informal care to father (within or outside HH), 0 otherwise.
  - **`care_to_mother_intensive`** / **`care_to_father_intensive`**: 1 if **daily** (or co-residential personal care) to mother/father, 0/NaN otherwise.
    Here “intensive” = daily help (GIVEN_HELP_DAILY) or personal care within household.
  - **`mother_alive`** / **`father_alive`**: 1/0 from `dn026_1` / `dn026_2` (parent still alive).
  - **`gender`**: 2 = female (only females in saved data).

**For statistic 1:**

- **Numerator**: Female respondents with **both** `care_to_mother == 1` **and** `care_to_father == 1` in the **same wave** (same row).
  → “Care for both parents simultaneously” = same interview year.
- **Denominator**: Female respondents who are “caregivers” in the sense of giving any care to at least one parent: `(care_to_mother == 1) | (care_to_father == 1)`.
- **Daily variant**: Use `care_to_mother_intensive` and `care_to_father_intensive` instead of `care_to_mother` / `care_to_father` for numerator and denominator (and restrict to non-missing where needed).

**New variable(s) needed in estimation data (optional but recommended):**

- **`care_to_both_parents`**: `(care_to_mother == 1) & (care_to_father == 1)`.
- **`care_to_both_parents_intensive`**: `(care_to_mother_intensive == 1) & (care_to_father_intensive == 1)` (with consistent handling of NaN).
- Optionally **`any_care_to_parent`**: `(care_to_mother == 1) | (care_to_father == 1)` and intensive counterpart, so the task does not depend on repeating the logic.

If we do **not** add these in the estimation pipeline, the descriptives task can compute them on the fly when building the table (same logic, no new columns in CSV).

---

### 2.2 Parent–child data (`task_create_parent_child_data_set.py`)

- **Output**: `BLD / "data" / "share_parent_child_data.csv"` (and e.g. `share_parent_child_data_couple.csv`).
- **Population**: Individuals with `age` in `[MIN_AGE_PARENTS, MAX_AGE_PARENTS + 5]` (older adults, “parents” in the sense of care recipients).
- **Relevant variables**:
  - **`married`**: 1 if partner present (mstat 1 or 2), 0 otherwise — from **`create_married_or_partner_alive()`**. So “partner alive” = **`married == 1`** (partner in same household).
  - **`informal_care_general`**: 1 if receives any informal care (inside HH personal care or outside HH any help), 0/NaN.
  - **`informal_care_daily`**: 1 if receives informal care at **daily** frequency (inside and/or outside), 0/NaN.
  - **`informal_care_child`**: 1 if receives informal care from **at least one (own) child** (any frequency for waves 1,2,5; child coded in sp003 for 6–9), 0/NaN.
  - **`informal_care_daily_child`**: 1 if at least one child provides care (inside or **daily** outside), 0/NaN.
  - **`informal_care_daily_two_children`**: 1 if **at least two children** provide care (daily definition: 2+ outside daily, or 1 inside + 1+ outside daily), 0/NaN.
    Used in shares among those who receive informal care.
  - **`has_two_children`**: 1 if at least two children (from `ch006_*`), 0/NaN — from **`create_children_information()`**.
  - **`sp021d1`**: Help with personal care **from spouse/partner** (inside household). From merge docs: “sp021d1 = R received help with personal care from: spouse/partner”.
  - **`sp003_1`, `sp003_2`, `sp003_3`**: Who gave help from **outside** household (person 1–3). Coding is wave-specific (e.g. 10 = child, 11 = stepchild; 101–107 = network persons in Wave 4). **Spouse/partner** in sp003 is typically code **1** in SHARE (to be confirmed in codebook).

**For statistic 2:**

- **Subsample**: Rows with **`married == 1`** (partner alive / in household).
- **Denominator**: Among these, we need a clear “receives any informal care” definition:
  - **General**: e.g. `informal_care_general == 1` (or `receives_any_care`-type variable if we restrict to care recipients only).
  - **Daily**: `informal_care_daily == 1`.
- **Shares to compute**:
  - Share of informal care **from spouse/partner**: need a variable “receives informal care from spouse/partner”.
    - **Inside HH**: `sp021d1 == 1` (help from spouse/partner with personal care).
    - **Outside HH**: sp003_* == spouse code (likely 1) — **needs verification** in SHARE codebook and possibly new variable.
  - Share **from (at least one) child**: **general** = `informal_care_child == 1`, **daily** = `informal_care_daily_child == 1`.

So we need:

- **New variable(s) in parent–child data**:
  **`care_from_spouse_partner`** (general): 1 if receives informal care from spouse/partner (inside: sp021d1; outside: sp003_* = spouse code if applicable).
  **`care_from_spouse_partner_daily`** (daily): same but restricting outside help to daily (sp005_* == DAILY when sp003_* is spouse).
  If spouse is only ever inside HH, we may only need inside-HH (sp021d1); document assumption.

**For statistic 3:**

- **Subsample**: Rows with **at least one informal caregiver** **and** **at least two (adult) children**:
  - “At least one informal caregiver” = receives any informal care: **general** → `informal_care_general == 1`, **daily** → `informal_care_daily == 1`.
  - “At least two children” = **`has_two_children == 1`**.
- **Numerator**: Among this subsample, count with **multiple caregiving children** (≥2 children provide care).
- **Existing**: **`informal_care_daily_two_children`** = 1 if ≥2 children provide care under the **daily** definition. So **daily** variant can use this (possibly among those with `informal_care_daily == 1` and `has_two_children == 1`).
- **General variant**: There is **no** existing “at least two children provide care” for **any frequency** (non-daily). The current “two children” logic uses daily outside count and inside count. So we need:
  - **New variable (parent–child data)**: e.g. **`informal_care_two_children`** (general): 1 if at least two children provide informal care (any frequency), 0/NaN — analogous to `informal_care_daily_two_children` but without requiring daily for outside help. This may require wave-specific rules (similar to existing child-coding in the module).

---

## 3. Definitions and clarifications

### 3.1 “Informal care in general” vs “daily caregiving”

- **Estimation data (statistic 1)**
  - **General**: `care_to_mother` / `care_to_father` (any care within or outside HH).
  - **Daily**: `care_to_mother_intensive` / `care_to_father_intensive` (daily or within-HH personal care).

- **Parent–child data (statistics 2 and 3)**
  - **General**: Variables that do **not** restrict to daily (e.g. `informal_care_general`, `informal_care_child`).
  - **Daily**: Variables that require daily (e.g. `informal_care_daily`, `informal_care_daily_child`, `informal_care_daily_two_children`).

### 3.2 Partner “alive” (statistic 2)

- Interpreted as **partner present (alive and in household)**:
  - **`married == 1`** from `create_married_or_partner_alive()` (mstat 1 or 2: married/registered partnership).
- So “individuals where the partner is alive” = **restrict to `married == 1`**.

### 3.3 Denominator for shares (statistic 2)

- Option A: **Among all with partner alive**, share who receive care from spouse/partner and share who receive from (at least one) child. (Denominator = all with married==1.)
- Option B: **Among those with partner alive who receive any informal care**, share from spouse/partner and share from child. (Denominator = married & receives informal care.)

The wording “what is the share of informal care from spouse/partner and … from a child” is most naturally **Option B** (shares among care recipients). Plan: **Option B**; if you prefer Option A, we can switch.

### 3.4 Statistic 3 – “families with at least one informal caregiver and at least two (adult) children”

- **Family** = one row (one individual/parent in the parent–child sample).
- **At least one informal caregiver**: the individual **receives** at least some informal care (general or daily, depending on column).
- **At least two (adult) children**: **`has_two_children == 1`**.
- **Multiple caregiving children**: **≥2 children** provide informal care (general or daily).

---

## 4. New variables to add (summary)

### 4.1 Estimation data (`task_create_estimation_data_set.py`)

| Variable | Definition | Used in |
|----------|------------|--------|
| **`care_to_both_parents`** | `(care_to_mother == 1) & (care_to_father == 1)` | Stat 1 general |
| **`care_to_both_parents_intensive`** | `(care_to_mother_intensive == 1) & (care_to_father_intensive == 1)` (handle NaN: treat as 0 or exclude) | Stat 1 daily |

Optional: `any_care_to_parent`, `any_care_to_parent_intensive` for denominator (can also be computed in descriptives).

### 4.2 Parent–child data (`task_create_parent_child_data_set.py`)

| Variable | Definition | Used in |
|----------|------------|--------|
| **`care_from_spouse_partner`** | 1 if receives informal care from spouse/partner (inside: sp021d1==1; outside: sp003_* == spouse code if used) | Stat 2 general |
| **`care_from_spouse_partner_daily`** | Same, but outside only if daily (sp005_* == DAILY) | Stat 2 daily |
| **`informal_care_two_children`** | 1 if ≥2 children provide informal care (any frequency); wave-specific like existing child logic | Stat 3 general |

**Spouse/partner code** in `sp003_*`: to be confirmed (likely 1). If only inside-HH spouse care is used, we can set “care from spouse” = (sp021d1 == 1) and note in the doc.

---

## 5. Implementation steps (outline)

### Step 1: Estimation data – new variables (if desired)

- In **`task_create_estimation_data_set.py`**, inside or after **`check_share_informal_care_to_mother_father()`** (or right before save):
  - Add **`care_to_both_parents`** = (care_to_mother == 1) & (care_to_father == 1).
  - Add **`care_to_both_parents_intensive`** = (care_to_mother_intensive == 1) & (care_to_father_intensive == 1), with explicit NaN handling (e.g. fillna(0) or only among non-missing intensive).
- Add these to any `static_cols` / columns written to CSV if weights are multiplied.

Alternatively, compute these only in the descriptives task (no pipeline change).

### Step 2: Parent–child data – new variables

- In **`task_create_parent_child_data_set.py`**, inside **`create_care_variables()`** (or right after):
  - **Care from spouse/partner**
    - **General**: `care_from_spouse_partner` = 1 if (sp021d1 == 1) or (any of sp003_1/2/3 == spouse code for waves where applicable).
    - **Daily**: `care_from_spouse_partner_daily` = 1 if inside (sp021d1) or outside daily with spouse (sp003_* == spouse, sp005_* == DAILY).
    - Confirm spouse code in SHARE (e.g. 1) and wave consistency.
  - **Multiple children (general)**
    - **`informal_care_two_children`**: mirror logic of `informal_care_daily_two_children` but for “at least two children provide care” **without** requiring daily for outside-HH help (wave-specific, using existing child-coding patterns).

### Step 3: Descriptives module and task

- **New file**: `src/caregiving/descriptives/task_intra_family_caregiving.py`.
- **Task**: `task_intra_family_statistics()`.
  - **Depends on**:
    - Estimation data: `BLD / "data" / "share_estimation_data.csv"`.
    - Parent–child data: `BLD / "data" / "share_parent_child_data.csv"`.
  - **Product**: e.g. `BLD / "tables" / "publication" / "descriptives" / "intra_family_caregiving_statistics.tex"` (or under `descriptives/` as you prefer).
  - **Steps**:
    1. Load estimation data; restrict to females (already the case).
    2. **Statistic 1 (general)**
       - Denominator: `(care_to_mother == 1) | (care_to_father == 1)`.
       - Numerator: `(care_to_mother == 1) & (care_to_father == 1)`.
       - Share = numerator.sum() / denominator.sum() (or use design weights if desired).
    3. **Statistic 1 (daily)**
       - Same with `care_to_mother_intensive` / `care_to_father_intensive` (and consistent NaN handling).
    4. Load parent–child data.
    5. **Statistic 2**
       - Restrict to `married == 1`.
       - Among those with any informal care (general: `informal_care_general == 1`; daily: `informal_care_daily == 1`):
         - Share from spouse/partner: mean of `care_from_spouse_partner` / `care_from_spouse_partner_daily`.
         - Share from child: mean of `informal_care_child` / `informal_care_daily_child`.
       - Report both shares (and optionally N).
    6. **Statistic 3**
       - Restrict to “receives any informal care” (general or daily) **and** `has_two_children == 1`.
       - Among them, share with multiple caregiving children: mean of `informal_care_two_children` (general) or `informal_care_daily_two_children` (daily).
    7. Build LaTeX table: e.g. one column per statistic (1–3), two rows (general vs daily), or one row per statistic with “General” and “Daily” columns. Add headers and caption as needed.
  - **Weights**: Decide whether to use design/individual weights for SHARE; if so, pass weight column and use weighted means.

### Step 4: Registration and tests

- Register the task in **`pyproject.toml`** under the descriptives/summary-statistics (or similar) marker if needed.
- Optionally add a small test or smoke run to ensure the task runs and the table is produced.

---

## 6. Open points and assumptions

1. **Statistic 1 denominator**
   - Current plan: “female caregivers” = any care to mother or father `(care_to_mother | care_to_father)`.
   - Alternative: restrict to those with **both parents alive** (`mother_alive == 1` & `father_alive == 1`) and then among those, take caregivers. Please confirm.

2. **Statistic 2 – spouse/partner from outside HH**
   - Need to confirm SHARE code for “spouse/partner” in **sp003_1/2/3** (likely 1). If not available or not used in your waves, we can define “care from spouse/partner” using **inside household only** (sp021d1).

3. **Weights**
   - Use design/individual weights for SHARE (e.g. `design_weight`) for nationally representative shares? If yes, use weighted means and report “weighted share”.

4. **Table layout**
   - Preferred: one table with rows = statistics 1–3, columns = “General” and “Daily” (and optionally N, sample description)? Or two separate tables (general vs daily)? Confirm layout preference.

5. **Statistic 3 – “at least two (adult) children”**
   - Implemented as **`has_two_children == 1`** (at least two children in the roster). Confirm that “adult” is not an extra filter (e.g. age of child) in your data; if it is, we need a variable for “number of adult children”.

Once these are confirmed (and any adjustments to the plan agreed), implementation can follow the steps above.
