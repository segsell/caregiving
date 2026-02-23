# Education Heterogeneity in Caregiving Policy Effects

## Model context

Education is binary: **Low Education** (0) and **High Education** (1).
It enters the model through:

- **Wage equation**: education-specific intercept and experience returns (`gamma_0`, `gamma_ln_exp`)
- **Working hours**: education-specific average annual hours (PT and FT)
- **Minimum wages**: education-specific PT floor
- **Partner income**: education-specific
- **Child benefits**: via `children_by_state[sex, edu, partner, period]`
- **Unemployment benefits**: education-dependent
- **Inheritance amounts**: education-specific matrix
- **Leave top-up (policy)**: through the wage equation — higher education → higher wages → larger absolute top-up

The key implication: **education determines the opportunity cost of caregiving**.
Low-education agents have lower wages and thus lose less income when reducing work for care;
high-education agents face a steeper trade-off.

---

## 1. Care arrangement choices by education

**Question**: Do low-edu agents provide more informal care? Do high-edu agents opt for formal care more often?

**Metrics** (reproduce Panel B, split by education):
- Share informal CG (low edu) vs. (high edu), by age group
- Share formal care (low edu) vs. (high edu)
- Share light vs. intensive informal care by education

**Why interesting**: Tests the theoretical prediction that high opportunity-cost agents substitute toward formal care. The leave policy should narrow this gap (by reducing the cost of informal care for high-edu agents).

**Policy dimension**: Percentage point change in informal care take-up should be larger for high-edu agents under the leave policy, since the top-up is proportionally more generous for higher earners.

---

## 2. Labor supply while caregiving, by education

**Question**: Among current caregivers, do high-edu agents work more? How does the leave policy shift the FT/PT/unemployed composition differently by education?

**Metrics** (reproduce Panel C, split by education):
- Share FT / PT / employed / unemployed among CG, by education × age group
- Percentage change in these shares under leave policy, by education

**Why interesting**: The leave policy replaces 65% (normal) or 100% (full) of prior wages. For high-edu agents with high wages, this creates a large absolute benefit when reducing hours or stopping work. But the replacement rate is the same — so the behavioral response might be symmetric. If it's asymmetric, that reveals something about preferences or constraints.

**Key hypothesis**: High-edu caregivers may show a larger shift from FT to unemployed (take-up of the leave) because the absolute benefit is larger, while low-edu caregivers may already be more likely to be unemployed while caregiving (lower opportunity cost even without the policy).

---

## 3. Caregiving duration and intensity by education

**Question**: Do low-edu agents have longer caregiving spells? Does the policy change spell length differently?

**Metrics**:
- Average total CG years by education (among ever-CG)
- Distribution of CG duration (1yr / 2-3yr / 4+yr) by education
- Share of intensive vs. light informal care by education

**Why interesting**: If low-edu agents have longer spells, the cumulative income and pension losses are compounded. The leave policy's long-term impact on pension points is then more consequential for this group. Conversely, if the policy induces high-edu agents to care for longer, the fiscal cost rises.

---

## 4. Benefits and top-ups received, by education

**Question**: How much do agents actually receive from the policy, by education?

**Metrics** (reproduce Panel E, split by education):
- Avg. Pflegegeld (baseline) by education × labor state
- Avg. leave top-up (normal / full) by education × labor state
- Ratio: top-up / prior gross wage by education (effective replacement rate)

**Why interesting**: The top-up is bounded (lower and upper bounds in the normal leave). Low-edu agents may hit the lower bound; high-edu agents may hit the upper bound (capped at 6G in the full leave). This creates non-linearities that differ by education, affecting who benefits proportionally more.

**Additional metric**: Share of caregivers receiving zero top-up (retired CG + no prior job) by education. Low-edu agents may be more likely to have `job_before_caregiving == 0`, making them ineligible.

---

## 5. Economic outcomes by education

**Question**: How do wealth, consumption, and savings differ by education, and does the policy close or widen the gap?

**Metrics** (reproduce Panel F, split by education):
- Avg. wealth, consumption, savings (< 63) by education
- Avg. gross labor income (< 63) by education
- Policy-induced change in these metrics by education

**Sub-analysis for ever-caregivers only**:
- Same metrics, but conditional on ever having been an informal caregiver
- Comparison: ever-CG vs. never-CG gap, by education

**Why interesting**: The wealth gap between ever-CG and never-CG agents might be larger for high-edu agents (they lose more income while caregiving). The policy should narrow this gap, but does it do so equally across education groups?

---

## 6. Experience and pension outcomes by education

**Question**: Does caregiving erode pension entitlements more for one education group? Does the leave policy mitigate this?

**Metrics** (reproduce Panel H, split by education):
- Avg. experience years at retirement: all agents vs. ever-CG, by education
- Experience gap (ever-CG minus all) by education
- Avg. retirement age by education × ever-CG status
- Avg. gross pension income by education × ever-CG status

**Why interesting**: This is arguably the most policy-relevant heterogeneity. If high-edu agents accumulate more experience, they have more to lose from caregiving interruptions. The leave policy (especially with job retention) preserves the employment relationship — but does it differentially protect pension entitlements by education?

**Key metric**: "Caregiving pension penalty" = pension income of ever-CG minus pension income of never-CG, by education. If the leave policy reduces this penalty more for one group, that's a distributional finding.

---

## 7. Fiscal cost heterogeneity by education

**Question**: Is the policy more expensive for high-edu agents (larger top-ups)?

**Metrics** (reproduce key rows from the fiscal table, split by education):
- Avg. gov. leave top-up cost per caregiver, by education
- Total gov. expenditure by education group
- Tax revenue change by education (high-edu agents pay more tax → larger revenue loss when they reduce work)
- Net fiscal cost per caregiver by education

**Why interesting**: The policy is funded from general revenue. If high-edu caregivers receive larger top-ups but also generate more tax revenue when working, the net fiscal impact depends on the balance. This informs whether the policy is fiscally progressive or regressive.

---

## 8. Distributional / equity perspective

**Question**: Is the caregiving leave policy progressive or regressive?

**Metrics** (cross-education comparison):
- Ratio of policy benefit received (top-up) to baseline income, by education
- Change in informal care take-up rate (pp), by education
- Change in employment rate (pp), by education
- Change in wealth at retirement, by education

**Framing**: A progressive policy would:
- Provide proportionally larger benefits to low-edu agents (relative to income)
- Increase informal care take-up more for low-edu agents
- Narrow the economic gap between education groups among caregivers

A regressive policy would do the opposite (larger absolute benefits to high-edu agents, potentially widening gaps).

---

## 9. Suggested implementation approach

### Option A: Separate education-stratified table
Create a new task module (e.g., `task_policy_changes_by_education.py`) that produces a table with columns: `Low Edu Baseline`, `Low Edu Normal Leave`, `Low Edu Full Leave`, `High Edu Baseline`, `High Edu Normal Leave`, `High Edu Full Leave`. Reuse the same panel structure but filter `df[df["education"] == edu]` before passing to panel functions.

**Pros**: Clean separation, easy to read side by side.
**Cons**: Very wide table (6+ data columns); might need to split into two tables (one per education group).

### Option B: Add education dimension to existing panels
For each existing metric, add `(low edu)` and `(high edu)` variants. This keeps everything in one table but doubles the number of rows.

**Pros**: All comparisons in one place.
**Cons**: Table becomes very long.

### Option C: Dedicated education-difference table
Show only the *difference* or *ratio* between education groups (e.g., high minus low, or high/low) for each metric and policy scenario. This directly highlights where heterogeneity is largest.

**Pros**: Compact, focuses on the question.
**Cons**: Loses the levels; harder to interpret without context.

### Recommended: Option A with two tables
One table per education group, same structure as the current `policy_behavioral_changes.tex`. Then a compact summary table showing the education-specific policy effects (Option C) for the most important metrics from sections 1, 2, 6, 7 above.

---

## 10. Priority ranking

| Priority | Section | Rationale |
|----------|---------|-----------|
| 1 | Care arrangement choices (§1) | Core behavioral response |
| 2 | Experience and pensions (§6) | Key long-run consequence |
| 3 | Benefits received (§4) | Who actually receives the policy |
| 4 | Labor while caregiving (§2) | Mechanism channel |
| 5 | Fiscal costs (§7) | Policy sustainability |
| 6 | Distributional (§8) | Equity framing |
| 7 | Duration (§3) | Moderating factor |
| 8 | Economic outcomes (§5) | Broader welfare |
