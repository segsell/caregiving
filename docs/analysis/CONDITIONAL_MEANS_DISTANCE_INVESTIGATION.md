# Investigation: Conditional Means by Distance to First Care Demand

## Issues Raised

1. **Black solid no-care-demand line has a weird kink at t=0** – There is no care demand in the counterfactual, so this should not occur (for employment and other outcomes).
2. **Employment rate drop seems too stark** compared to other plots.
3. **Need to understand sample construction and conditioning**.

---

## Sample Construction

### Who is in the sample?

1. **Restrict to `caregiving_type == 1`** (agents who can provide informal care).
2. **Restrict to agents who ever had `care_demand > 0` in the original (baseline) simulation** – This is enforced by the `dist_map` from `add_distance_to_first_care_demand(df_o)`, which only has `first_care_demand_period` for agents with at least one period with `care_demand > 0`. The inner merge keeps only those agents.
3. **Window**: Observations with `distance = period - first_care_demand_period` in `[-window, window]` (default ±20 years).

### Original/baseline profiles (colored lines, groups 1–5)

- **Merged** comes from `_prepare_merged_type1(path_to_original_data, ...)`:
  - Loads baseline (Jan7) data.
  - Filters to alive, type 1, and (via `dist_map`) ever care demand.
  - Adds `distance_to_first_care_demand = period - first_care_demand_period`.
- **Duration groups** (mutually exclusive, from original data):
  - 1-year: care at t=0 only, then stop (exactly 1 year).
  - 2-year: care at t=0 and t=1, then stop.
  - 3-year: care at t=0, t=1, t=2, then stop.
  - 4-year: care at t=0, t=1, t=2, t=3, then stop.
  - 5+ year: care at t=0, 1, 2, 3, 4 (at least 5 years).
- **Outcomes** are computed with `_add_outcome_columns(merged)` using `calculate_simple_outcomes(merged, "original")` (WORK, FULL_TIME, PART_TIME).

### No-care-demand line (black solid)

- **Merged_ncd** in `_compute_no_care_demand_profile`:
  1. Loads original and no-care-demand data.
  2. Restricts both to alive, type 1, and (when flags set) ever caregivers / ever care demand.
  3. Computes `dist_map` from **original** data.
  4. **Merges no-care-demand data with `dist_map`** on `agent` (inner join).
  5. Sets `distance_to_first_care_demand = period - first_care_demand_period`.
  6. Filters to `distance` in `[-window, window]`.
  7. Adds outcomes with `_add_outcome_columns_no_care_demand(merged_ncd)`, i.e. `calculate_simple_outcomes(merged_ncd, "no_care_demand")` (WORK_NO_CARE_DEMAND, etc.).
- So: same agents as original (type 1, ever care demand), same alignment in event time, but outcomes from the **no-care-demand** simulation.

### Ever-caregiver line (black dashed)

- Same agents as above who **ever provided informal care** in the original, using original outcomes.

---

## Why a kink at t=0 in the no-care-demand line?

In the no-care-demand counterfactual there is no care demand. t=0 is simply “the period when care demand would have started in the baseline”; there is no discrete shock at t=0 in the counterfactual.

Possible causes of a visible kink:

### 1. Composition changes by distance

- At each `t`, the sample is different agents at different ages.
- `first_care_demand_period` differs across agents, so age at t=0 varies.
- If age composition changes sharply around t=0 (unlikely for a single period), that could create a kink.
- Check: plot mean age by `distance_to_first_care_demand` around t=0.

### 2. Different numbers of observations at t=0

- If fewer agents contribute at t=0 (e.g. due to death or censoring), the mean could jump.
- Check: plot `n` by `distance_to_first_care_demand` around t=0 (see `debug_conditional_means_distance_to_first_care.py`).

### 3. Outcome coding in no-care-demand data

- `calculate_simple_outcomes(merged_ncd, "no_care_demand")` uses `WORK_NO_CARE_DEMAND` = [2, 3] (part-time and full-time in the 4-choice model).
- The no-care-demand simulation uses `choice` ∈ {0,1,2,3} (retirement, unemployed, part-time, full-time).
- `merged_ncd` comes from the no-care-demand data; its `choice` column should reflect that model.
- If the no-care-demand output had a different coding or missing values around certain periods, that could cause artifacts.
- Check: inspect `choice` distribution by `distance_to_first_care_demand` around t=0.

### 4. Life-cycle effects around typical care demand age

- If care demand typically starts in a narrow age band (e.g. 55–65), moving from t<0 to t>0 shifts the average age and thus average employment.
- This would be a smooth gradient, not a sharp kink, unless there is a strong non-linearity (e.g. retirement cliff).
- Check: plot employment and mean age vs. `distance_to_first_care_demand` in a fine window around t=0.

### 5. Structural difference at t=0

- No structural reason in the setup for a kink at t=0 in the no-care-demand data.
- Baseline and no-care-demand share the same `initial_states.pkl` and seed, so agents are aligned.

---

## Employment drop magnitude

### What differs from other plots?

1. **task_plot_employment_rate_by_distance_to_first_care** (and similar):
   - Sample: type 1 (or type 0), ever care demand.
   - Plots baseline vs. no-care-demand employment by distance (often with `work_o` and `work_c` from matched (agent, period)).
   - Drop = difference between those two series.

2. **task_conditional_means_distance_to_first_care_demand**:
   - Same base sample (type 1, ever care demand).
   - Extra conditioning: **duration groups 1–5** (exact 1/2/3/4 years, 5+ years).
   - Each colored line = employment in baseline for that duration group.
   - Black solid = mean employment in no-care-demand for **all** care-demand agents.

### Why employment might drop more here

1. **Heterogeneity by duration**: Agents with longer care demand (e.g. 5+ years) likely reduce employment more and for longer. The 5+ year line will look “starker” than an overall average.
2. **Comparing group-specific lines to overall counterfactual**: The no-care-demand line is the mean over all care-demand agents. The treatment effect for each duration group can exceed this average, especially for the 5+ year group.
3. **Selection**: Agents with longer spells may be selected on unobservables that correlate with stronger labor supply responses.

### Plausibility

- A larger drop for long-duration groups than in the unconditional comparison is expected.
- A very large drop (e.g. >20 pp) would be worth double-checking, especially sample construction and outcome definitions.

---

## Diagnostic script

Run:

```bash
PYTHONPATH=src python debug_conditional_means_distance_to_first_care.py
```

(Use the project’s conda/env if needed.)

It reports:

- Sample sizes and distance range.
- Observations per distance around t=0.
- Mean age by distance (if available).
- No-care-demand profiles around t=0 for work, working_hours_weekly, part_time, full_time.
- Group sizes for duration groups 1–5.
- Employment at t = -2, 0, 2 for no-care-demand, 1-year, and 5+ year groups.

---

## Recommendations

1. Run the diagnostic script and inspect:
   - Whether `n` or mean age change sharply at t=0.
   - The exact employment path around t=0 for the no-care-demand line.
2. If a kink remains:
   - Check whether it appears in all outcomes (work, hours, part-time, full-time) or only some.
   - Inspect `choice` distribution by distance around t=0 in the no-care-demand data.
3. Compare conditioning and samples between this plot and `task_plot_employment_rate_by_distance_to_first_care` (and any other reference plots).
4. If the employment drop still looks implausible, consider:
   - Adding age-at-first-care-demand as a control or stratifier.
   - Checking for differences in how baseline vs. no-care-demand outcomes are computed in this task vs. others.
