# Structure, setup, and file handling: `task_plot_event_study_total_caregiving_years.py`

This document describes the organization of the event-study (difference) tasks that are grouped by **total caregiving years** (1, 2, 3, 4, 5+ over the lifecycle). The goal is to support creating a parallel module for **reverse** event studies (t = 0 = mother’s death) with the same outcome/data structure and a consistent naming and path convention.

---

## 1. Purpose and setup

- **What the module does:** Plots the **difference** in an outcome (baseline minus no-care-demand) by **distance to an event** (t = 0), with the same layout as the consecutive event studies (dashed baseline, horizontal line at 0, vertical line at t = −0.5, subgroup lines with markers).
- **Event definitions (forward):**
  - **First care demand:** t = 0 = first period with `care_demand > 0`.
  - **First caregiving spell:** t = 0 = first period in which the agent provides informal care.
- **Subgroups:** Agents are grouped by **total caregiving years over the lifecycle** (up to `end_age_caregiving` from specs): exactly 1, 2, 3, 4, or 5+ years (not necessarily consecutive). Implemented via `identify_agents_by_total_caregiving_over_lifecycle(df_o, start_age, end_age_caregiving)` from `caregiving.counterfactual.plotting_helpers`.
- **Sample:** Ever caregivers only (`ever_caregivers=True`, `ever_care_demand=False`). Baseline and no-care-demand data are merged on `(agent, period)`; only common agent-periods are kept.
- **Data variants:**
  - **Standard:** `simulated_data_estimated_params.pkl` + `simulated_data_no_care_demand.pkl`.
  - **back_to_Jan7:** `simulated_data_estimated_params_back_to_Jan7.pkl` + `simulated_data_no_care_demand_back_to_Jan7.pkl`.
- **Age groups:** Four groups used for filtering and in task ids/filenames: `all_ages` (no age filter), `ages_40_49`, `ages_50_59`, `ages_60_70`. Implemented as `_AGE_GROUPS = ((None, None, "all_ages"), (40, 49, "ages_40_49"), (50, 59, "ages_50_59"), (60, 70, "ages_60_70"))`.

---

## 2. Task organization (loop-per-variant pattern)

- **No `globals()` or dynamic function names.** Each task is a normal function with a fixed name.
- **One loop per (outcome, event, data variant).** The loop runs only over the four age groups (`age_min_val`, `age_max_val`, `age_label_val`).
- **All task arguments are written directly in the function signature** as default values: `path_to_plot`, `path_to_original_data`, `path_to_no_care_demand_data`, `path_to_specs`, `age_min`, `age_max`. The loop variables are used in those defaults so each iteration binds a different path and age filter.
- **Same function name in every iteration of a given loop.** Pytask still gets four distinct tasks per loop because `@pytask.task(id=...)` uses a **unique id** per age group (e.g. `f"{age_label_val}_employment_first_care_demand_estimated_params"`).
- **Pytask marks:** Each task is decorated with `@pytask.mark.publication_event_study`, `@pytask.mark.publication_counterfactual`, and `@pytask.mark.publication`.

---

## 3. Outcomes and task blocks

| Outcome        | Description / ylabel                                      | Endogenous y-axis? |
|----------------|------------------------------------------------------------|---------------------|
| employment     | Difference in employment rate                              | No                  |
| full_time      | Difference in full-time rate                               | No                  |
| part_time      | Difference in part-time rate                               | No                  |
| working_hours  | Difference in weekly working hours (from `working_hours/52`) | Yes                 |
| labor_income   | Difference in monthly gross labor income (from `gross_labor_income/12`) | Yes                 |

For each outcome there are **four task blocks** (four loops):

1. First care demand, standard data
2. First care demand, back_to_Jan7 data
3. First caregiving spell, standard data
4. First caregiving spell, back_to_Jan7 data

Each block is a loop over `_AGE_GROUPS` defining a single task function (same name for all four ages in that block).

**Total:** 5 outcomes × 4 (event × data) × 4 ages = **80 tasks**.

---

## 4. File and path layout

**Base directory for all outputs:**

```text
BLD / "figures" / "publication" / "counterfactual" / "event_study"
```

**Per-outcome subdirectory:** `event_study / {outcome}`, where `{outcome}` is one of:

- `employment`
- `full_time`
- `part_time`
- `working_hours`
- `labor_income`

**Subfolder for total-caregiving-years plots:** `total_caregiving_years`

So the full path pattern is:

```text
BLD / "figures" / "publication" / "counterfactual" / "event_study" / {outcome} / "total_caregiving_years" / {filename}.pdf
```

**Filename conventions:**

- **Standard data (estimated_params + no_care_demand):**
  `event_study_{outcome_descriptor}_by_distance_to_{event}_total_caregiving_{age_label}.pdf`
  Examples:
  - Employment: `event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving_all_ages.pdf`
  - Full-time: `event_study_full_time_by_distance_to_first_care_demand_total_caregiving_ages_40_49.pdf`
  - Working hours: `event_study_working_hours_weekly_by_distance_to_first_caregiving_spell_total_caregiving_all_ages.pdf`
  - Labor income: `event_study_monthly_gross_labor_income_by_distance_to_first_care_demand_total_caregiving_all_ages.pdf`

- **back_to_Jan7 data:**
  Same name with prefix `back_to_Jan7_`, e.g.
  `back_to_Jan7_event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving_all_ages.pdf`

`{event}` in the filename is either `first_care_demand` or `first_caregiving_spell`. `{age_label}` is `all_ages`, `ages_40_49`, `ages_50_59`, or `ages_60_70`.

---

## 5. Function naming convention

Task function names are fixed per block (same name for all four ages in the loop). They encode outcome, event, and data variant:

- **Employment, first care demand, standard:**
  `task_plot_event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving`
- **Employment, first care demand, back_to_Jan7:**
  `task_plot_event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7`
- **Employment, first caregiving spell, standard:**
  `task_plot_event_study_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving`
- **Employment, first caregiving spell, back_to_Jan7:**
  `task_plot_event_study_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7`

For other outcomes the pattern is:

- `task_plot_event_study_{outcome}_by_distance_to_first_care_demand_total_caregiving`
- `task_plot_event_study_{outcome}_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7`
- `task_plot_event_study_{outcome}_by_distance_to_first_caregiving_spell_total_caregiving`
- `task_plot_event_study_{outcome}_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7`

where `{outcome}` is `full_time`, `part_time`, `working_hours`, or `labor_income`.

---

## 6. Task ids

Task ids are unique per (age, outcome, event, data). Examples:

- `{age_label_val}_employment_first_care_demand_estimated_params`
- `{age_label_val}_employment_first_care_demand_back_to_Jan7`
- `{age_label_val}_full_time_first_care_demand_estimated_params`
- `{age_label_val}_labor_income_first_caregiving_spell_back_to_Jan7`

So each of the 80 tasks has a distinct id.

---

## 7. Shared logic inside the module

- **`event_study_total_caregiving_merged_and_profiles(...)`**
  Builds the merged (baseline vs no-care-demand) dataframe, computes distance to the event (care_demand or caregiving_spell), filters by window and optional age at event, computes outcome difference, then uses `identify_agents_by_total_caregiving_over_lifecycle(df_o, start_age, end_age_caregiving)` to get five agent groups and returns (merged, prof_diff, prof_1_year_diff, …, prof_5_year_diff). All profiles use a common distance column name (`_DIST_COL = "distance_to_first_care"`) and a `"diff"` column so the same plot function can be used for all outcomes.

- **`plot_outcome_difference_by_distance_total_caregiving(...)`**
  Takes the six profile DataFrames (baseline diff plus five subgroup diffs), window, path_to_plot, xlabel, ylabel, and optional `endogenous_ylim`. Produces the event-study figure (dashed baseline, 0 line, vline at −0.5, five subgroup lines with labels "1 total care year", …, "5+ total care years"). For working_hours and labor_income, `endogenous_ylim=True` is used.

- **Specs:** Each task loads `path_to_specs` (BLD / "model" / "specs" / "specs_full.pkl") to read `start_age` and `end_age_caregiving` for the lifecycle grouping. No hardcoded MAX_AGE_CAREGIVING.

- **Data loading:** Each task calls `prepare_dataframes_simple(pd.read_pickle(path_to_original_data), pd.read_pickle(path_to_no_care_demand_data), ever_caregivers, ever_care_demand)` then computes the outcome series (work, full-time, part-time, or derived working_hours/labor_income) for both scenarios and passes them into `event_study_total_caregiving_merged_and_profiles`.

---

## 8. Dependencies (imports)

- **Counterfactual plotting:** `add_distance_to_first_care`, `add_distance_to_first_care_demand`, `calculate_simple_outcomes`, `get_age_at_first_event`, `identify_agents_by_total_caregiving_over_lifecycle`, `prepare_dataframes_simple` from `caregiving.counterfactual.plotting_helpers`.
- **Model:** `INFORMAL_CARE` from `caregiving.model.shared`.
- No imports from other `task_plot_*` modules.

---

# Part B: Outline for `task_plot_reverse_event_study_total_caregiving_years.py`

This section outlines what would be needed to add **reverse** event studies (t = 0 = mother’s death) that mirror the total-caregiving-years structure: same outcomes, same two data pairs, same loop-per-variant style and naming, with a new subfolder under the existing reverse-event-study path used in `task_plot_event_study_employment_rate_mother_death.py`.

---

## 9. Relation to existing mother-death module

- **`task_plot_event_study_employment_rate_mother_death.py`** already defines reverse event studies where t = 0 = mother’s death (`mother_dead == PARENT_RECENTLY_DEAD`). It uses:
  - **Path:** `BLD / "figures" / "publication" / "counterfactual" / "event_study_reverse" / "employment" / {filename}.pdf` (no subfolder like `total_caregiving_years`).
  - **Helpers:** `add_distance_to_mother_death`, `identify_agents_by_caregiving_before_death`, `identify_agents_by_caregiving_before_death_at_least`, etc. from `caregiving.figures.publication.plotting_helpers_mother_death`.
  - **Grouping so far:** “at least N years before death”, “consecutive N years before death”, or care-demand-based groups — **not** total caregiving years over the full lifecycle.
  - **Data:** Only one data pair (estimated_params + no_care_demand). No back_to_Jan7.
  - **Outcomes:** Only employment.
  - **Loop:** Same four age groups; path and args in the function signature; same function name per block.

The **new** module would:

- Keep **event** definition as in the mother_death module: t = 0 = mother’s death (clear, no “end of caregiving spell”).
- Use **grouping by total caregiving years over the lifecycle** (1, 2, 3, 4, 5+), as in `task_plot_event_study_total_caregiving_years.py`, via `identify_agents_by_total_caregiving_over_lifecycle` and specs (`start_age`, `end_age_caregiving`).
- Add **all alternative outcomes** (full_time, part_time, working_hours, labor_income) and **both data pairs** (standard and back_to_Jan7).
- Introduce a **new subfolder** `total_caregiving_years` under each outcome under `event_study_reverse`, and use consistent **function and file naming** aligned with both the mother_death tasks and the total_caregiving_years event-study module.

---

## 10. Proposed path and subfolder

- **Base (existing in mother_death):**
  `BLD / "figures" / "publication" / "counterfactual" / "event_study_reverse"`

- **Per-outcome (existing for employment):**
  `event_study_reverse / {outcome}` with `{outcome}` in `employment`, `full_time`, `part_time`, `working_hours`, `labor_income`.

- **New subfolder for total-caregiving-years reverse plots:**
  `event_study_reverse / {outcome} / "total_caregiving_years"`

So the full path pattern for the new tasks would be:

```text
BLD / "figures" / "publication" / "counterfactual" / "event_study_reverse" / {outcome} / "total_caregiving_years" / {filename}.pdf
```

This mirrors the forward event-study layout (`event_study / {outcome} / total_caregiving_years`), but under `event_study_reverse`.

---

## 11. Proposed function and file naming

Align with:

1. **Mother_death module:** event is “mother death”, so names and filenames should include `mother_death` (e.g. `event_study_employment_rate_by_distance_to_mother_death_...`).
2. **Total_caregiving_years module:** grouping is “total caregiving” and data variant appears in name/filename when back_to_Jan7.

**Proposed task function names (one per block; same name for all four ages in the loop):**

- Employment, standard data:
  `task_plot_event_study_employment_rate_by_distance_to_mother_death_total_caregiving`
- Employment, back_to_Jan7:
  `task_plot_event_study_employment_rate_by_distance_to_mother_death_total_caregiving_back_to_Jan7`
- Full-time, standard:
  `task_plot_event_study_full_time_by_distance_to_mother_death_total_caregiving`
- Full-time, back_to_Jan7:
  `task_plot_event_study_full_time_by_distance_to_mother_death_total_caregiving_back_to_Jan7`
- (Same pattern for part_time, working_hours, labor_income.)

**Proposed output filenames:**

- Standard:
  `event_study_{outcome_descriptor}_by_distance_to_mother_death_total_caregiving_{age_label}.pdf`
  Examples:
  `event_study_employment_rate_by_distance_to_mother_death_total_caregiving_all_ages.pdf`,
  `event_study_full_time_by_distance_to_mother_death_total_caregiving_ages_40_49.pdf`,
  `event_study_working_hours_weekly_by_distance_to_mother_death_total_caregiving_all_ages.pdf`,
  `event_study_monthly_gross_labor_income_by_distance_to_mother_death_total_caregiving_all_ages.pdf`.
- back_to_Jan7:
  Prefix `back_to_Jan7_`, e.g.
  `back_to_Jan7_event_study_employment_rate_by_distance_to_mother_death_total_caregiving_all_ages.pdf`.

**Proposed task ids (unique per age × outcome × data):**

- `{age_label_val}_employment_mother_death_total_caregiving_estimated_params`
- `{age_label_val}_employment_mother_death_total_caregiving_back_to_Jan7`
- `{age_label_val}_full_time_mother_death_total_caregiving_estimated_params`
- … (and similarly for part_time, working_hours, labor_income and for back_to_Jan7).

(Exact id scheme can be simplified to e.g. `{age_label_val}_employment_mother_death_total_caregiving` and `{age_label_val}_employment_mother_death_total_caregiving_back_to_Jan7` if preferred.)

---

## 12. What needs to be implemented (checklist)

1. **New module**
   - Create `task_plot_reverse_event_study_total_caregiving_years.py` (no changes to existing mother_death or total_caregiving_years code in this step).

2. **Event and distance**
   - Reuse `add_distance_to_mother_death(df_o)` from `plotting_helpers_mother_death` to get `first_death_period` and distance.
   - Filter to agent-periods with valid `first_death_period` and distance in [−window, +window].
   - Filter by age at mother’s death (age_min / age_max) using the same pattern as in the mother_death tasks (e.g. `age_at_death` from first period where `mother_dead == PARENT_RECENTLY_DEAD`).

3. **Merged data and outcome difference**
   - Merge baseline outcome and no-care-demand outcome on (agent, period); compute diff = outcome_o − outcome_c.
   - Build overall profile: mean(diff) by distance.
   - No “event end” or “end of caregiving spell” — the only event is mother’s death.

4. **Total caregiving years grouping**
   - Load specs (path_to_specs) to get `start_age` and `end_age_caregiving`.
   - Call `identify_agents_by_total_caregiving_over_lifecycle(df_o, start_age, end_age_caregiving)` from counterfactual plotting_helpers to get five agent arrays (1, 2, 3, 4, 5+ total care years over lifecycle).
   - For each group, compute profile of mean(diff) by distance.
   - Ensure profile DataFrames use a single distance column name (e.g. `distance_to_first_care` or a shared constant) and a `"diff"` column so the same plotting function can be reused.

5. **Plotting**
   - Reuse the same layout as in `task_plot_event_study_total_caregiving_years`: dashed baseline diff, horizontal line at 0, vertical line at t = −0.5, five subgroup lines (“1 total care year”, …, “5+ total care years”).
   - Either call `plot_outcome_difference_by_distance_total_caregiving` from the existing total_caregiving_years module (if the distance column name and signatures are compatible) or implement a local wrapper / duplicate that accepts the mother-death distance column name and the same six profiles.
   - xlabel: e.g. “Year relative to mother’s death”.
   - ylabel: same as in the forward module per outcome; use `endogenous_ylim=True` for working_hours and labor_income.

6. **Outcomes**
   - Employment: work indicator from `calculate_simple_outcomes`.
   - Full-time / part-time: from `calculate_simple_outcomes`.
   - Working hours: `df["working_hours"].astype(float) / 52.0` (weekly).
   - Labor income: `df["gross_labor_income"].astype(float) / 12.0` (monthly).
   Same as in `task_plot_event_study_total_caregiving_years.py`.

7. **Data pairs**
   - Standard: `simulated_data_estimated_params.pkl` + `simulated_data_no_care_demand.pkl`.
   - back_to_Jan7: `simulated_data_estimated_params_back_to_Jan7.pkl` + `simulated_data_no_care_demand_back_to_Jan7.pkl`.
   - path_to_specs: BLD / "model" / "specs" / "specs_full.pkl" for all tasks.

8. **Task structure**
   - One loop per (outcome, data variant) over the four age groups.
   - Each loop defines a single named task function with path_to_plot, path_to_original_data, path_to_no_care_demand_data, path_to_specs, age_min, age_max written directly in the signature (no globals()).
   - Task count: 5 outcomes × 2 data variants × 4 ages = **40 tasks**.

9. **Pytask marks**
   - Use the same marks as in the mother_death module for reverse event studies (e.g. `@pytask.mark.publication_event_study_reverse`, `@pytask.mark.publication_counterfactual`, `@pytask.mark.publication`). Do not reuse `publication_event_study` if that is reserved for forward event studies.

10. **Sample**
   - Decide whether to use ever_caregivers=True/False and ever_care_demand=True/False. The forward total_caregiving_years module uses ever_caregivers=True, ever_care_demand=False. The existing mother_death tasks use ever_caregivers=False, ever_care_demand=False. For consistency with “total caregiving years” (which is defined only for caregivers), using ever_caregivers=True (and ever_care_demand=False) in the new reverse total_caregiving_years tasks would align with the forward module; document the choice in the docstring.

11. **Helper**
   - Implement a single helper in the new module (e.g. `reverse_event_study_total_caregiving_merged_and_profiles`) that: takes df_o, df_c, outcome_o_series, outcome_c_series, window, age_min, age_max, path_to_specs (or start_age/end_age_caregiving), and returns (merged, prof_diff, prof_1_year_diff, …, prof_5_year_diff) and the distance column name used in the profiles, so the same plot function can be called for all outcomes.
   - Distance is computed from `add_distance_to_mother_death`; grouping from `identify_agents_by_total_caregiving_over_lifecycle`; no care_demand or first_care_period logic.

12. **Naming consistency**
   - Function names and filenames should clearly indicate: (1) event study, (2) outcome, (3) “by distance to mother death”, (4) “total caregiving”, (5) data variant (suffix or prefix back_to_Jan7).
   - Subfolder `total_caregiving_years` under `event_study_reverse/{outcome}/` keeps reverse total-caregiving-years plots separate from any other reverse plots (e.g. at_least, consecutive) in the same outcome folder.

---

## 13. Summary table (reverse module)

| Outcome       | Data        | Task function name (pattern)                                                                 | Output path (pattern)                                                                 |
|---------------|-------------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| employment    | estimated_params | task_plot_event_study_employment_rate_by_distance_to_mother_death_total_caregiving           | event_study_reverse/employment/total_caregiving_years/event_study_employment_rate_by_distance_to_mother_death_total_caregiving_{age}.pdf |
| employment    | back_to_Jan7     | task_plot_event_study_employment_rate_by_distance_to_mother_death_total_caregiving_back_to_Jan7 | event_study_reverse/employment/total_caregiving_years/back_to_Jan7_event_study_employment_rate_by_distance_to_mother_death_total_caregiving_{age}.pdf |
| full_time     | estimated_params | task_plot_event_study_full_time_by_distance_to_mother_death_total_caregiving                 | event_study_reverse/full_time/total_caregiving_years/event_study_full_time_by_distance_to_mother_death_total_caregiving_{age}.pdf |
| full_time     | back_to_Jan7     | ..._back_to_Jan7                                                                             | .../back_to_Jan7_event_study_full_time_by_distance_to_mother_death_total_caregiving_{age}.pdf |
| part_time     | (same)           | task_plot_event_study_part_time_by_distance_to_mother_death_total_caregiving [/_back_to_Jan7] | event_study_reverse/part_time/total_caregiving_years/... |
| working_hours | (same)           | task_plot_event_study_working_hours_by_distance_to_mother_death_total_caregiving [/_back_to_Jan7] | event_study_reverse/working_hours/total_caregiving_years/... |
| labor_income  | (same)           | task_plot_event_study_labor_income_by_distance_to_mother_death_total_caregiving [/_back_to_Jan7] | event_study_reverse/labor_income/total_caregiving_years/... |

Each task is defined in a loop over the four age groups; path_to_plot and other arguments are set in the function signature; task id includes age and outcome and data variant so all 40 tasks are distinct.

This completes the descriptive analysis and the outline for the reverse event-study module with total caregiving years, alternative outcomes, and two data pairs, without modifying or creating any code.
