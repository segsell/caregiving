# Task organization in `task_plot_employment_rate_by_distance_to_first_care.py`

This document describes how task functions are organized in this module, with emphasis on **total caregiving years** tasks (publication_employment_check and publication_other_check).

## Principle: one loop per task variant, arguments in the function signature

- **No `globals()` or dynamic function names.** Each task is a normal Python function with a clear, fixed name.
- **One loop per (outcome, event, data variant).** The loop runs over the four age groups only: `all_ages`, `ages_40_49`, `ages_50_59`, `ages_60_70`.
- **Path to plot and all other task arguments are written directly in the function signature** as default values. The loop variables (`age_min_val`, `age_max_val`, `age_label_val`) are used in those defaults so each iteration binds a different path and age filter.
- **Same function name in every iteration.** Each iteration redefines the same task function (e.g. `task_plot_full_time_share_by_distance_to_first_care_demand_total_caregiving`). Pytask collects one task per iteration because the `@pytask.task(id=...)` decorator runs each time and registers a new task with a **unique id** (e.g. `f"{age_label_val}_care_demand_total_caregiving_full_time"`).

## Loop pattern (total caregiving years)

```python
for age_min_val, age_max_val, age_label_val in (
    (None, None, "all_ages"),
    (40, 49, "ages_40_49"),
    (50, 59, "ages_50_59"),
    (60, 70, "ages_60_70"),
):

    @pytask.mark.publication_other_check
    @pytask.mark.publication_counterfactual
    @pytask.mark.publication_full_time
    @pytask.mark.publication
    @pytask.task(id=f"{age_label_val}_care_demand_total_caregiving_full_time")
    def task_plot_full_time_share_by_distance_to_first_care_demand_total_caregiving(
        age_min: int | None = age_min_val,
        age_max: int | None = age_max_val,
        age_label: str = age_label_val,
        path_to_original_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_estimated_params.pkl",
        path_to_no_care_demand_data: Path = BLD
        / "solve_and_simulate"
        / "simulated_data_no_care_demand.pkl",
        path_to_specs: Path = BLD / "model" / "specs" / "specs_full.pkl",
        path_to_plot: Annotated[Path, Product] = BLD
        / "figures"
        / "publication"
        / "counterfactual"
        / "full_time"
        / "total_caregiving_years"
        / (
            f"full_time_share_by_distance_to_first_care_demand_"
            f"total_caregiving_{age_label_val}.pdf"
        ),
        ever_caregivers: bool = True,
        ever_care_demand: bool = False,
        window: int = 20,
    ) -> None:
        """..."""
        # body: load specs, prepare data, compute profiles, plot
```

- **Task id** includes the age group so that the four tasks are distinct (e.g. `all_ages_care_demand_total_caregiving_full_time`, `ages_40_49_care_demand_total_caregiving_full_time`, ...).
- **path_to_plot** is spelled out in full; the filename uses `age_label_val` so each iteration gets a different path.
- **path_to_original_data** and **path_to_no_care_demand_data** are written directly (standard or back_to_Jan7 depending on the block).

## Total caregiving years: list of task variants

For each variant we have **one loop over the four age groups** and **one task function name**. Duration is always: 1, 2, 3, 4, 5+ **total** caregiving years over the lifecycle (from specs: `end_age_caregiving`). Sample: ever caregivers (`ever_caregivers=True`, `ever_care_demand=False`).

### Employment (publication_employment_check)

| Event              | Data        | Task function name (one per age loop) | Output dir / filename pattern |
|--------------------|------------|---------------------------------------|-------------------------------|
| First care demand   | standard   | `task_plot_employment_rate_by_distance_to_first_care_demand_total_caregiving` | `employment/total_caregiving_years/employment_rate_by_distance_to_first_care_demand_total_caregiving_{age}.pdf` |
| First care demand   | back_to_Jan7 | `task_plot_employment_rate_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7` | `.../back_to_Jan7_employment_rate_by_distance_to_first_care_demand_total_caregiving_{age}.pdf` |
| First caregiving spell | standard   | `task_plot_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving` | `.../employment_rate_by_distance_to_first_caregiving_spell_total_caregiving_{age}.pdf` |
| First caregiving spell | back_to_Jan7 | `task_plot_employment_rate_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7` | `.../back_to_Jan7_...` |

(Jan7 variants exist in the file as well; see code for exact names.)

### Other outcomes (publication_other_check): full_time, part_time, working_hours, labor_income

Same structure: for each (outcome, event, data) there is **one loop over the four age groups** and **one function name**.

- **First care demand, standard data:**  
  `task_plot_<outcome>_share_by_distance_to_first_care_demand_total_caregiving` (or `task_plot_working_hours_by_distance_to_...` / `task_plot_labor_income_by_distance_to_...`).  
  Output: `counterfactual/<outcome>/total_caregiving_years/<outcome>_share_by_distance_to_first_care_demand_total_caregiving_{age}.pdf` (or the corresponding working_hours / labor_income filename).
- **First care demand, back_to_Jan7:**  
  `task_plot_<outcome>_share_by_distance_to_first_care_demand_total_caregiving_back_to_Jan7` (same for working_hours / labor_income).  
  Filename prefix: `back_to_Jan7_`.
- **First caregiving spell, standard:**  
  `task_plot_<outcome>_share_by_distance_to_first_caregiving_spell_total_caregiving`.
- **First caregiving spell, back_to_Jan7:**  
  `task_plot_<outcome>_share_by_distance_to_first_caregiving_spell_total_caregiving_back_to_Jan7`.

So in total: **one loop per (outcome, event, data)**; inside the loop, **one named task function** with **path_to_plot and all other arguments written directly in the signature**; no globals, no factory, no dynamic names.

---

## Same pattern in `task_plot_event_study_total_caregiving_years.py`

The event-study (difference) tasks in that module follow the same organization:

- **One loop per (outcome, event, data variant)** over the four age groups only.
- **Same function name** in every iteration of a given loop (e.g. `task_plot_event_study_employment_rate_by_distance_to_first_care_demand_total_caregiving`).
- **path_to_plot, path_to_original_data, path_to_no_care_demand_data, path_to_specs**, and **age_min / age_max** are written **directly in the function signature** as default values using the loop variables `age_min_val`, `age_max_val`, `age_label_val`.
- **No `globals()` or dynamic function names.** Each task is a normal function; the decorator `@pytask.task(id=...)` uses a unique id per age group (e.g. `f"{age_label_val}_employment_first_care_demand_estimated_params"`).
