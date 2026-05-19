# Investigation: Reverse Employment Plots (Distance to Mother Death)

## Location

All plots in `bld/figures/publication/counterfactual/reverse_employment/` are created in:
**`src/caregiving/figures/publication/task_plot_employment_rate_by_distance_to_mother_death.py`**

## Naming and Groupings Summary

| Output filename pattern | Task function | Groupings (1,2,3,4,(5) lines) | Helper |
|-------------------------|---------------|-------------------------------|--------|
| `employment_rate_by_distance_to_mother_death_exact_caregiving_{age}.pdf` | `task_plot_employment_rate_by_distance_to_mother_death_exact_caregiving` | **EXACT caregiving** (informal care provision) before death: 1,2,3,4,5 years exactly, mutually exclusive | `identify_agents_by_caregiving_before_death` |
| `employment_rate_by_distance_to_mother_death_at_least_caregiving_{age}.pdf` | `task_plot_employment_rate_by_distance_to_mother_death_at_least_caregiving` | **AT LEAST caregiving** (informal care) before death: 1,2,3,4 years (overlapping) | `identify_agents_by_caregiving_before_death_at_least` |
| `employment_rate_by_distance_to_mother_death_total_caregiving_{age}.pdf` | `task_plot_employment_rate_by_distance_to_mother_death_total` | **TOTAL caregiving** years (cumulative, non-consecutive) before death: 1,2,3,4,5+ total care years | `identify_agents_by_total_caregiving_before_death` |
| `employment_rate_by_distance_to_mother_death_care_demand_exact_{age}.pdf` | `task_plot_employment_rate_by_distance_to_mother_death_care_demand_exact` | **EXACT care demand** (parent needs care) before death: 1,2,3,4,5 years exactly | `identify_agents_by_care_demand_before_death` |
| `employment_rate_by_distance_to_mother_death_care_demand_at_least_{age}.pdf` | `task_plot_employment_rate_by_distance_to_mother_death_care_demand_at_least` | **AT LEAST care demand** before death: 1,2,3,4,5 years (overlapping) | `identify_agents_by_care_demand_before_death_at_least` |

## Key Distinction

- **Caregiving** = informal care *provision* (agent actually provides care; choice ∈ INFORMAL_CARE)
- **Care demand** = parent *needs* care (care_demand > 0; includes formal care, sibling care, etc.)

The filenames `exact` and `at_least` without a qualifier were ambiguous—they refer to **caregiving** spell duration, not care demand.

## Changes Made

1. **exact** → **exact_caregiving**: filename and task now explicit that groupings are by caregiving spell
2. **at_least** → **at_least_caregiving**: same clarification
