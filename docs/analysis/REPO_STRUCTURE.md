# `caregiving` Repository Structure

A neutral structural map of the `caregiving` repo, intended both as a long-lived
reference and as the basis for spinning up a derived project (a slimmed-down
**pension model** without the caregiving choice). The migration triage lives in
the appendix at the end and can be ignored if you only need the repo map.

The codebase is large (~300 Python modules excluding the `dcegm` submodule).
The body of this document gives **full file-by-file listings** for the model
core, specs, estimation, simulation, and moments — those are the parts most
likely to be copied or refactored. Peripheral areas (figures, tables,
counterfactuals, pre/post-estimation) are grouped, with key files highlighted.

---

## 1. Overview

The project implements a **discrete-choice life-cycle model of female labor
supply, savings, and informal caregiving** for Germany. Parameters are
estimated by **method of simulated moments (MSM)** against SOEP (German
Socio-Economic Panel) and SHARE (Survey of Health, Ageing and Retirement in
Europe) data. Counterfactual policy experiments evaluate caregiving leave,
formal care costs, retirement age changes, and related policies.

**Tech stack:**

- `dcegm` (git submodule, editable install) — discrete-choice EGM solver.
- `JAX` + `numpy` + `numba` — numerical core.
- `optimagic` / `estimagic` / `tranquilo` / `DFO-LS` — optimization & estimation.
- `pytask` + `pytask-parallel` — workflow orchestration; build is invoked simply
  with `pytask` from the repo root.
- `pandas`, `pyarrow`, `statsmodels`, `lifelines` — data + auxiliary models.
- `plotly`, `seaborn` — figures.

`README.md` is a 26-line quickstart (`conda env create`, then `pytask`); there
is no `CLAUDE.md` and, prior to this document, no structural overview.

---

## 2. Top-Level Layout

### Directories

| Path | Role |
|------|------|
| `src/caregiving/` | Main package — model, estimation, data, figures (detailed below). |
| `src/sandbox/` | Experimental notebooks and scratch scripts. |
| `dcegm/` | **Git submodule** — discrete-choice EGM solver from OpenSourceEconomics; pinned via `.gitmodules`, installed editable. |
| `tests/` | Pytest suite (~13 test modules covering budget, utility, state space, transitions, moments, etc.). Includes `tests/data/`, `tests/debug/`, `tests/plotting/`, `tests/temp/`, `tests/temp_figs/`. |
| `scripts/` | Standalone diagnostic scripts (`check_experience_at_retirement.py`, `report_inheritance_sample_sizes.py`). |
| `docs/` | Manuscript chapters (`policy_experiments_chapter/`, thesis LaTeX, several documentation `*.md` files). |
| `examples/` | `scalar_optimization_example.py` and similar standalone demos. |
| `bld/` | **Pytask build output** — `model/`, `moments/`, `figures/`, `plots/`, `estimation/`, `solve_and_simulate/`, `tables/`, `counterfactual/`, `data/`, `latex/`, `event_study/`, `output/`, `descriptives/`. Generated, not source. |
| `.pytask/` | Pytask metadata (`file_hashes.json`, `pytask.sqlite3`). |
| `.github/`, `.vscode/`, `.cursor/`, `.idea/`, `.envs/` | IDE / CI configuration. |

### Top-level configuration files

| File | Role |
|------|------|
| `pyproject.toml` | Build (`setuptools_scm`), pytask config (`paths = ["./src/caregiving"]`), 70+ pytask markers, ruff/black/interrogate/yamlfix rules. |
| `environment.yml` | Conda environment definition (key dependencies above). |
| `setup.cfg` | Package metadata. |
| `tox.ini` | Test/docs build matrix. |
| `.pre-commit-config.yaml` | Pre-commit hooks (basic checks + pyupgrade; ruff/black currently commented). |
| `.yamllint.yml` | YAML linting rules. |
| `.gitmodules` | Pins the `dcegm` submodule to `https://github.com/OpenSourceEconomics/dcegm.git` `main`. |
| `.gitignore`, `.gitattributes`, `LICENSE`, `MANIFEST.in`, `CITATION` | Standard project files. |
| `README.md` | Minimal quickstart. |

### Stray ad-hoc files in repo root (not part of the codebase)

These are untracked or unstructured artefacts left over from interactive
analysis. Treat them as scratch — they are listed here so they aren't mistaken
for codebase modules.

- **Analysis writeups (~17 untracked `*.md`):** `AGE_INDEXING_ANALYSIS.md`,
  `ANALYSIS_ASSET_ACCUMULATION_COMPARISON.md`,
  `ANALYSIS_WORKING_SHARE_PLOTS_COMPARISON.md`,
  `analysis_partner_bad_health_variable.md`,
  `CAREGIVING_LEAVE_NORWAY_ALIGNMENT.md`,
  `CAREGIVING_LEAVE_WITH_JOB_RETENTION_MODEL.md`,
  `CONDITIONAL_MEANS_DISTANCE_INVESTIGATION.md`,
  `ELTERNGELD_ALIGNMENT_IMPLEMENTATION.md`,
  `FISCAL_CALCULATIONS_ANALYSIS.md`,
  `gross_labor_income_period_0_issue.md`,
  `MIN_MAX_AGE_USAGE.md`,
  `NORWEGIAN_LEAVE_NET_COST_IMPLEMENTATION.md`,
  `PARENT_AGE_OFFSET_ANALYSIS.md`,
  `PERIOD_INDEXING_ANALYSIS.md`,
  `REVERSE_EMPLOYMENT_PLOTS_INVESTIGATION.md`,
  `RUFF_ERRORS_OVERVIEW.md`,
  `START_AGE_CAREGIVING_PARENTS_ANALYSIS.md`.
- **Debug scripts (`debug_*.py`):** `debug_conditional_means_distance_to_first_care.py`,
  `debug_education_distribution.py`, `debug_initial_states_comparison.py`,
  `debug_tax_revenue.py`.
- **Linting helpers:** `fix_e501.py`, `.ruff_errors.json`.
- **Stray scratch:** `_test_task_create_rv_sample.py`, `across`, `also`, `d`,
  `individuals`, `threshold` (mostly empty or near-empty marker files).
- **Caches:** `.coverage`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`,
  `__pycache__/`, `.pytask.sqlite3`.
- **`bld/latex/counterfactuals_section (Copy 3).tex`** — stray copy.

---

## 3. Build & Task System (pytask)

The project is orchestrated by **pytask**. There is no separate pytask config
file; everything lives under `[tool.pytask.ini_options]` in `pyproject.toml`,
with `paths = ["./src/caregiving"]`.

- All `task_*.py` files under `src/caregiving/` are auto-discovered.
- `pyproject.toml` defines **70+ markers** to categorize tasks. Major groups:
  - Sample creation: `soep_moments`, `wealth_sample`, `cip`, `dip`,
    `estimation_sample`, `caregivers_sample`, `event_study_sample`.
  - Model variants: `baseline_model`, `no_inheritance`, `job_retention`,
    `caregiving_leave`, `no_care_demand`, `higher_ret_age`, `lower_ret_age`,
    `care_pension_credit`, `caregiving_leave_beirat`, `formal_care_costs`.
  - Simulation phases: `sim`, `initial_conditions`,
    `generate_initial_conditions_1m` / `_10k`, `job_separation`.
  - Analysis: `counterfactual_differences`, `model_fit`, `career_costs`,
    `government_budget`, `tables`, `policy_changes`.
  - Output: `figures`, `bar_chart`, `line_chart`, `publication`,
    `post_estimation`, `policy_changes`, `fiscal_costs`.
  - Misc: `debugging`, `debug_assets`, `explore`.
- Build output lives in `bld/` (regenerated by pytask, not committed).

---

## 4. The `dcegm` Dependency

- **Status:** Git submodule at `dcegm/`, pinned via `.gitmodules` to
  `https://github.com/OpenSourceEconomics/dcegm.git` `main` branch.
- **Install:** `pip install -e dcegm` (editable, declared in `environment.yml`).
- **Role:** Solves the discrete-choice EGM (endogenous grid method) life-cycle
  problem — the numerical engine the model is built on top of.
- **Note:** `git status` shows uncommitted modifications inside the submodule.
  Anyone consuming this repo (including a derived pension repo) should pin the
  same commit / set of patches.

---

## 5. Model Core — `src/caregiving/model/`

Most-edited area of the codebase. **Full file listing.** Subdirectories are
grouped together for readability.

### 5.1 `model/utility/`

| File | Role |
|------|------|
| `utility_functions.py` | Generic utility primitives (CRRA, etc.). |
| `utility_functions_additive.py` | Per-period utility aggregator: consumption term + labor disutility + caregiving disutility/utility. Main entry point used by the baseline model. |
| `utility_functions_additive_no_care_demand.py` | Same, but with all caregiving terms removed (matches the "no-care-demand" model variant). |
| `utility_components.py` (255 lines) | Decomposed disutility/utility terms: work disutility (FT/PT/unemployed) by education/health/partner state, plus all caregiving terms (light/intensive informal, formal, level shifts). **Imports `SEX` from `shared.py`** (line 8); `nb_children` indexed by `[SEX, education, has_partner, period]` (lines 50, 253). |
| `utility_components_no_care_demand.py` | Caregiving-stripped variant. |
| `bequest_utility.py` | Final-period bequest utility (uses inheritance state). |

### 5.2 `model/wealth_and_budget/`

| File | Role |
|------|------|
| `budget_equation.py` (241 lines) | Main budget constraint. **Sets `sex_var = SEX` at line 39**; passes `sex=sex_var` to wage, partner-income, unemployment-benefit, and pension calls (lines 55, 75, 93, 120). Aggregates labor income, pensions, partner income, child benefits, unemployment benefits, caregiving benefits/costs, inheritance. |
| `wages.py` | `calc_labor_income_after_ssc`, `calculate_gross_labor_income` — sex-, education-, experience-dependent wage; takes `sex` parameter. |
| `wages_no_care_demand.py` | Variant. |
| `partner_income.py` | `calc_partner_income_after_ssc` — partner wage/pension; `sex` is parameter at line 29. |
| `pension_payments.py` | Public pension calculation from accumulated experience (German Rentenpunkte logic). |
| `transfers.py` | Child benefits, unemployment benefits, **caregiving cash benefits (Pflegegeld)**, formal care costs, inheritance amounts. Caregiving-specific blocks live here. |
| `transfers_no_care_demand.py` | Care-stripped variant. |
| `tax_and_ssc.py` | Income-tax + social-security-contribution deductions. |
| `savings_grid.py` | Asset/savings grid specification for EGM. |
| `government_budget.py` | Tracks fiscal flows (caregiving leave context). |
| `government_budget_caregiving_leave_with_job_retention.py` | Variant. |
| `caregiving_leave_top_up.py` | Cash top-up policy. |
| `budget_equation_no_care_demand.py` | Variant: caregiving removed. |
| `budget_equation_no_inheritance.py` | Variant: bequest motive removed. |
| `budget_equation_no_care_demand_no_inheritance.py` | Combined variant. |
| `budget_equation_no_cash_benefits.py` | Variant: no Pflegegeld. |
| `budget_equation_higher_formal_care_costs.py` / `..._lower_formal_care_costs.py` | Formal-care-cost counterfactuals. |
| `budget_equation_caregiving_leave_with_job_retention.py` / `..._full_..._with_job_retention.py` / `..._beirat.py` / `..._full_beirat.py` | Caregiving-leave policy variants. |

### 5.3 `model/stochastic_processes/`

| File | Role |
|------|------|
| `health_transition.py` | Agent health: bad / good / dead. Indexed by `[sex, education, period, current_health]`. |
| `partner_transition.py` | Partner state: no partner / working / retired. |
| `job_transition.py` | Job offer probabilities (sex- and education-dependent). |
| `job_transition_no_care_demand.py` / `job_transition_job_retention.py` | Variants. |
| `adl_transition.py` | Parent ADL limitations (none / light / intensive). **Caregiving-specific.** |
| `caregiving_transition.py` | Maps parent ADL → care demand state. **Caregiving-specific.** |
| `inheritance_transition.py` | Mother death + inheritance payout (choice-conditional). |
| `inheritance_transition_no_care_demand.py` | Variant. |

### 5.4 `model/state_space.py` and variants

`model/state_space.py` (902 lines) is the central state-space + choice-set
module.

- **Deterministic states** (carried to next period): `caregiving_type` ∈ {0, 1},
  `education` ∈ {0, 1}, `already_retired` ∈ {0, 1}.
- **Stochastic states**: `partner_state` ∈ {0,1,2}, `health` ∈ {0,1,2},
  `job_offer` ∈ {0,1}, `mother_dead` ∈ {0,1,2}, `mother_adl` ∈ {0,1,2},
  `care_demand` ∈ {0,1,2}.
- **Continuous states**: `assets_end_of_period`, `experience`.
- **Choice-set construction** is in
  `def state_specific_choice_set_with_caregiving(...)` at **line 544**
  (an older commented-out version sits at line 411). The function takes
  `caregiving_type` as a parameter (line 549) and branches at line 593
  (`if caregiving_type == 1:` — the "can provide informal care" case)
  vs. the type-0 branch later in the function.
- `caregiving_type` is also used heavily inside `next_period_deterministic_state`
  and the sparsity / admissibility helpers (lines 134, 183, 216, 250, 269,
  287, 334, 351 …).

Variant state-space modules (one per model spec):

| File | Role |
|------|------|
| `state_space_no_inheritance.py` | Drops bequest channel. |
| `state_space_no_care_demand.py` | Drops parent / care-demand states entirely. |
| `state_space_higher_ret_age.py` / `state_space_lower_ret_age.py` | Retirement-age policy variants. |
| `state_space_job_retention.py` | Caregiving leave with job-retention guarantee. |
| `state_space_caregiving_leave_with_job_retention.py` | Combined caregiving-leave + job-retention. |
| `state_space_caregiving_leave_beirat.py` | "Beirat" leave proposal variant. |
| `state_space_care_pension_credit.py` | Adds care-pension credits to retirement points. |

### 5.5 `model/shared.py`

Constants and helpers shared across the model.

- `MALE = 1`, `FEMALE = 2`, **`SEX = 1` at line 32** — the canonical
  hard-coded "agent is female" assumption.
- Other constants: `MOTHER`, age bounds (`MIN_AGE = 40`, `MAX_AGE = 70`,
  `MIN_AGE_PARENTS`, `MAX_AGE_PARENTS`), `PERIOD_SCALE`, `LEAVE_CAP_YEARS`,
  parent-weights, machine-zero / fill / missing-value sentinels.
- State decoders / predicates (`is_retired`, `is_informal_care`,
  `PARENT_RECENTLY_DEAD`, choice-set enum membership helpers).

`shared_no_care_demand.py` is the parallel constants module for the
care-stripped variant.

### 5.6 Other model-level modules

| File | Role |
|------|------|
| `experience_baseline_model.py` | Experience accumulation rules (baseline). |
| `experience_no_care_demand.py` / `experience_no_inheritance.py` / `experience_caregiving_leave_model.py` | Variants. |
| `taste_shocks.py` | Taste-shock specification for EGM. |
| `pension_system/experience_stock.py` / `pension_system/early_retirement_paths.py` | German pension-system mechanics. |
| `task_specify_model.py` | **Top-level model-spec entry point.** Defines `caregiving_type` as a discrete deterministic state at line 85: `"caregiving_type": np.arange(2, dtype=int)`. |
| `task_specify_model_no_care_demand.py`, `task_specify_model_no_inheritance.py`, `task_specify_model_higher_ret_age.py`, `task_specify_model_lower_ret_age.py`, `task_specify_model_higher_formal_care_costs.py`, `task_specify_model_lower_formal_care_costs.py`, `task_specify_model_no_cash_benefits_higher_formal_care_costs.py`, `task_specify_model_job_retention.py`, `task_specify_model_caregiving_leave_with_job_retention.py`, `task_specify_model_full_caregiving_leave_with_job_retention.py`, `task_specify_model_caregiving_leave_beirat.py`, `task_specify_model_caregiving_leave_full_beirat.py`, `task_specify_model_no_care_demand_no_inheritance.py`, `task_specify_model_care_pension_credit.py` | Per-variant model-spec entry points. |
| `_task_model_fit.py`, `_task_check_solve.py`, `_task_debugging.py` | Internal/debug tasks. |

---

## 6. Specifications — `src/caregiving/specs/`

Specs are **Python builders** (not YAML), compiled at runtime into a single
`specs_full.pkl` dictionary by `task_write_specs.py`. Full file listing:

| File | Role |
|------|------|
| `task_write_specs.py` (~492 lines) | Top-level specs compilation task. |
| `derive_specs.py` | Orchestrator that calls each spec builder and assembles the full dict. |
| `health_specs.py` | Sex-indexed health-transition matrices, survival probabilities. |
| `family_specs.py` | Partner / children demographics. |
| `income_specs.py` | Wage levels, pension replacement, transfer amounts. |
| `inheritance_specs.py` | Inheritance amount tables (choice-conditional). |
| `care_costs_specs.py` | Formal care cost schedules. |
| `experience_pp_specs.py` | Experience accumulation rules / pension points. |
| `caregiving_specs.py` (661 lines) | ADL transitions, care supply, caregiving-type distribution; iterates `for sex_idx, sex_label in enumerate(specs["sex_labels"])` (line 48 area). **Caregiving-specific.** |
| `task_plot_inheritance_amount.py`, `task_plot_weighted_adl_transitions.py` | Spec-side diagnostic plotting tasks. |

Estimated parameters land in
`bld/model/params/estimated_params_model.yaml`.

---

## 7. Estimation — `src/caregiving/estimation/`

Full file listing:

| File | Role |
|------|------|
| `estimation_setup.py` | MSM setup: optimizer interface (`optimagic`), parameter bounds, weighting matrix, initial-states / empirical-moment loading. |
| `estimation_setup_no_care_demand.py` | Setup for the no-care-demand variant. |
| `criterion.py` | Criterion function (residuals, weighting, Cholesky decomp). |
| `prepare_estimation.py` | Pre-estimation model preparation glue. |
| `standard_errors.py` | Post-estimation SE calculations. |
| `start_params_and_bounds/task_set_start_params.py` | Parameter starting values and bounds (contains caregiving-specific entries like `utility_informal_care_parent`). |
| `task_solve_and_simulate.py` | Pytask: solve baseline, simulate panel. |
| `task_solve_and_simulate_estimated_params.py` | Same, at the estimated parameter vector. |
| `task_solve_and_simulate_no_inheritance.py` | Variant. |
| `task_estimate_standard_errors.py`, `_task_estimate_standard_errors.py` | Standard-error tasks. |

---

## 8. Simulation & Moments

### 8.1 `src/caregiving/simulation/`

Full file listing:

**Core machinery:**
| File | Role |
|------|------|
| `simulate.py` | Forward simulation: solve → simulate → construct panel; handles dead agents, derived variables. |
| `_simulate.py` | Helper functions for panel construction. |
| `simulation_utils.py` | State extraction, discrete-state handling at forced periods. |
| `_initial_conditions.py` | Initial-state-distribution construction. |
| `simulate_no_care_demand.py` | Variant simulation entry point. |
| `simulate_forced_care_demand_at_50.py` | Counterfactual experiment: force care demand at age 50. |
| `plot_model_fit.py` | Reusable plotting functions for empirical-vs-simulated comparisons. |

**Moment computation (called from criterion / pytask):**
| File | Role |
|------|------|
| `simulate_moments.py` | Compute MSM moments from simulated panel (labor supply, hours, income, wealth, caregiving hours). |
| `simulate_moments_no_care_demand.py` | Care-stripped variant. |
| `simulate_moments_restricted.py` | Smaller moment set. |
| `simulate_moments_alternative.py` | Alternative moment definitions. |

**Pytask entry points (initial conditions, moments, model fit):**
| File | Role |
|------|------|
| `task_generate_initial_conditions.py` | Generate initial states (1m / 10k size) from empirical distribution. |
| `task_generate_initial_conditions_no_care_demand.py` | Variant. |
| `task_generate_initial_conditions_job_retention.py` | Variant. |
| `task_simulate_moments.py` | Main moment-simulation task. |
| `task_simulate_moments_estimated_params.py` | Same, at estimated params. |
| `task_simulate_moments_estimated_params_no_care_demand.py` | Variant. |
| `task_simulate_moments_job_retention_estimated_params.py` | Variant. |
| `task_plot_model_fit.py` | Empirical-vs-simulated moment plot (baseline). |
| `task_plot_model_fit_estimated_params.py` | At estimated params. |
| `task_plot_model_fit_estimated_params_moments.py` | Extended diagnostics. |
| `task_plot_model_fit_no_care_demand.py` | Variant. |
| `task_plot_model_fit_higher_ret_age.py` | Variant. |
| `task_plot_model_fit_job_retention.py` | Variant. |
| `task_plot_model_fit_caregiving_leave_with_job_retention.py` | Variant. |
| `task_plot_initial_states.py` | Diagnostic: initial-state distribution. |
| `task_generate_plots_for_slides.py` | Presentation plots. |
| `publication/task_plot_model_fit_publication.py` | Publication-grade model-fit figure. |

### 8.2 `src/caregiving/moments/`

Full file listing:

| File | Role |
|------|------|
| `transform_data.py` | Generic data transformations: load, scale experience, correct wealth, dcegm asset-correction. |
| `task_create_cpi_germany.py` | German CPI deflator. |
| `task_create_soep_moments.py` | Main empirical-moments task: labor supply, hours, income, wealth, caregiving hours. |
| `task_create_soep_moments_no_care_demand.py` | Variant excluding caregiving moments. |
| `task_create_soep_moments_restricted.py` | Smaller moment set. |
| `task_create_share_moments.py` | Moments from SHARE (parent health, care demand). **Caregiving-specific.** |
| `_task_create_empirical_moments.py` | SHARE-based empirical moments. **Caregiving-specific.** |
| `_task_create_auxiliary_moments.py` | Auxiliary moments (parent health, distance-to-first-care). **Caregiving-specific.** |
| `_task_initial_conditions.py` | Empirical initial-state distribution. |
| `_task_create_empirical_cov.py` | Empirical covariance matrix for the MSM weighting matrix. |
| `_task_care_mix_statistical_office.py` | Care-mix statistics from external admin data. **Caregiving-specific.** |
| `_task_create_care_mix_coefficients.py` | Estimate care-mix regression coefficients. **Caregiving-specific.** |
| `_task_simulate_moments.py` | Sandbox simulate-moments task. |
| `task_plot_empirical_moments.py` | Diagnostic plot of empirical moments. |

---

## 9. Data Management — `src/caregiving/data_management/`

### 9.1 `data_management/soep/` — full file listing

**Library / helpers:**
| File | Role |
|------|------|
| `variables.py` | Variable construction (employment choice, health, partner state, education, policy state, working hours, wealth). |
| `auxiliary.py` | Lag/lead helpers, choice-restriction enforcement, age/year filters. |
| `wealth.py` | Wealth deflation and transformation. |
| `soep_variables/experience.py` | Experience-stock construction (with caregiving-time deductions). |

**Sample-creation tasks:**
| File | Role |
|------|------|
| `task_create_wage_sample.py` | Sample for wage-process estimation. |
| `task_create_partner_wage_sample.py` | Partner wage sample. |
| `task_create_health_sample.py` | Health-transition sample. |
| `task_create_partner_transition_sample.py` | Partner-state-transition sample. |
| `task_create_survival_transition_sample.py` | Mortality / survival sample. |
| `task_create_wealth_sample.py` | Wealth sample. |
| `task_create_job_separation_sample.py` | Job-separation sample. |
| `task_create_innovation_sample.py` | Innovation sample (for diagnostic checks). |
| `task_create_structural_estimation_sample.py` | Main estimation sample (includes caregiving variables). |
| `task_create_event_study_sample.py` | Event-study sample around caregiving start / parent death. |
| `task_create_informal_care_sample.py` | (referenced in earlier exploration; informal-care-hours sample). |
| `task_create_inheritance_sample.py` | Inheritance receipt sample. |
| `task_create_formal_care_costs_sample.py` | Formal-care-cost data. |
| `task_create_rv_sample.py` | German pension-account (Rentenversicherung) sample. |

**Merge tasks:**
| File | Role |
|------|------|
| `merge_data/task_load_and_merge_structural_samples.py` | Merge structural samples. |
| `merge_data/task_load_and_merge_structural_innovation_samples.py` | Variant including innovation. |
| `merge_data/task_load_and_merge_event_study_sample.py` | Event-study merge. |
| `merge_data/task_load_and_merge_rv_data.py` | RV merge. |

### 9.2 `data_management/share/` — grouped (all caregiving-specific)

`task_merge_data.py`, `task_merge_waves.py`,
`task_merge_parent_child_waves.py`, `task_create_parent_child_data_set.py`,
`task_create_estimation_data_set.py`, plus a backup module
`backup_create_parent_child_data_set.py`. All deal with parent-child linked
SHARE data and exist exclusively to support caregiving estimation.

### 9.3 Raw data — `src/caregiving/data/`

The raw data directory ships SHARE wave releases (`sharew1` … `sharew9`,
plus `share-wave-8/`), SOEP (`soep/`, `soep_c40/`, `soep_is/`,
`soep_rv_vskt/`), and `statistical_office/`. Source data only — read by the
data-management tasks above.

---

## 10. Stochastic-Process Estimation — `src/caregiving/stochastic_processes/`

Standalone reduced-form estimation of the exogenous processes that feed the
structural model. Full file listing:

| File | Role |
|------|------|
| `auxiliary.py` | Helpers shared across the estimation tasks. |
| `task_estimate_wage_process_soep.py` | Wage AR(1) process by sex/education. |
| `task_estimate_partner_wage_soep.py` | Partner-wage process. |
| `task_estimate_survival_soep_logit.py` | Mortality logit (sex-, age-, education-dependent). |
| `task_estimate_survival_kroll_lampert.py` | Alternative mortality model (Kroll/Lampert). |
| `task_estimate_health_transition_soep_good_bad.py` | Two-state health transitions. |
| `task_estimate_health_transition_soep_good_medium_bad.py` | Three-state health transitions. |
| `task_estimate_job_offer_soep.py` | Job-offer probability. |
| `task_estimate_job_separation_soep.py` | Job separation. |
| `task_estimate_family_transitions_soep.py` | Partner-state transitions. |
| `task_estimate_parental_health_transitions_share.py` | Parent health (SHARE). **Caregiving-specific.** |
| `task_estimate_limitations_with_adl_share.py` | ADL transitions. **Caregiving-specific.** |
| `task_estimate_exog_care_supply.py` | Exogenous care supply. **Caregiving-specific.** |
| `task_estimate_inheritance_soep.py` | Inheritance receipt. **Caregiving-related.** |
| `task_estimate_formal_care_costs.py` | Formal care costs. **Caregiving-specific.** |
| `_task_create_exog_processes.py` | Bundle all exog processes into one object consumed by the model. |

---

## 11. Figures & Tables (grouped)

### 11.1 `src/caregiving/figures/` — base level

Diagnostic plots, mostly general-purpose:

- **General DC-EGM diagnostics:** `task_plot_death_transition.py`,
  `task_plot_expected_health.py`, `task_plot_family_transition.py`,
  `task_plot_income.py`, `task_plot_wealth.py`, `task_plot_job_offer.py`,
  `task_plot_initial_conditions.py`, `task_plot_pension_npv.py`,
  `task_plot_pension_rates.py`.
- **Caregiving-specific:** `task_plot_adl_transition.py`,
  `task_plot_inheritance.py`, `task_plot_estimated_inheritance.py`,
  `task_plot_inheritance_specifications.py`,
  `task_plot_inheritance_specifications_no_care.py`,
  `task_plot_inheritance_two_specs.py`,
  `task_plot_inheritance_probability_no_care_demand.py`,
  `task_plot_utility.py` (plots caregiving-utility components).

### 11.2 `src/caregiving/figures/publication/`

Publication-grade plots. **Mostly caregiving-specific** (event studies,
distance-to-first-care, cost-effectiveness, mother-death event studies):

- Reusable: `plotting_helpers.py`, `plotting_functions.py`,
  `task_plot_model_fit_labor_supply.py`,
  `appendix/task_plot_stochastic_processes_publication.py`.
- Caregiving-specific: `task_plot_pre_estimation.py`,
  `task_plot_caregiving_by_distance.py`,
  `task_plot_care_demand_post_estimation.py`,
  `task_plot_employment_rate_by_distance_to_first_care.py`,
  `task_plot_employment_rate_by_distance_to_mother_death.py`,
  `task_plot_event_study_employment_rate_consecutive.py`,
  `task_plot_event_study_employment_rate_mother_death.py`,
  `task_plot_event_study_total_caregiving_years.py`,
  `task_plot_event_study_caregiving_leave.py`,
  `task_plot_event_study_full_caregiving_leave.py`,
  `_task_plot_event_study_caregiving_leave.py`,
  `task_plot_reverse_event_study_total_caregiving_years.py`,
  `task_plot_reverse_by_distance_to_mother_death.py`,
  `task_plot_cost_effectiveness_scatter.py`,
  `task_plot_differences_wealth_no_inheritance.py`,
  `plotting_helpers_mother_death.py`,
  `final/task_conditional_means_distance_to_first_care_demand.py`,
  `post_estimation/task_plot_care_demand_spells.py`,
  `post_estimation/task_plot_asset_and_savings_dec_differences_baseline_vs_no_care_demand.py`.

### 11.3 `src/caregiving/tables/` and `tables/publication/`

Almost entirely caregiving-leave / fiscal:

- General: `publication/task_stochastic_processes_tables.py`.
- Caregiving-policy: `publication/task_government_budget.py`,
  `publication/task_government_budget_caregiving_leave.py`,
  `publication/task_government_budget_caregiving_leave_top_up.py`,
  `publication/task_government_budget_caregiving_leave_labor_supply.py`,
  `publication/task_government_budget_normal_leave_vs_baseline.py`,
  `publication/task_fiscal.py`, `publication/task_policy_changes.py`,
  `publication/task_self_financing.py`,
  top-level `task_explore_government_budget.py`.

---

## 12. Counterfactuals — `src/caregiving/counterfactual/`

**~50 caregiving-policy-specific tasks** (caregiving leave, full caregiving
leave, beirat variants, job retention, formal-care-cost shifts, retirement-age
shifts, no-care-demand world, care-pension credits, career-cost computation,
forced-care-demand-at-50). Not enumerated file by file because the area is
unlikely to be ported into a pension-only model.

Key files: `simulate_counterfactual.py`, `_task_simulate_counterfactual.py`,
`task_specify_model_counterfactual.py`, `task_compute_career_costs.py`,
`task_plots_for_presentation.py`, `plotting_helpers.py`,
`plotting_utils.py`. Subdirectories: `matched_differences/`,
`matched_differences_end_of_caregiving/`, `debugging/`. There is also a
duplicate `plotting_helpers (Copy).py` (stray).

---

## 13. Pre- and Post-Estimation

### `src/caregiving/pre_estimation/`

- `task_plot_savings_grid.py` — general (asset-grid diagnostic).
- `task_working_hours_table.py` — general.
- `task_plot_care_demand.py`, `task_siblings.py`,
  `task_plot_inheritance_probability_no_care_demand.py`,
  `task_plot_mother_dead_probability_no_care_demand.py` — caregiving-specific.

### `src/caregiving/post_estimation/`

- General age-profile diagnostics: `task_plot_age_profiles.py`,
  `task_plot_exp_years_by_age.py`, `task_plot_income_by_age.py`,
  `task_plot_assets_and_savings_by_age.py`,
  `task_plot_partner_state_by_age.py`.
- Caregiving-specific: `task_plot_care_demand_post.py`,
  `task_plot_care_demand_by_age_2_by_2.py`,
  `task_plot_labor_supply_transitions_after_care_demand.py`,
  `task_plot_caregiving_leave_top_up.py`, `task_post_inheritance.py`,
  `task_post_inheritance_no_care_demand.py`,
  `task_post_inheritance_caregiving_leave.py`.
- `publication/`: `task_create_params_table.py` (general),
  `task_create_moments_table.py` (general),
  `task_create_job_offer_params_table.py` (general),
  `task_create_care_utility_params_table.py` (caregiving-specific),
  `task_describe_caregiving.py` (caregiving-specific).

### `src/caregiving/descriptives/`

Descriptive analyses, mostly caregiving-themed:
`task_summary_statistics_intensive_care.py`,
`task_intensive_care_within_outside_household.py`,
`task_intra_family_caregiving.py`,
`task_female_male_caregiving_ratios.py`,
`task_share_women_with_young_children_and_parents_bad_health.py`,
`task_sibling_comparison_by_education.py`,
`task_care_arrangements_by_age_latex.py`, `task_moving_behavior_analysis.py`,
`task_soep_is.py`, `publication/task_summary_statistics.py`.

### `src/caregiving/temp/`

`task_recreate_simulated_data_frames.py` — single utility script.

---

## 14. Top-Level Package Files

| File | Role |
|------|------|
| `src/caregiving/__init__.py` | Package init. |
| `src/caregiving/_version.py` | Version (managed by `setuptools_scm`). |
| `src/caregiving/config.py` | Build / source paths, color maps, project-wide constants. |
| `src/caregiving/utils.py` | Shared utilities (formatting tables, describe/count helpers, statsmodels conversion, pickle helpers). |
| `src/caregiving/_simulate.py`, `src/caregiving/_estimate.py`, `src/caregiving/_task_estimate.py` | Archived / alternate scaffolding modules (kept for reference). |

---

# Appendix — Migration Triage (Pension Model)

This appendix captures the migration-specific reading of the structure above.
It exists so the main body can stay neutral. Three changes drive the appendix:

1. **No caregiving choice** — the new model is pure labor supply +
   consumption / savings. All caregiving states, choices, transitions,
   benefits, costs, parent demographics, and inheritance-by-care-conditioning
   come out.
2. **Sex distinction** — currently `SEX = 1` is hard-coded in
   `model/shared.py:32`. The new model treats `sex` as a state variable
   covering both men and women.
3. **Heterogeneity-type repurposing** — `caregiving_type ∈ {0, 1}` currently
   gates the choice set in `state_space.py:544–750`. In the pension model it
   is renamed `leisure_pref_type` (or similar) and enters the **utility**
   function as a leisure-preference shifter, not the choice set. Implementation
   details deferred per user instruction.

## A. Module-level KEEP / REFACTOR / DROP

| Area | Verdict | Reason |
|------|---------|--------|
| `model/utility/utility_functions_additive.py` | REFACTOR | Strip caregiving terms; add `sex` and `leisure_pref_type` arguments. |
| `model/utility/utility_components.py` | REFACTOR | Keep the work-disutility scaffolding, drop care-utility blocks (the lower portion of the file). Replace hard-coded `SEX` indexing with parameter. |
| `model/utility/*_no_care_demand.py` | KEEP (as starting point) | Closer to the target — caregiving already removed. |
| `model/utility/bequest_utility.py` | REFACTOR | Decide whether bequests stay; if yes, decouple from inheritance-by-care logic. |
| `model/wealth_and_budget/budget_equation.py` | REFACTOR | Drop care-benefit / care-cost / care-conditional-inheritance terms; thread `sex` everywhere instead of `SEX`. |
| `model/wealth_and_budget/wages.py`, `partner_income.py`, `pension_payments.py`, `tax_and_ssc.py`, `savings_grid.py` | KEEP / REFACTOR | General income-side machinery; mostly drop-in once `sex` is parameterized. |
| `model/wealth_and_budget/transfers.py` | REFACTOR | Drop Pflegegeld, formal-care-cost, care-conditional-inheritance blocks. Keep child / unemployment benefits. |
| `model/wealth_and_budget/government_budget*.py` | DROP | Caregiving-leave fiscal accounting. |
| `model/wealth_and_budget/caregiving_leave_top_up.py` | DROP | Policy-specific. |
| `model/wealth_and_budget/budget_equation_*.py` variants | DROP | All caregiving-policy variants. |
| `model/stochastic_processes/health_transition.py`, `partner_transition.py`, `job_transition.py` | KEEP / REFACTOR | General; ensure sex is a real state index and not the hard-coded literal. |
| `model/stochastic_processes/adl_transition.py`, `caregiving_transition.py`, `inheritance_transition*.py` | DROP | Parent / care-demand machinery. |
| `model/state_space.py` | REFACTOR | Largest single rewrite. Remove care-state branches in choice-set construction; remove parent / mother-dead / care-demand / mother-adl from state lists; add `sex` to the deterministic-state list. |
| `model/state_space_*.py` variants | DROP (mostly) | Policy variants are caregiving-specific. `state_space_no_care_demand.py` is a useful reference template. |
| `model/shared.py` | REFACTOR | Remove `SEX = 1` literal; keep age constants; remove parent / care helpers. |
| `model/experience_baseline_model.py` | REFACTOR | Drop caregiving-credit logic. |
| `model/experience_*.py` variants | DROP (most) | Policy variants. |
| `model/pension_system/*.py` | KEEP | Pure pension mechanics — central to the new model. |
| `model/taste_shocks.py` | KEEP | General. |
| `model/task_specify_model.py` | REFACTOR | Drop caregiving states; add `sex`; rename `caregiving_type` → leisure-preference type. |
| `model/task_specify_model_*.py` variants | DROP | Caregiving-policy variants. |
| `specs/derive_specs.py`, `health_specs.py`, `family_specs.py`, `income_specs.py`, `experience_pp_specs.py` | KEEP / REFACTOR | General; broaden indexing to both sexes. |
| `specs/inheritance_specs.py` | DROP if bequests removed; otherwise REFACTOR | Currently care-conditional. |
| `specs/care_costs_specs.py`, `caregiving_specs.py` | DROP | Caregiving-specific. |
| `specs/task_write_specs.py` | REFACTOR | Trim spec assembly to the kept builders. |
| `estimation/estimation_setup.py`, `criterion.py`, `prepare_estimation.py`, `standard_errors.py` | KEEP | General MSM machinery. |
| `estimation/estimation_setup_no_care_demand.py` | KEEP (as starting point) | Already care-stripped. |
| `estimation/start_params_and_bounds/task_set_start_params.py` | REFACTOR | Drop caregiving params (`utility_informal_care_parent`, formal-care-cost params, etc.); add per-sex / per-leisure-type entries. |
| `estimation/task_solve_and_simulate.py`, `..._estimated_params.py` | KEEP / REFACTOR | General. |
| `estimation/task_solve_and_simulate_no_inheritance.py` | DROP unless mirrored. |
| `simulation/simulate.py`, `_simulate.py`, `simulation_utils.py`, `_initial_conditions.py`, `plot_model_fit.py` | KEEP | General. |
| `simulation/simulate_moments.py` | REFACTOR | Drop caregiving-hour moments; ensure moments are computed by sex. |
| `simulation/simulate_moments_no_care_demand.py` | KEEP (as starting point) | Closer to the target. |
| `simulation/simulate_moments_restricted.py`, `..._alternative.py` | KEEP if useful; otherwise drop. |
| `simulation/simulate_forced_care_demand_at_50.py`, `simulate_no_care_demand.py` | DROP | Caregiving-experiment. |
| `simulation/task_generate_initial_conditions.py` | REFACTOR | Initialize both sexes; drop caregiving-type-as-choice-gate logic; introduce leisure-preference-type draw. |
| `simulation/task_generate_initial_conditions_*.py` variants | DROP | Policy variants. |
| `simulation/task_simulate_moments*.py`, `task_plot_model_fit*.py` | KEEP / REFACTOR | General; drop policy variants. |
| `simulation/publication/task_plot_model_fit_publication.py` | KEEP | General. |
| `moments/transform_data.py`, `_task_initial_conditions.py`, `_task_create_empirical_cov.py`, `task_create_cpi_germany.py`, `task_plot_empirical_moments.py` | KEEP | General. |
| `moments/task_create_soep_moments.py` | REFACTOR | Strip caregiving-hour moments; ensure all moments are computed by sex. |
| `moments/task_create_soep_moments_no_care_demand.py` | KEEP (as starting point) | Care-stripped. |
| `moments/task_create_share_moments.py`, `_task_create_empirical_moments.py`, `_task_create_auxiliary_moments.py`, `_task_care_mix_*.py` | DROP | Caregiving-specific (SHARE-based). |
| `data_management/soep/variables.py`, `auxiliary.py`, `wealth.py`, `soep_variables/experience.py` | KEEP / REFACTOR | General; drop caregiving-time deductions in `experience.py`. |
| `data_management/soep/task_create_{wage,partner_wage,health,partner_transition,survival_transition,wealth,job_separation,innovation,structural_estimation}_sample.py` | KEEP / REFACTOR | Generally useful; drop caregiving variables when assembled. |
| `data_management/soep/task_create_{event_study,informal_care,formal_care_costs,inheritance,rv}_sample.py` | DROP (mostly) | Event-study / care-cost / inheritance / RV are caregiving- or German-specific. RV may be retained if you keep Rentenversicherung mechanics. |
| `data_management/soep/merge_data/*.py` | KEEP / REFACTOR | Plumbing; drop caregiving merges. |
| `data_management/share/**` | DROP | All SHARE work is caregiving-specific. |
| `stochastic_processes/task_estimate_wage_process_soep.py`, `..._partner_wage_soep.py`, `..._survival_soep_logit.py`, `..._survival_kroll_lampert.py`, `..._health_transition_soep_*.py`, `..._job_offer_soep.py`, `..._job_separation_soep.py`, `..._family_transitions_soep.py` | KEEP | General reduced-form estimations. |
| `stochastic_processes/task_estimate_parental_health_transitions_share.py`, `..._limitations_with_adl_share.py`, `..._exog_care_supply.py`, `..._inheritance_soep.py`, `..._formal_care_costs.py` | DROP | Caregiving-specific. |
| `stochastic_processes/_task_create_exog_processes.py` | REFACTOR | Bundle only the kept processes. |
| `figures/` (general diagnostics) | KEEP / REFACTOR | Health, mortality, family, income, wealth, job offer, pension. |
| `figures/` (caregiving) and `figures/publication/*` (event studies, distance-to-care, cost-effectiveness) | DROP | Caregiving-specific. |
| `tables/publication/*` (fiscal / government budget / policy changes) | DROP | Caregiving-policy. |
| `tables/publication/task_stochastic_processes_tables.py` | KEEP | General. |
| `post_estimation/task_create_params_table.py`, `..._moments_table.py`, `..._job_offer_params_table.py` | KEEP | General. |
| `post_estimation/task_create_care_utility_params_table.py`, `task_describe_caregiving.py`, all `..._care_demand*.py`, `..._post_inheritance*.py`, `..._caregiving_leave_top_up.py` | DROP | Caregiving-specific. |
| `post_estimation/` general age-profile tasks (`task_plot_age_profiles.py`, `..._exp_years_by_age.py`, `..._income_by_age.py`, `..._assets_and_savings_by_age.py`, `..._partner_state_by_age.py`) | KEEP / REFACTOR | General. |
| `pre_estimation/task_plot_savings_grid.py`, `task_working_hours_table.py` | KEEP | General. |
| `pre_estimation/` caregiving plots (`task_plot_care_demand.py`, `task_siblings.py`, etc.) | DROP | Caregiving-specific. |
| `descriptives/**` | DROP | Almost entirely caregiving-themed; the new project will need its own descriptives. |
| `counterfactual/**` (~50 modules) | DROP | All caregiving-policy. |
| `temp/`, `_estimate.py`, `_simulate.py`, `_task_estimate.py` | DROP | Archive / scratch. |
| `config.py`, `utils.py` | KEEP | Project-wide infrastructure (paths, colors, helpers). |
| `pyproject.toml`, `environment.yml`, `setup.cfg`, `tox.ini`, `.pre-commit-config.yaml`, `.yamllint.yml` | KEEP / REFACTOR | Reuse the toolchain; trim pytask markers to what the new project actually has. |
| `dcegm/` submodule + `.gitmodules` | KEEP | Re-pin in the new repo. |

## B. Sex-hardcoding sites — files to update

`SEX = 1` lives in `src/caregiving/model/shared.py:32`. It is imported in
**33 files** under `model/`. The sites that must change (each currently
imports `SEX` and uses it as a constant; in the new project all should accept
a `sex` argument from the state vector):

- `model/shared.py:32` — replace literal with state-variable usage; keep
  `MALE = 1`, `FEMALE = 2` constants.
- `model/utility/utility_components.py:8, 50, 253` — `SEX` in `model_specs`
  index lookups (`children_by_state[SEX, education, has_partner, period]` and
  similar). Index by parametric `sex` instead.
- `model/utility/utility_components_no_care_demand.py` — same pattern.
- `model/utility/utility_functions.py` — same pattern.
- `model/wealth_and_budget/budget_equation.py:39, 55, 75, 93, 120` —
  `sex_var = SEX`, propagated into `wages`, `partner_income`,
  `unemployment_benefits`, `pension_payments`. Replace `sex_var` with the
  state variable.
- `model/wealth_and_budget/budget_equation_*.py` (all variants — drop the
  variants per Appendix A but the same pattern shows up in each).
- `model/wealth_and_budget/transfers.py` — `SEX`-indexed lookups.
- `model/stochastic_processes/health_transition.py` — `trans_mat[SEX, …]`.
- `model/stochastic_processes/job_transition.py`, `..._no_care_demand.py`,
  `..._job_retention.py` — `SEX`-indexed offer/separation tables.
- `model/stochastic_processes/partner_transition.py` — `SEX` index.
- `model/stochastic_processes/inheritance_transition.py`,
  `..._no_care_demand.py` — `SEX` index (drop entirely if no inheritance).
- `model/state_space.py`, `state_space_*.py` variants — `SEX`-conditional
  age bounds / retirement rules.
- `model/experience_baseline_model.py`, `experience_no_inheritance.py`,
  `experience_no_care_demand.py`, `experience_caregiving_leave_model.py`
  — sex-dependent experience accumulation.
- `specs/caregiving_specs.py` (line 48 area) — explicit
  `for sex_idx, sex_label in enumerate(specs["sex_labels"])`. Pattern is
  already half-built; the spec dictionary already carries `"sex_labels"`,
  which suggests both sexes can be supported with modest plumbing changes.
- `specs/health_specs.py`, `family_specs.py`, `income_specs.py`,
  `inheritance_specs.py` — re-emit per-sex matrices.
- `simulation/task_generate_initial_conditions.py` — currently draws
  initial states only for one sex; needs to draw both.

The full list of 33 hits is enumerable with
`rg --files-with-matches '\bSEX\b' src/caregiving/model`.

## C. `caregiving_type` touchpoints

Where the heterogeneity type currently lives (must be repurposed to a
leisure-preference type that enters utility instead of the choice set):

- **Definition (deterministic state):**
  `model/task_specify_model.py:85` —
  `"caregiving_type": np.arange(2, dtype=int)`.
- **Choice-set gating (the central piece to undo):**
  `model/state_space.py`, function
  `state_specific_choice_set_with_caregiving` at **line 544** (older
  commented version at line 411). Branches on `caregiving_type` at line 593
  (`if caregiving_type == 1:`); subsequent type-0 branch later in the
  function.
- **State-transition helpers in the same file:**
  `model/state_space.py:134, 183, 216, 250, 269, 287, 334, 351, 549, 561,
  593, …` — all reference `caregiving_type`.
- **Initial draw:**
  `simulation/task_generate_initial_conditions.py` — randomly assigns
  `caregiving_type ∈ {0, 1}` per agent (50/50 by default).
- **Variant state-spaces:** `state_space_no_care_demand.py` already removes
  the choice-set branching — useful as a structural template.
- **MSM moment computation:** `simulate_moments*.py` does **not** currently
  read `caregiving_type` directly; it stratifies on labor-supply choices,
  income, and wealth. Adding a leisure-preference dimension to moments would
  require an explicit aggregator change.

**Suggested rename / move (deferred to later per user):**

- Rename: `caregiving_type` → `leisure_pref_type` (or chosen final name).
- Remove the type from choice-set construction — choice set becomes pure
  labor / retirement / unemployment.
- Add the type as a parameter argument to `utility_components.py` work-
  disutility blocks and duplicate the relevant utility parameters by type.

## D. Suggested minimum-viable copy list

The fastest path to a runnable pension-model skeleton is to start with the
`*_no_care_demand` variants (they already encode "no caregiving choice")
and layer the sex-distinction work on top.

**Copy as-is (toolchain / infrastructure):**

- `pyproject.toml`, `environment.yml`, `setup.cfg`, `tox.ini`,
  `.pre-commit-config.yaml`, `.yamllint.yml`, `.gitignore`, `.gitattributes`,
  `LICENSE`, `MANIFEST.in`.
- `.gitmodules` (with the `dcegm` submodule re-added).
- `src/caregiving/config.py`, `utils.py` (rename package).
- `src/caregiving/_version.py` scaffold.
- `simulation/simulation_utils.py`, `_initial_conditions.py`,
  `plot_model_fit.py`.
- `moments/transform_data.py`, `_task_create_empirical_cov.py`,
  `task_create_cpi_germany.py`, `_task_initial_conditions.py`,
  `task_plot_empirical_moments.py`.
- `data_management/soep/variables.py`, `auxiliary.py`, `wealth.py`,
  `soep_variables/experience.py` (then trim caregiving-time deductions).
- `data_management/soep/task_create_{wage,partner_wage,health,partner_transition,survival_transition,wealth,job_separation,innovation}_sample.py`.
- `stochastic_processes/auxiliary.py` and the eight general estimation
  tasks listed under §10 KEEP.
- `figures/task_plot_{death_transition,expected_health,family_transition,income,wealth,job_offer,initial_conditions,pension_npv,pension_rates}.py`.
- `figures/publication/plotting_helpers.py`, `plotting_functions.py`,
  `task_plot_model_fit_labor_supply.py`,
  `appendix/task_plot_stochastic_processes_publication.py`.
- `tables/publication/task_stochastic_processes_tables.py`.
- `post_estimation/publication/task_create_params_table.py`,
  `task_create_moments_table.py`, `task_create_job_offer_params_table.py`.
- `post_estimation/task_plot_{age_profiles,exp_years_by_age,income_by_age,assets_and_savings_by_age,partner_state_by_age}.py`.
- `pre_estimation/task_plot_savings_grid.py`,
  `task_working_hours_table.py`.

**Selectively port and edit (model-core spine):**

- `model/utility/utility_functions_additive_no_care_demand.py`,
  `utility_components_no_care_demand.py` — start from these; add `sex` and
  `leisure_pref_type` to interfaces and split parameters by type.
- `model/wealth_and_budget/budget_equation_no_care_demand.py` (or
  `..._no_care_demand_no_inheritance.py` if dropping bequests),
  `wages_no_care_demand.py`, `transfers_no_care_demand.py`,
  `partner_income.py`, `pension_payments.py`, `tax_and_ssc.py`,
  `savings_grid.py`. Thread `sex` through.
- `model/state_space_no_care_demand.py` — start here; add `sex` to
  deterministic states; rename `caregiving_type` to leisure-preference type
  and remove all care-related branching.
- `model/shared.py` (trim heavily).
- `model/stochastic_processes/health_transition.py`,
  `partner_transition.py`, `job_transition_no_care_demand.py`.
- `model/experience_no_care_demand.py`.
- `model/pension_system/experience_stock.py`,
  `early_retirement_paths.py`.
- `model/taste_shocks.py`.
- `model/task_specify_model_no_care_demand.py` (or
  `..._no_care_demand_no_inheritance.py`).
- `specs/derive_specs.py`, `health_specs.py`, `family_specs.py`,
  `income_specs.py`, `experience_pp_specs.py`, `task_write_specs.py`.
- `estimation/estimation_setup_no_care_demand.py`, `criterion.py`,
  `prepare_estimation.py`, `standard_errors.py`,
  `start_params_and_bounds/task_set_start_params.py`, the
  `task_solve_and_simulate*.py` family.
- `simulation/simulate.py` (or the no-care-demand simulate variants),
  `simulate_moments_no_care_demand.py`,
  `task_generate_initial_conditions.py`,
  `task_simulate_moments*.py`, `task_plot_model_fit*.py`,
  `publication/task_plot_model_fit_publication.py`.
- `moments/task_create_soep_moments_no_care_demand.py`.

**Skip entirely:**

- `data_management/share/**`, `data_management/soep/task_create_event_study_sample.py`,
  `task_create_informal_care_sample.py`,
  `task_create_formal_care_costs_sample.py`,
  `task_create_inheritance_sample.py` (depending on the bequest decision),
  `task_create_rv_sample.py` (depending on whether to keep RV).
- `counterfactual/**`.
- `descriptives/**`, `temp/**`.
- All caregiving-leave / fiscal tables and figures.
- All `*_caregiving_leave*`, `*_job_retention*`, `*_beirat*`,
  `*_higher_ret_age*`, `*_lower_ret_age*`, `*_care_pension_credit*`,
  `*_higher_formal_care_costs*`, `*_lower_formal_care_costs*` model and
  policy variants.
- `model/wealth_and_budget/government_budget*.py`,
  `caregiving_leave_top_up.py`.
- `model/stochastic_processes/adl_transition.py`,
  `caregiving_transition.py`, `inheritance_transition*.py` (the last only
  if dropping bequests).
- `model/utility/bequest_utility.py` (if dropping bequests).
- `_estimate.py`, `_simulate.py`, `_task_estimate.py` (top-level scratch).

## E. Open questions deferred for later

- **Agent mortality.** The current model treats death only on the parent
  side; the agent's own mortality enters indirectly through the health
  transition. The pension model needs an explicit agent-survival process —
  `stochastic_processes/task_estimate_survival_soep_logit.py` already
  estimates this by sex and is a viable starting point.
- **German pension specifics.** `task_create_rv_sample.py` and the
  `model/pension_system/` module embed German Rentenversicherung mechanics.
  Decide whether to keep them or generalize.
- **Bequests.** `bequest_utility.py` and `inheritance_transition.py` are
  currently care-conditional; if bequests stay, decouple them from the
  caregiving choice.
- **Leisure-preference parametrization.** Which utility parameters get
  duplicated by leisure type, and how the type interacts with sex, is
  explicitly deferred.
- **Choice-set design.** Confirm the new choice set is `{full-time,
  part-time, unemployed, retired}` with retirement absorbing — i.e., the
  current set with care choices stripped.
- **Moment design.** Decide whether MSM moments are pooled across sexes
  with sex as a regressor, or stratified by sex (each moment computed twice).

---

*Generated 2026-05-08. This document reflects the repo state on the
`refine-pre-commit` branch (HEAD `9055c819`); regenerate after major refactors.*
