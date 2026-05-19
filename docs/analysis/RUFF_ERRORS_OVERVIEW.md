# Ruff Linter Errors Overview

**Generated**: From ruff check output
**Total Errors**: 1408
**Status**: Ready for systematic fixing

---

## Executive Summary

This document provides a comprehensive overview of all Ruff linter errors in the codebase. Errors are organized by priority and type to facilitate systematic fixing.

### Priority Levels

1. **Critical (Must Fix)**: F821 (Undefined name), invalid-syntax - These cause runtime failures
2. **High Priority**: F811 (Redefinition), F601 (Dictionary key shadowing) - May indicate bugs
3. **Medium Priority**: F841 (Unused variables), B007 (Unused loop vars) - Code cleanliness
4. **Low Priority**: E501 (Line too long), W291/W293 (Whitespace) - Style issues
5. **Refactoring**: PLR0912/PLR0915 (Complexity) - Requires careful refactoring

---

## Error Breakdown by Code

- **B007**: 206 errors
- **B020**: 4 errors
- **E501**: 719 errors
- **E741**: 6 errors
- **F601**: 1 errors
- **F811**: 14 errors
- **F821**: 206 errors
- **F841**: 89 errors
- **PIE810**: 1 errors
- **PLR0133**: 2 errors
- **PLR0912**: 17 errors
- **PLR0915**: 6 errors
- **PLR2004**: 105 errors
- **PLR2044**: 4 errors
- **W291**: 2 errors
- **W293**: 14 errors
- **invalid-syntax**: 12 errors

---

## Detailed Errors by Category

### F821 - Undefined Name (206 errors) - **CRITICAL**

**Severity**: Critical - These will cause runtime failures
**Action Required**: Add missing imports or define missing variables/functions

**Files affected**: [To be populated from detailed output]

**Common causes**:
- Missing imports
- Typographical errors in variable names
- Variables used before definition
- Missing function definitions

---

### E501 - Line Too Long (719 errors) - **STYLE**

**Severity**: Style - Code formatting
**Action Required**: Break long lines to stay within 88 character limit

**Files affected**: [Multiple files across codebase]

**Note**: Many files already have `E501` in `per-file-ignores` in `pyproject.toml`. Consider:
- Adding more files to per-file-ignores if appropriate
- Using black formatter to auto-fix where possible
- Manually breaking long lines for better readability

---

### B007 - Unused Loop Variable (206 errors) - **CODE CLEANLINESS**

**Severity**: Low - Code clarity
**Action Required**: Rename unused loop variables to `_variable_name`

**Files affected**: [Multiple files]

**Example fix**:
```python
# Before
for i in range(10):
    do_something()

# After
for _i in range(10):
    do_something()
```

---

### F841 - Unused Local Variable (89 errors) - **CODE CLEANLINESS**

**Severity**: Low - Code clarity
**Action Required**: Remove unused variables or use them if intended

**Files affected**: [Multiple files]

**Note**: Some variables may be intentionally unused for documentation. Rename to `_variable_name` in such cases.

---

### PLR2004 - Magic Value Used in Comparison (105 errors) - **CODE QUALITY**

**Severity**: Medium - Code maintainability
**Action Required**: Replace magic numbers with named constants

**Example fix**:
```python
# Before
if age == 65:
    ...

# After
RETIREMENT_AGE = 65
if age == RETIREMENT_AGE:
    ...
```

---

### F811 - Redefinition of Unused Variable (14 errors) - **HIGH PRIORITY**

**Severity**: High - May indicate bugs
**Action Required**: Remove first definition or use variable before redefining

---

### PLR0912 - Too Many Branches (17 errors) - **REFACTORING**

**Severity**: Medium - Code complexity
**Action Required**: Refactor functions to reduce conditional complexity

**Approach**:
- Extract complex conditionals into separate functions
- Use dictionary dispatch for multiple if/elif chains
- Consider strategy pattern for complex branching

---

### PLR0915 - Too Many Statements (6 errors) - **REFACTORING**

**Severity**: Medium - Code complexity
**Action Required**: Break large functions into smaller, focused functions

**Approach**:
- Extract logical blocks into separate functions
- Follow single responsibility principle
- Target: Keep functions under ~50 statements

---

### Other Errors

- **W291/W293** (16 errors): Trailing/whitespace in blank lines - Auto-fix with black
- **E741** (6 errors): Ambiguous variable names (e.g., `l`, `O`) - Rename to be clearer
- **B020** (4 errors): Loop variable overwrites function argument - Rename loop variable
- **PLR2044** (4 errors): Empty comments - Remove or add content
- **invalid-syntax** (12 errors): Syntax errors - Must fix before code runs
- **PLR0133** (2 errors): Constant imported from non-constant - Review imports
- **F601** (1 error): Dictionary key name shadowed - Review variable names
- **PIE810** (1 error): Multiple starts-with ends-with - Simplify conditionals

---

## Fixing Strategy

### Phase 1: Critical Fixes (Priority 1)
1. Fix all `invalid-syntax` errors (12 errors)
2. Fix all `F821` undefined name errors (206 errors)
3. Verify code runs without errors

### Phase 2: High Priority Fixes (Priority 2)
1. Fix `F811` redefinition errors (14 errors)
2. Fix `F601` dictionary key shadowing (1 error)

### Phase 3: Code Cleanliness (Priority 3)
1. Fix `F841` unused variables (89 errors)
2. Fix `B007` unused loop variables (206 errors)
3. Fix `W291/W293` whitespace issues (16 errors) - Auto-fix with black

### Phase 4: Code Quality (Priority 4)
1. Replace `PLR2004` magic values with constants (105 errors)
2. Fix `E741` ambiguous variable names (6 errors)
3. Fix `B020` loop variable overwrites (4 errors)

### Phase 5: Style (Priority 5)
1. Fix `E501` line too long errors (719 errors) - Prioritize frequently used files
   - Use black formatter where appropriate
   - Add to per-file-ignores if justified
   - Manually fix critical files

### Phase 6: Refactoring (Priority 6)
1. Refactor functions with `PLR0912` too many branches (17 errors)
2. Refactor functions with `PLR0915` too many statements (6 errors)

---

## Quick Fix Commands

### Auto-fix what can be auto-fixed:
```bash
# Run ruff with --fix to auto-fix safe issues
ruff check src/ --fix

# Format code with black (fixes E501, W291, W293)
black src/
```

### Fix specific error types:
```bash
# Fix only unused variables
ruff check src/ --select F841 --fix

# Fix only unused loop variables
ruff check src/ --select B007 --fix

# Fix whitespace
ruff check src/ --select W291,W293 --fix
```

---

## Notes

- Many files already have `E501` (line too long) in `per-file-ignores` in `pyproject.toml`
- Consider adding more files to per-file-ignores if they have legitimate reasons for long lines
- Use `# noqa: CODE` comments for errors that are intentionally not fixed (document why)
- Some complexity warnings (PLR0912, PLR0915) may be acceptable for certain functions - document rationale

---

## Next Steps

1. **Run detailed error extraction**: `ruff check src/ --output-format=json > ruff_errors.json`
2. **Parse JSON and update this document** with specific file locations and line numbers
3. **Start fixing errors** in priority order (Phase 1 → Phase 6)
4. **Track progress** by updating error counts as fixes are applied
5. **Run tests** after each phase to ensure nothing breaks

---

## Current Pre-Commit Issues (from `pre-commit run --all-files`)

### Debug Statements / AST Parsing

- **Hook**: `debug-statements`
- **Status**: Fails because Python cannot parse two files (syntax errors):
  - `src/caregiving/post_estimation/task_plot_caregiving_leave_top_up.py`
    - Unterminated string literal at `ax.set_title(...)` around line 343.
  - `src/caregiving/post_estimation/task_post_inheritance.py`
    - Unterminated f-string at the `print` of lagged inheritance share around line 380.

**Planned action**:
- Rewrite problematic `ax.set_title(...)` and `print(...)` blocks as valid multi-line strings and f-strings.
- Re-run `python -m compileall src` and the pre-commit `debug-statements` hook.

### Black (Code Formatting)

- **Hook**: `black`
- **Status**: Fails because it cannot parse the same two files:
  - `task_plot_caregiving_leave_top_up.py`: Unterminated string at line 343.
  - `task_post_inheritance.py`: Unterminated string at line 380.

**Planned action**:
- Once syntax is fixed as above, re-run `black` on the affected files and then on the repo.

### Ruff Lint (Complexity, Style, Undefined Names)

- **Hook**: `ruff`
- **Main issues currently reported**:
  - **Complexity / size**:
    - `PLR0912` (too many branches) and `PLR0915` (too many statements) in a set of large plotting tasks, especially:
      - `src/caregiving/counterfactual/debugging/task_compare_assets_baseline_vs_no_care_demand.py`
      - `src/caregiving/counterfactual/matched_differences/task_plot_age_profiles_*`
      - `src/caregiving/counterfactual/matched_differences_end_of_caregiving/task_plot_employment_rate_by_distance_to_first_care.py`
      - `src/caregiving/counterfactual/task_plot_labor_supply_differences_*` (various caregiving-leave / full-cg / higher-ret-age variants)
      - `src/caregiving/post_estimation/task_plot_age_profiles.py`
      - `src/caregiving/post_estimation/task_plot_assets_and_savings_by_age.py` (helper `_plot_asset_savings_outcome`)
  - **Magic values** (`PLR2004`):
    - Repeated comparisons against literal values like `2`, `3`, `4`, `5`, `1e-10` in:
      - `task_plot_employment_rate_by_distance_to_first_care.py`
      - `task_compute_career_costs.py`
      - `model/pension_system/early_retirement_paths.py`
  - **Unused loop variables** (`B007`):
    - Many `for ... in education_specs` / `for ... in caregiving_type_specs` loops in:
      - `task_plot_labor_supply_differences_caregiving_leave_with_job_retention.py`
      - `task_plot_labor_supply_differences_full_caregiving_leave_with_job_retention.py`
  - **Line too long** (`E501`):
    - A few long path definitions in:
      - `task_plot_age_profiles_full_caregiving_leave_vs_caregiving_leave.py`
      - `task_plot_labor_supply_differences_full_caregiving_leave_with_job_retention.py`
      - Two test files:
        - `tests/test_estimate_model_interface.py`
        - `tests/test_estimate_model_with_unobserved_type_shares_interface.py`
  - **Undefined names / redefinitions**:
    - `F821` undefined names:
      - `path_to_full_cg_leave_data` in `task_plot_labor_supply_differences_full_caregiving_leave_with_job_retention_age_functions.py`
      - `edu_name` / `cg_name` in `task_plot_labor_supply_differences_no_care_demand.py`
    - `F811` redefinition:
      - `task_plot_matched_differences_by_age_cg_leave_vs_baseline` redefined in the same
        `full_caregiving_leave_with_job_retention_age_functions` module.

**Planned action**:
- For **complexity** (`PLR0912/PLR0915`):
  - Prefer small, targeted refactors (extract helper functions) where straightforward.
  - Where functions are intentionally large but stable plotting “scripts”, add documented `# noqa: PLR0912,PLR0915` on the `def` line.
- For **magic values** (`PLR2004`):
  - Introduce named constants (e.g., `CARE_AT_2_PERIOD = 2`, `EPSILON_NPV = 1e-10`) near the top of each module and replace literals.
- For **B007**:
  - Rename unused loop variables to `_edu_label`, `_cg_type_label`, `_outcome_key`, etc.
- For **E501**:
  - Break the longest paths / strings into multiple concatenated pieces or use implicit
    parentheses; in tests, consider `# noqa: E501` if the message is intentionally long.
- For **F821/F811**:
  - Fix missing imports/arguments (e.g. pass `path_to_full_cg_leave_data` where needed).
  - Remove or rename duplicated function definitions in the age-functions module.

### Refurb

- **Hook**: `refurb`
- **Status**: Currently fails on the same syntax error as Ruff/Black:
  - `task_plot_caregiving_leave_top_up.py:343` (unterminated string literal).

**Planned action**:
- Once syntax is fixed, re-run `refurb` and triage any remaining suggestions separately (they are not currently blocking syntax).

### Interrogate (Docstring Coverage)

- **Hook**: `interrogate`
- **Status**: Currently failing because it cannot parse:
  - `task_plot_assets_and_savings_by_age.py` at line ~1515 (unterminated `Index names:` f-string).

**Planned action**:
- Fix remaining `Index names` f-strings (same pattern as other post-estimation modules).
- Only after syntax is clean, revisit interrogate’s actual coverage report.

### Codespell (Spelling)

- **Hook**: `codespell`
- **Status**: Failing due to a set of misspellings:
  - `src/caregiving/model/pension_system/early_retirement_paths.py`
    - `threshhold` → **threshold**
  - `.../matched_differences_end_of_caregiving/task_plot_employment_rate_by_distance_to_first_care.py`
    - `patter` → **pattern**
    - `roviding` → **providing** (two occurrences)
    - `includ` → **include**
    - `stopp` → **stop**
    - `ncluded` → **included**
  - `src/caregiving/post_estimation/task_post_inheritance.py`
    - `nd` → **and** (around line 718)
    - `hav` → **have** (around lines 902, 1023)
  - `src/caregiving/post_estimation/task_post_inheritance_caregiving_leave.py`
    - `nd` → **and** (around line 727)
    - `hav` → **have** (around lines 916, 1035)
  - `src/caregiving/specs/experience_pp_specs.py`
    - `accross` → **across**
  - `src/caregiving/counterfactual/plotting_utils.py`
    - `bu` → likely **by** (or another intended word; to verify in context)
  - `src/caregiving/model/experience_baseline_model.py`
    - `individals` → **individuals**
  - `src/caregiving/counterfactual/task_plot_labor_supply_differences_no_care_demand.py`
    - `savin` → **saving** (or **savings**, to verify in context)
  - `src/caregiving/data_management/soep/soep_variables/experience.py`
    - `als` → **also**

**Planned action**:
- Fix each misspelling in place, verifying intended meaning from surrounding context.
- Re-run `codespell` to ensure the hook passes.

---

**Last Updated**: Based on latest `ruff` and `pre-commit` output
**Total Errors (original Ruff snapshot)**: 1408
