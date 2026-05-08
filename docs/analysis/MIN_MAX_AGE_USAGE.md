# Usage of MIN_AGE and MAX_AGE constants in shared.py

## Summary

**`MIN_AGE` and `MAX_AGE` in `shared.py` are ONLY used for calculating `AGE_BINS`.**

## Where MIN_AGE and MAX_AGE are used:

### In `shared.py` itself:
- **Lines 62-68**: Used to calculate `AGE_BINS`:
  ```python
  AGE_BINS = [
      (AGE_40 - MIN_AGE, AGE_45 - MIN_AGE),
      (AGE_45 - MIN_AGE, AGE_50 - MIN_AGE),
      ...
  ]
  ```

### Where `AGE_BINS` from `shared.py` is used:
- **NOT USED** in active codebase
- Only found in:
  - `_simulate.py` (but this file has its own local `MIN_AGE` and `MAX_AGE` definitions)
  - Commented-out code in `simulation/_simulate.py`
  - Sandbox notebooks (which define their own local versions)

## Other files with their own MIN_AGE/MAX_AGE:
- `data_management/share/task_create_estimation_data_set.py`: `MIN_AGE = 30`, `MAX_AGE = 100`
- `moments/_task_initial_conditions.py`: `MIN_AGE = 50`, `MAX_AGE = 65`
- `_simulate.py`: `MIN_AGE = 40`, `MAX_AGE = 75`
- Various sandbox notebooks: local definitions

## Conclusion:

**`MIN_AGE` and `MAX_AGE` in `shared.py` appear to be legacy/unused constants** that are only used to calculate `AGE_BINS`, which itself is not used in the active codebase.

**Recommendation**: These can likely be removed or kept for backward compatibility if `AGE_BINS` might be used in the future.
