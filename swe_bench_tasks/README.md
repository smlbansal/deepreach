# SWE-bench Tasks for DeepReach

This directory contains SWE-bench tasks for the DeepReach repository. Each task represents a real bug or missing feature that needs to be fixed.

**Note:** Tasks 1-3 have been fixed and PRs created. Only tasks 4-5 remain as exercises.

## Available Tasks

| Task | Title | Difficulty | Type | Status |
|------|-------|------------|------|--------|
| 04 | Implement Missing sample_target_state Methods | Medium | Feature | ✅ Ready |
| 05 | Add BRS Support to Experiments | Medium-Hard | Feature | ✅ Ready |

## Completed Tasks (Fixed via PRs)

The following tasks have been fixed and are no longer available as exercises:

| Task | Title | PR | Status |
|------|-------|-----|--------|
| 01 | Fix Sine Module Initialization Bug | #2 | ✅ Fixed |
| 02 | Fix Device Mismatch in Loss Functions | #3 | ✅ Fixed |
| 03 | Fix Gradient Computation in SingleBVPNet | #4 | ✅ Fixed |

## Task Details

### Task 04: Implement Missing sample_target_state Methods
**Difficulty:** Medium  
**File:** `dynamics/dynamics.py` (multiple classes)

Several dynamics classes raise `NotImplementedError` for the `sample_target_state()` method. This method is required when `num_target_samples > 0` is used in the dataset.

**Affected classes:**
- ParameterizedVertDrone2D
- Air3D
- Dubins3D
- Dubins4D
- NarrowPassage
- Quadrotor
- MultiVehicleCollision

**Impact:** Limits dataset functionality - users cannot use target sampling with these dynamics.

### Task 05: Add BRS Support to Experiments
**Difficulty:** Medium-Hard  
**Files:** `experiments/experiments.py`, `utils/error_evaluators.py`

The codebase has TODOs indicating BRS (Backward Reachable Set) support is not fully implemented. Currently, only BRT (Backward Reachable Tube) is supported in testing and validation code.

**Impact:** Limits functionality to BRT only; BRS analysis not available.

## Running Tasks

Each task directory contains:
- `README.md` - Detailed description
- `reproduce.py` - Script to reproduce the issue
- `test_fix.py` - Test to verify the fix

### Run a specific task:

```bash
# Task 4: Missing sample_target_state
python swe_bench_tasks/task_04_missing_sample_target/reproduce.py
python -m pytest swe_bench_tasks/task_04_missing_sample_target/test_fix.py -v

# Task 5: BRS support
python swe_bench_tasks/task_05_brs_support/reproduce.py
python -m pytest swe_bench_tasks/task_05_brs_support/test_fix.py -v
```

## Task Format

Each task follows SWE-bench format:
1. **Issue description** - Clear problem statement
2. **Reproduction script** - Demonstrates the bug
3. **Test file** - Verifies the fix
4. **Expected changes** - Files to modify

## Notes

- Tasks are ordered by difficulty
- Each task is self-contained and independent
- Tests should fail before fix and pass after fix
- Tasks represent real issues in the codebase
