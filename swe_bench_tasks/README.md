# SWE-bench Tasks for DeepReach

This directory contains 5 SWE-bench tasks for the DeepReach repository. Each task represents a real bug or missing feature that needs to be fixed.

## Task Overview

| Task | Title | Difficulty | Type | Status |
|------|-------|------------|------|--------|
| 01 | Fix Sine Module Initialization Bug | Easy | Bug Fix | ✅ Ready |
| 02 | Fix Device Mismatch in Loss Functions | Easy-Medium | Bug Fix | ✅ Ready |
| 03 | Fix Gradient Computation in SingleBVPNet | Medium | Bug Fix | ✅ Ready |
| 04 | Implement Missing sample_target_state Methods | Medium | Feature | ✅ Ready |
| 05 | Add BRS Support to Experiments | Medium-Hard | Feature | 📝 Scaffolded |

## Task Details

### Task 01: Fix Sine Module Initialization Bug
**Difficulty:** Easy  
**File:** `utils/modules.py` (line 49)

The `Sine` class has a typo: `def __init(self):` should be `def __init__(self):`. The method is never called due to the misspelling.

**Impact:** Low severity but high risk for future changes.

### Task 02: Fix Device Mismatch in Loss Functions
**Difficulty:** Easy-Medium  
**File:** `utils/losses.py` (lines 8, 34)

Loss functions use `torch.Tensor([0])` which always creates CPU tensors, causing device mismatch errors when training on GPU.

**Impact:** Prevents GPU training in pretraining mode.

### Task 03: Fix Gradient Computation in SingleBVPNet
**Difficulty:** Medium  
**File:** `utils/modules.py` (line 134)

The `forward()` method uses `.clone().detach().requires_grad_(True)` which breaks the computation graph, preventing gradients from flowing back to inputs.

**Impact:** Prevents end-to-end differentiability with perception modules.

### Task 04: Implement Missing sample_target_state Methods
**Difficulty:** Medium  
**File:** `dynamics/dynamics.py` (multiple classes)

Several dynamics classes raise `NotImplementedError` for `sample_target_state()`, preventing use of `num_target_samples > 0`.

**Affected classes:**
- ParameterizedVertDrone2D
- Air3D
- Dubins3D
- Dubins4D
- NarrowPassage
- Quadrotor
- MultiVehicleCollision

**Impact:** Limits dataset functionality.

### Task 05: Add BRS Support to Experiments
**Difficulty:** Medium-Hard  
**File:** `experiments/experiments.py`, `utils/error_evaluators.py`

The code has TODOs for BRS (Backward Reachable Set) support but only implements BRT (Backward Reachable Tube).

**Impact:** Limits functionality to BRT only.

## Running Tasks

Each task directory contains:
- `README.md` - Task description
- `reproduce.py` - Script to reproduce the issue
- `test_fix.py` - Test to verify the fix

### Run a specific task:

```bash
# Reproduce the issue
python swe_bench_tasks/task_01_sine_init_bug/reproduce.py

# Run the test (should fail before fix, pass after fix)
python -m pytest swe_bench_tasks/task_01_sine_init_bug/test_fix.py -v
```

### Run all task tests:

```bash
for task in swe_bench_tasks/task_*/; do
    echo "Testing $task..."
    python -m pytest "$task" -v
done
```

## Task Format

Each task follows SWE-bench format:
1. **Issue description** - Clear problem statement
2. **Reproduction script** - Demonstrates the bug
3. **Test file** - Verifies the fix
4. **Expected changes** - Files to modify

## Creating New Tasks

To create a new SWE-bench task:

1. Create task directory: `swe_bench_tasks/task_XX_description/`
2. Add `README.md` with issue description
3. Add `reproduce.py` to demonstrate the bug
4. Add `test_fix.py` to verify the fix
5. Update this README

## Notes

- Tasks are ordered by difficulty (easy to hard)
- Each task is self-contained and independent
- Tests should fail before fix and pass after fix
- Tasks represent real issues in the codebase
