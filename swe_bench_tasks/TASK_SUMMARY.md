# SWE-bench Tasks Summary

This directory contains 5 SWE-bench tasks for the DeepReach repository.

## Quick Start

### List all tasks:
```bash
ls -la /Users/zhih/AAI/deepreach/swe_bench_tasks/
```

### Run a specific task:

```bash
# Task 1: Sine init bug
python swe_bench_tasks/task_01_sine_init_bug/reproduce.py
python -m pytest swe_bench_tasks/task_01_sine_init_bug/test_fix.py -v

# Task 2: Device mismatch
python swe_bench_tasks/task_02_device_mismatch/reproduce.py
python -m pytest swe_bench_tasks/task_02_device_mismatch/test_fix.py -v

# Task 3: Gradient detach
python swe_bench_tasks/task_03_gradient_detach/reproduce.py
python -m pytest swe_bench_tasks/task_03_gradient_detach/test_fix.py -v

# Task 4: Missing sample_target_state
python swe_bench_tasks/task_04_missing_sample_target/reproduce.py
python -m pytest swe_bench_tasks/task_04_missing_sample_target/test_fix.py -v

# Task 5: BRS support
python swe_bench_tasks/task_05_brs_support/reproduce.py
python -m pytest swe_bench_tasks/task_05_brs_support/test_fix.py -v
```

## Task Details

### Task 01: Sine Module Initialization Bug
- **File:** `utils/modules.py` (line 49)
- **Bug:** `def __init(self):` should be `def __init__(self):`
- **Difficulty:** Easy
- **Impact:** Low (works by accident, but risky)

### Task 02: Device Mismatch in Loss Functions
- **File:** `utils/losses.py` (lines 8, 34)
- **Bug:** `torch.Tensor([0])` creates CPU tensor, fails on GPU
- **Difficulty:** Easy-Medium
- **Impact:** High (prevents GPU training)

### Task 03: Gradient Computation Bug
- **File:** `utils/modules.py` (line 134)
- **Bug:** `.detach()` breaks computation graph
- **Difficulty:** Medium
- **Impact:** Medium (prevents end-to-end training)

### Task 04: Missing sample_target_state
- **File:** `dynamics/dynamics.py` (multiple classes)
- **Bug:** 7 classes raise NotImplementedError
- **Difficulty:** Medium
- **Impact:** Medium (limits dataset options)

### Task 05: BRS Support
- **File:** `experiments/experiments.py`, `utils/error_evaluators.py`
- **Bug:** TODOs indicate BRS not implemented
- **Difficulty:** Medium-Hard
- **Impact:** Medium (limits functionality)

## Verification

All reproduce scripts have been tested and confirm the bugs exist:

✅ Task 1: Bug reproduced - Sine __init__ is misspelled
✅ Task 2: Bug reproduced - torch.Tensor([0]) creates CPU tensor
✅ Task 3: Bug reproduced - Gradients don't flow to input
✅ Task 4: Bug reproduced - NotImplementedError raised
✅ Task 5: Bug reproduced - BRS TODOs found

## Usage in SWE-bench

These tasks can be used to evaluate AI coding assistants:

1. **Setup:** Provide the repository and task description
2. **Task:** Ask the assistant to fix the bug
3. **Verify:** Run reproduce script (should show bug)
4. **Verify:** Run test_fix.py (should pass after fix)
5. **Verify:** Run existing tests to ensure no regression

## Expected Fixes

### Task 1 Fix
```python
# utils/modules.py line 49
# Before:
def __init(self):
# After:
def __init__(self):
```

### Task 2 Fix
```python
# utils/losses.py lines 8, 34
# Before:
diff_constraint_hom = torch.Tensor([0])
# After:
diff_constraint_hom = torch.zeros(1, device=state.device)
```

### Task 3 Fix
```python
# utils/modules.py line 134
# Before:
coords_org = model_input['coords'].clone().detach().requires_grad_(True)
# After:
coords_org = model_input['coords']
if not coords_org.requires_grad:
    coords_org = coords_org.requires_grad_(True)
# Plus updates to calling code to use retain_grad()
```

### Task 4 Fix
Implement `sample_target_state()` for each dynamics class to sample states from target set.

### Task 5 Fix
Implement BRS support in experiments and error_evaluators, removing TODOs.

## Notes

- Tasks 1-3 are good starter tasks (clear bugs, easy fixes)
- Task 4 is good for testing understanding of the domain
- Task 5 is more complex, good for advanced evaluation
- All tasks have been verified to reproduce the issues
