# SWE-bench Tasks Summary

## Available Tasks: 2

### Task 04: Missing sample_target_state
- **Status:** Ready for SWE-bench
- **Difficulty:** Medium
- **Files:** `dynamics/dynamics.py`
- **Lines:** ~100

### Task 05: BRS Support
- **Status:** Ready for SWE-bench  
- **Difficulty:** Medium-Hard
- **Files:** `experiments/experiments.py`, `utils/error_evaluators.py`
- **Lines:** ~50

## Completed Tasks: 3

These tasks have been fixed and PRs created:

1. **Task 01:** Sine __init__ bug (PR #2) - 1 line
2. **Task 02:** Device mismatch (PR #3) - 2 lines
3. **Task 03:** Gradient computation (PR #4) - 9 lines

## Quick Start

```bash
# List tasks
ls /Users/zhih/AAI/deepreach/swe_bench_tasks/

# Run Task 4
python swe_bench_tasks/task_04_missing_sample_target/reproduce.py
python -m pytest swe_bench_tasks/task_04_missing_sample_target/test_fix.py -v

# Run Task 5
python swe_bench_tasks/task_05_brs_support/reproduce.py
python -m pytest swe_bench_tasks/task_05_brs_support/test_fix.py -v
```
