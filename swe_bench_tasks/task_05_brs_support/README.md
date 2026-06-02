# SWE-bench Task 05: Add BRS Support to Experiments

## Difficulty: Medium-Hard
## Type: Feature Implementation

## Issue Description

The codebase has TODOs indicating that BRS (Backward Reachable Set) support is not fully implemented. Currently, only BRT (Backward Reachable Tube) is supported in the testing and validation code.

### Location
- **File:** `experiments/experiments.py`
- **Lines:** 275, 289, 302
- **File:** `utils/error_evaluators.py`

### Current Code
```python
# experiments.py
set_type="BRT", control_type="value", # TODO: implement option for BRS too
```

### Problem
1. Test methods hardcode `set_type="BRT"`
2. BRS (Backward Reachable Set) computation not available
3. Limits the types of reachability analysis that can be performed

### Difference Between BRT and BRS
- **BRT (Backward Reachable Tube):** States from which the system can reach the target within the time horizon
- **BRS (Backward Reachable Set):** States from which the system can reach the target at exactly the final time

### Impact
- **Severity:** Medium
- **Impact:** Limits functionality to BRT only
- **Use case:** Users who need BRS analysis cannot use the codebase

## Expected Behavior

The experiments should support both BRT and BRS set types, selectable via configuration.

## Solution

1. Add `set_type` parameter to test methods
2. Implement BRS logic in scenario optimization
3. Update validation and plotting for BRS
4. Add tests for BRS functionality

## Files to Modify

- `experiments/experiments.py`: Add BRS support to test methods
- `utils/error_evaluators.py`: Update scenario_optimization for BRS
- `run_experiment.py`: Add CLI argument for set_type in test mode

## Testing

```bash
# Reproduce the issue
python swe_bench_tasks/task_05_brs_support/reproduce.py

# Run the test
python -m pytest swe_bench_tasks/task_05_brs_support/test_fix.py -v
```

## Verification

After the fix:
1. BRS option should be available in test mode
2. Scenario optimization should work with BRS
3. All existing BRT tests should still pass
4. New BRS tests should pass
