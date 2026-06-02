# SWE-bench Task 04: Implement Missing sample_target_state Methods

## Difficulty: Medium
## Type: Feature Implementation

## Issue Description

Several dynamics classes in `dynamics/dynamics.py` raise `NotImplementedError` for the `sample_target_state()` method. This method is required when `num_target_samples > 0` is used in the dataset, but many dynamics classes don't implement it.

### Affected Classes
The following dynamics classes raise `NotImplementedError`:
- `ParameterizedVertDrone2D`
- `Air3D`
- `Dubins3D`
- `Dubins4D`
- `NarrowPassage`
- `Quadrotor`
- `MultiVehicleCollision`

### Location
- **File:** `dynamics/dynamics.py`
- **Methods:** `sample_target_state()` in multiple classes

### Current Code
```python
def sample_target_state(self, num_samples):
    raise NotImplementedError
```

### Problem
1. Users cannot use `num_target_samples > 0` with these dynamics
2. Dataset creation fails with `NotImplementedError` when target sampling is requested
3. Limits training options and flexibility

### Impact
- **Severity:** Medium
- **Impact:** Feature incomplete, limits dataset options
- **Workaround:** Set `num_target_samples=0` (but loses functionality)

## Expected Behavior

Each dynamics class should implement `sample_target_state()` to sample states from within the target set.

## Solution

Implement `sample_target_state()` for each affected dynamics class. The method should:
1. Sample `num_samples` states from within the target set
2. Return tensor of shape `(num_samples, state_dim)`
3. States should satisfy the boundary condition (be inside target)

### Implementation Guidelines

**For Dubins3D:**
- Target is a circle with radius `goalR` centered at origin
- Sample x, y uniformly within circle, theta uniformly in [-pi, pi]

**For Air3D:**
- Target is collision region (distance < collision radius)
- Sample states where aircraft are close

**For Quadrotor:**
- Target is landing region
- Sample positions near landing pad with low velocities

**General approach:**
```python
def sample_target_state(self, num_samples):
    # Sample states from target set
    # Return tensor of shape (num_samples, state_dim)
    pass
```

## Files to Modify

- `dynamics/dynamics.py`:
  - `ParameterizedVertDrone2D.sample_target_state()` (line 173)
  - `Air3D.sample_target_state()` (line 242)
  - `Dubins3D.sample_target_state()` (line 316)
  - `Dubins4D.sample_target_state()` (line 407)
  - `NarrowPassage.sample_target_state()` (line 562)
  - `Quadrotor.sample_target_state()` (line 971)
  - `MultiVehicleCollision.sample_target_state()` (line 1138)

## Testing

```bash
# Reproduce the issue
python swe_bench_tasks/task_04_missing_sample_target/reproduce.py

# Run the test
python -m pytest swe_bench_tasks/task_04_missing_sample_target/test_fix.py -v
```

## Verification

After the fix:
1. `sample_target_state()` should return valid states for all dynamics
2. States should be within target set (boundary_fn(state) <= 0)
3. Dataset creation with `num_target_samples > 0` should work
4. All existing tests should still pass
