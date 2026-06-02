# SWE-bench Task 02: Fix Device Mismatch in Loss Functions

## Difficulty: Easy-Medium
## Type: Bug Fix

## Issue Description

The loss functions in `utils/losses.py` create tensors without specifying a device, which causes device mismatch errors when training on GPU. The tensors are created on CPU by default, but the model and data may be on GPU.

### Location
- **File:** `utils/losses.py`
- **Lines:** 8, 34
- **Functions:** `init_brt_hjivi_loss`, `init_brat_hjivi_loss`

### Current Code

In `init_brt_hjivi_loss`:
```python
def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
    if torch.all(dirichlet_mask):
        # pretraining loss
        diff_constraint_hom = torch.Tensor([0])  # BUG: Always on CPU!
```

In `init_brat_hjivi_loss`:
```python
def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask, output):
    if torch.all(dirichlet_mask):
        # pretraining loss
        diff_constraint_hom = torch.Tensor([0])  # BUG: Always on CPU!
```

### Problem
1. `torch.Tensor([0])` creates a tensor on CPU by default
2. When training on GPU, `state`, `value`, etc. are on GPU
3. Operations between CPU and GPU tensors cause runtime errors:
   ```
   RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
   ```
4. This prevents GPU training entirely when dirichlet_mask is all True (pretraining mode)

### Impact
- **Severity:** High
- **Impact:** Prevents GPU training in pretraining mode
- **Workaround:** None (must fix to use GPU)

## Expected Behavior

Tensors created in loss functions should be on the same device as input tensors.

## Solution

Use `torch.zeros()` with device parameter or create tensor on same device as inputs:

```python
# Option 1: Use device from input tensor
diff_constraint_hom = torch.zeros(1, device=state.device)

# Option 2: Use torch.tensor with device
diff_constraint_hom = torch.tensor([0.0], device=state.device)

# Option 3: Use scalar tensor
diff_constraint_hom = torch.tensor(0.0, device=state.device)
```

## Files to Modify

- `utils/losses.py`:
  - Line 8: Fix `torch.Tensor([0])` in `brt_hjivi_loss`
  - Line 34: Fix `torch.Tensor([0])` in `brat_hjivi_loss`

## Testing

```bash
# Reproduce the issue (requires GPU or mock)
python swe_bench_tasks/task_02_device_mismatch/reproduce.py

# Run the test
python -m pytest swe_bench_tasks/task_02_device_mismatch/test_fix.py -v
```

## Verification

After the fix:
1. Loss functions should work on both CPU and GPU
2. No device mismatch errors during pretraining
3. All existing tests should still pass
