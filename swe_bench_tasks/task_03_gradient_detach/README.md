# SWE-bench Task 03: Fix Gradient Computation in SingleBVPNet

## Difficulty: Medium
## Type: Bug Fix / Improvement

## Issue Description

The `SingleBVPNet.forward()` method in `utils/modules.py` uses `.clone().detach().requires_grad_(True)` on input coordinates, which breaks the computation graph. This prevents gradients from flowing back to the original input, making end-to-end differentiability impossible.

### Location
- **File:** `utils/modules.py`
- **Line:** 134
- **Class:** `SingleBVPNet`
- **Method:** `forward`

### Current Code
```python
def forward(self, model_input, params=None):
    if params is None:
        params = OrderedDict(self.named_parameters())

    # Enables us to compute gradients w.r.t. coordinates
    # TODO: should not need to .clone().detach().requires_grad_(True); 
    # instead, use .retain_grad() on input in calling script
    # otherwise, .detach() removes input from the graph so grad 
    # cannot propagate back end-to-end, e.g., percept -> NN -> state 
    # estimation (input)
    coords_org = model_input['coords'].clone().detach().requires_grad_(True)
    coords = coords_org

    output = self.net(coords)
    return {'model_in': coords_org, 'model_out': output}
```

### Problem
1. `.detach()` removes the tensor from the computation graph
2. Gradients cannot flow back to `model_input['coords']`
3. Prevents end-to-end training with perception modules
4. The TODO comment acknowledges this is a problem

### Impact
- **Severity:** Medium
- **Impact:** Prevents end-to-end differentiability
- **Use case:** Cannot train perception + DeepReach jointly
- **Workaround:** None clean; requires refactoring

## Expected Behavior

Gradients should flow from loss back through the network to the input coordinates, enabling end-to-end training.

## Solution

Remove `.clone().detach()` and use `.retain_grad()` on inputs in the calling code, OR modify the approach to preserve the computation graph.

**Option 1:** Modify `SingleBVPNet.forward()` to not detach:
```python
coords_org = model_input['coords']
if not coords_org.requires_grad:
    coords_org = coords_org.requires_grad_(True)
coords = coords_org
```

**Option 2:** Update calling code to retain gradients:
```python
# In experiments.py and dataio.py
coords = coords.requires_grad_(True)
coords.retain_grad()
model_input = {'coords': coords}
```

## Files to Modify

- `utils/modules.py` (line 134): Remove `.clone().detach()`
- `experiments/experiments.py`: Update to retain gradients on inputs
- `utils/dataio.py`: Update to retain gradients if needed

## Testing

```bash
# Reproduce the issue
python swe_bench_tasks/task_03_gradient_detach/reproduce.py

# Run the test
python -m pytest swe_bench_tasks/task_03_gradient_detach/test_fix.py -v
```

## Verification

After the fix:
1. Gradients should flow from loss to input coordinates
2. `model_input['coords'].grad` should be populated after backward pass
3. All existing tests should still pass
4. End-to-end differentiability should work
