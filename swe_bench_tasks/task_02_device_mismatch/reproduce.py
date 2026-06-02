"""
Reproduction script for device mismatch bug in loss functions.

This script demonstrates that loss functions create CPU tensors
even when inputs are on GPU, causing device mismatch errors.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

import torch
from utils import losses
from dynamics import dynamics

print("=" * 70)
print("Reproducing Device Mismatch Bug in Loss Functions")
print("=" * 70)
print()

# Create a simple dynamics instance
dubins = dynamics.Dubins3D(
    goalR=0.25,
    velocity=0.6,
    omega_max=1.1,
    angle_alpha_factor=1.2,
    set_mode="avoid",
    freeze_model=False,
)
dubins.deepreach_model = "exact"

# Create loss function
loss_fn = losses.init_brt_hjivi_loss(
    dubins, minWith="target", dirichlet_loss_divisor=1.0
)

print("Test 1: Testing loss function with CPU tensors...")
print("-" * 70)

# Test with CPU tensors (should work even with bug)
batch_size = 5
state_cpu = torch.randn(batch_size, 3)
value_cpu = torch.randn(batch_size)
dvdt_cpu = torch.randn(batch_size)
dvds_cpu = torch.randn(batch_size, 3)
boundary_value_cpu = torch.randn(batch_size)
dirichlet_mask_cpu = torch.ones(
    batch_size, dtype=torch.bool
)  # All True = pretraining mode
output_cpu = torch.randn(batch_size, 1)

try:
    result = loss_fn(
        state_cpu,
        value_cpu,
        dvdt_cpu,
        dvds_cpu,
        boundary_value_cpu,
        dirichlet_mask_cpu,
        output_cpu,
    )
    print("✓ CPU test passed")
    print(f"  Result keys: {result.keys()}")
    print(f"  diff_constraint_hom device: {result['diff_constraint_hom'].device}")
except Exception as e:
    print(f"✗ CPU test failed: {e}")

print()

# Test 2: Check if GPU is available and test device mismatch
print("Test 2: Checking for GPU and testing device mismatch...")
print("-" * 70)

if torch.cuda.is_available():
    print("GPU is available. Testing with CUDA tensors...")

    # Move tensors to GPU
    state_gpu = state_cpu.cuda()
    value_gpu = value_cpu.cuda()
    dvdt_gpu = dvdt_cpu.cuda()
    dvds_gpu = dvds_cpu.cuda()
    boundary_value_gpu = boundary_value_cpu.cuda()
    dirichlet_mask_gpu = dirichlet_mask_cpu.cuda()
    output_gpu = output_cpu.cuda()

    try:
        result = loss_fn(
            state_gpu,
            value_gpu,
            dvdt_gpu,
            dvds_gpu,
            boundary_value_gpu,
            dirichlet_mask_gpu,
            output_gpu,
        )
        print("✓ GPU test passed")
        print(f"  Result keys: {result.keys()}")
        print(f"  diff_constraint_hom device: {result['diff_constraint_hom'].device}")
        bug_exists = False
    except RuntimeError as e:
        if "device" in str(e).lower() or "cuda" in str(e).lower():
            print(f"✗ GPU test failed with device mismatch error:")
            print(f"  {e}")
            print()
            print("  BUG CONFIRMED: Loss function creates CPU tensor even when")
            print("  inputs are on GPU, causing device mismatch!")
            bug_exists = True
        else:
            print(f"✗ GPU test failed with different error: {e}")
            bug_exists = False
else:
    print("GPU not available. Testing device mismatch by simulation...")
    print()
    print("Simulating the bug by checking tensor device...")

    # Even on CPU, we can check if the created tensor is on the right device
    # The bug is that torch.Tensor([0]) always creates CPU tensor
    result = loss_fn(
        state_cpu,
        value_cpu,
        dvdt_cpu,
        dvds_cpu,
        boundary_value_cpu,
        dirichlet_mask_cpu,
        output_cpu,
    )

    result_device = result["diff_constraint_hom"].device
    input_device = state_cpu.device

    print(f"  Input device: {input_device}")
    print(f"  Result device: {result_device}")

    if result_device != input_device:
        print()
        print("  BUG CONFIRMED: Result tensor is on different device than inputs!")
        print(f"  Expected: {input_device}, Got: {result_device}")
        bug_exists = True
    else:
        # On CPU, both will be CPU even with bug, so we check the code
        print()
        print("  Checking source code for the bug...")
        import inspect

        source = inspect.getsource(loss_fn)
        if "torch.Tensor([0])" in source:
            print(
                "  BUG CONFIRMED: Code uses torch.Tensor([0]) which always creates CPU tensor!"
            )
            print("  This will fail when inputs are on GPU.")
            bug_exists = True
        else:
            print("  Bug not found in source code.")
            bug_exists = False

print()
print("=" * 70)
if bug_exists:
    print("RESULT: Bug reproduced successfully!")
    print("Loss functions create CPU tensors, causing device mismatch on GPU.")
else:
    print("RESULT: Bug not reproduced or already fixed.")
print("=" * 70)
