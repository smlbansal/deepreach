"""
Test script to verify the device mismatch bug is fixed.

This test should FAIL before the fix and PASS after the fix.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

import torch
from utils import losses
from dynamics import dynamics
import inspect


def test_brt_loss_device():
    """Test that BRT loss creates tensors on correct device."""
    dubins = dynamics.Dubins3D(
        goalR=0.25,
        velocity=0.6,
        omega_max=1.1,
        angle_alpha_factor=1.2,
        set_mode="avoid",
        freeze_model=False,
    )
    dubins.deepreach_model = "exact"

    loss_fn = losses.init_brt_hjivi_loss(
        dubins, minWith="target", dirichlet_loss_divisor=1.0
    )

    # Test with CPU
    batch_size = 5
    state = torch.randn(batch_size, 3)
    value = torch.randn(batch_size)
    dvdt = torch.randn(batch_size)
    dvds = torch.randn(batch_size, 3)
    boundary_value = torch.randn(batch_size)
    dirichlet_mask = torch.ones(batch_size, dtype=torch.bool)
    output = torch.randn(batch_size, 1)

    result = loss_fn(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output)

    # Check that result tensors are on same device as inputs
    assert result["diff_constraint_hom"].device == state.device, (
        f"Result device {result['diff_constraint_hom'].device} != input device {state.device}"
    )

    print("✓ BRT loss creates tensors on correct device (CPU)")


def test_brat_loss_device():
    """Test that BRAT loss creates tensors on correct device."""
    narrow = dynamics.NarrowPassage(avoid_fn_weight=1.0, avoid_only=False)
    narrow.deepreach_model = "exact"

    loss_fn = losses.init_brat_hjivi_loss(
        narrow, minWith="target", dirichlet_loss_divisor=1.0
    )

    # Test with CPU
    batch_size = 5
    state = torch.randn(batch_size, 10)
    value = torch.randn(batch_size)
    dvdt = torch.randn(batch_size)
    dvds = torch.randn(batch_size, 10)
    boundary_value = torch.randn(batch_size)
    reach_value = torch.randn(batch_size)
    avoid_value = torch.randn(batch_size)
    dirichlet_mask = torch.ones(batch_size, dtype=torch.bool)
    output = torch.randn(batch_size, 1)

    result = loss_fn(
        state,
        value,
        dvdt,
        dvds,
        boundary_value,
        reach_value,
        avoid_value,
        dirichlet_mask,
        output,
    )

    # Check that result tensors are on same device as inputs
    assert result["diff_constraint_hom"].device == state.device, (
        f"Result device {result['diff_constraint_hom'].device} != input device {state.device}"
    )

    print("✓ BRAT loss creates tensors on correct device (CPU)")


def test_loss_no_hardcoded_cpu():
    """Test that loss functions don't hardcode CPU tensors."""
    # Check source code
    brt_source = inspect.getsource(losses.init_brt_hjivi_loss)
    brat_source = inspect.getsource(losses.init_brat_hjivi_loss)

    # Should not use torch.Tensor([0]) which always creates CPU tensor
    assert "torch.Tensor([0])" not in brt_source, (
        "BRT loss should not use torch.Tensor([0]) - use device-aware creation"
    )

    assert "torch.Tensor([0])" not in brat_source, (
        "BRAT loss should not use torch.Tensor([0]) - use device-aware creation"
    )

    print("✓ Loss functions don't hardcode CPU tensors")


def test_loss_gpu_compatible():
    """Test that loss functions work on GPU if available."""
    if not torch.cuda.is_available():
        print("⊘ GPU not available, skipping GPU compatibility test")
        return

    dubins = dynamics.Dubins3D(
        goalR=0.25,
        velocity=0.6,
        omega_max=1.1,
        angle_alpha_factor=1.2,
        set_mode="avoid",
        freeze_model=False,
    )
    dubins.deepreach_model = "exact"

    loss_fn = losses.init_brt_hjivi_loss(
        dubins, minWith="target", dirichlet_loss_divisor=1.0
    )

    # Test with GPU tensors
    batch_size = 5
    state = torch.randn(batch_size, 3).cuda()
    value = torch.randn(batch_size).cuda()
    dvdt = torch.randn(batch_size).cuda()
    dvds = torch.randn(batch_size, 3).cuda()
    boundary_value = torch.randn(batch_size).cuda()
    dirichlet_mask = torch.ones(batch_size, dtype=torch.bool).cuda()
    output = torch.randn(batch_size, 1).cuda()

    # This should not raise device mismatch error
    result = loss_fn(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output)

    # Check result is on GPU
    assert result["diff_constraint_hom"].is_cuda, (
        "Result should be on GPU when inputs are on GPU"
    )

    print("✓ Loss functions work correctly on GPU")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Device Mismatch Fix in Loss Functions")
    print("=" * 70)
    print()

    try:
        test_brt_loss_device()
        test_brat_loss_device()
        test_loss_no_hardcoded_cpu()
        test_loss_gpu_compatible()

        print()
        print("=" * 70)
        print("All tests PASSED! ✓")
        print("The device mismatch bug is fixed.")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test FAILED: {e}")
        print("The bug still exists or fix is incomplete.")
        print("=" * 70)
        sys.exit(1)
    except RuntimeError as e:
        if "device" in str(e).lower():
            print()
            print("=" * 70)
            print(f"Test FAILED with device error: {e}")
            print("The device mismatch bug still exists!")
            print("=" * 70)
            sys.exit(1)
        raise
