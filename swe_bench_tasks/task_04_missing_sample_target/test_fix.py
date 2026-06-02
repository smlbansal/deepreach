"""
Test script to verify sample_target_state implementations.

This test should FAIL before the fix and PASS after the fix.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

import torch
from dynamics import dynamics


def test_dubins3d_sample_target():
    """Test Dubins3D.sample_target_state implementation."""
    dubins = dynamics.Dubins3D(
        goalR=0.25,
        velocity=0.6,
        omega_max=1.1,
        angle_alpha_factor=1.2,
        set_mode="avoid",
        freeze_model=False,
    )

    # Sample target states
    num_samples = 10
    samples = dubins.sample_target_state(num_samples)

    # Check shape
    assert samples.shape == (num_samples, 3), (
        f"Expected shape ({num_samples}, 3), got {samples.shape}"
    )

    # Check that samples are within target (boundary_fn <= 0)
    boundary_values = dubins.boundary_fn(samples)
    assert torch.all(boundary_values <= 1e-5), (
        "Sampled states should be within target set (boundary_fn <= 0)"
    )

    print("✓ Dubins3D.sample_target_state works correctly")


def test_air3d_sample_target():
    """Test Air3D.sample_target_state implementation."""
    air3d = dynamics.Air3D(
        collisionR=0.25,
        velocity=0.6,
        omega_max=1.1,
        angle_alpha_factor=1.2,
        set_mode="avoid",
        freeze_model=False,
    )

    # Sample target states
    num_samples = 10
    samples = air3d.sample_target_state(num_samples)

    # Check shape
    assert samples.shape == (num_samples, 3), (
        f"Expected shape ({num_samples}, 3), got {samples.shape}"
    )

    # Check that samples are within target
    boundary_values = air3d.boundary_fn(samples)
    assert torch.all(boundary_values <= 1e-5), (
        "Sampled states should be within target set"
    )

    print("✓ Air3D.sample_target_state works correctly")


def test_sample_target_state_not_implemented():
    """Test that sample_target_state no longer raises NotImplementedError."""
    dynamics_to_test = [
        (
            "Dubins3D",
            {
                "goalR": 0.25,
                "velocity": 0.6,
                "omega_max": 1.1,
                "angle_alpha_factor": 1.2,
                "set_mode": "avoid",
                "freeze_model": False,
            },
        ),
        (
            "Air3D",
            {
                "collisionR": 0.25,
                "velocity": 0.6,
                "omega_max": 1.1,
                "angle_alpha_factor": 1.2,
                "set_mode": "avoid",
                "freeze_model": False,
            },
        ),
    ]

    for name, kwargs in dynamics_to_test:
        cls = getattr(dynamics, name)
        dyn = cls(**kwargs)

        try:
            samples = dyn.sample_target_state(5)
            # Should not raise NotImplementedError
            assert samples is not None, f"{name}.sample_target_state returned None"
            assert samples.shape[0] == 5, (
                f"{name}.sample_target_state returned wrong number of samples"
            )
            print(f"✓ {name}.sample_target_state implemented")
        except NotImplementedError:
            raise AssertionError(
                f"{name}.sample_target_state still raises NotImplementedError"
            )


def test_dataset_with_target_samples():
    """Test that dataset works with num_target_samples > 0."""
    from utils import dataio

    dubins = dynamics.Dubins3D(
        goalR=0.25,
        velocity=0.6,
        omega_max=1.1,
        angle_alpha_factor=1.2,
        set_mode="avoid",
        freeze_model=False,
    )

    # This should work now (previously raised NotImplementedError)
    dataset = dataio.ReachabilityDataset(
        dynamics=dubins,
        numpoints=100,
        pretrain=False,
        pretrain_iters=100,
        tMin=0.0,
        tMax=1.0,
        counter_start=0,
        counter_end=10,
        num_src_samples=10,
        num_target_samples=20,  # This requires sample_target_state
    )

    # Get a sample
    item = dataset[0]
    coords_dict, data_dict = item
    coords = coords_dict.get("coords", coords_dict.get("model_coords"))

    # Should have numpoints + num_target_samples
    assert coords.shape[0] == 120, (
        f"Expected 120 samples (100 + 20), got {coords.shape[0]}"
    )

    print("✓ Dataset works with num_target_samples > 0")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing sample_target_state Implementations")
    print("=" * 70)
    print()

    try:
        test_dubins3d_sample_target()
        test_air3d_sample_target()
        test_sample_target_state_not_implemented()
        test_dataset_with_target_samples()

        print()
        print("=" * 70)
        print("All tests PASSED! ✓")
        print("sample_target_state is implemented for all dynamics classes.")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test FAILED: {e}")
        print("sample_target_state not fully implemented.")
        print("=" * 70)
        sys.exit(1)
    except NotImplementedError as e:
        print()
        print("=" * 70)
        print(f"Test FAILED: NotImplementedError raised")
        print("sample_target_state not implemented for some dynamics.")
        print("=" * 70)
        sys.exit(1)
