"""
Reproduction script for missing sample_target_state implementations.

This script demonstrates that many dynamics classes raise NotImplementedError
when sample_target_state is called.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

from dynamics import dynamics
import torch

print("=" * 70)
print("Reproducing Missing sample_target_state Implementations")
print("=" * 70)
print()

# List of dynamics classes to test
dynamics_to_test = [
    (
        "ParameterizedVertDrone2D",
        {"gravity": 9.81, "input_multiplier_max": 1.5, "input_magnitude_max": 10.0},
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
        "Dubins4D",
        {
            "goalR": 0.25,
            "velocity": 0.6,
            "omega_max": 1.1,
            "a_max": 1.0,
            "angle_alpha_factor": 1.2,
            "set_mode": "avoid",
            "freeze_model": False,
        },
    ),
]

print("Testing sample_target_state for various dynamics classes...")
print("-" * 70)

not_implemented = []
implemented = []

for name, kwargs in dynamics_to_test:
    print(f"\n{name}:")
    try:
        # Create dynamics instance
        cls = getattr(dynamics, name)
        dyn = cls(**kwargs)

        # Try to sample target states
        try:
            samples = dyn.sample_target_state(5)
            print(f"  ✓ sample_target_state works")
            print(f"    Returned shape: {samples.shape}")
            print(f"    Expected shape: (5, {dyn.state_dim})")
            if samples.shape == (5, dyn.state_dim):
                print(f"    ✓ Shape correct")
                implemented.append(name)
            else:
                print(f"    ✗ Shape incorrect")
                not_implemented.append(name)
        except NotImplementedError:
            print(f"  ✗ sample_target_state raises NotImplementedError")
            not_implemented.append(name)
        except Exception as e:
            print(f"  ✗ sample_target_state raised: {type(e).__name__}: {e}")
            not_implemented.append(name)

    except Exception as e:
        print(f"  ✗ Failed to create instance: {e}")
        not_implemented.append(name)

print()
print("=" * 70)
print("Summary:")
print(f"  Implemented: {len(implemented)} classes")
print(f"  Not implemented: {len(not_implemented)} classes")
print()

if not_implemented:
    print("Classes with missing implementations:")
    for name in not_implemented:
        print(f"  - {name}")
    print()
    print("RESULT: Bug reproduced!")
    print("Many dynamics classes don't implement sample_target_state().")
    bug_exists = True
else:
    print("RESULT: All classes implement sample_target_state().")
    print("Bug may be fixed or not reproduced.")
    bug_exists = False

print("=" * 70)
