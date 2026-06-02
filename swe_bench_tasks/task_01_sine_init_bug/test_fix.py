"""
Test script to verify the Sine module initialization bug is fixed.

This test should FAIL before the fix and PASS after the fix.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

from utils import modules
import torch
import inspect


def test_sine_has_proper_init():
    """Test that Sine class has properly named __init__ method."""
    # Check that Sine defines its own __init__ (not inherited from nn.Module)
    assert hasattr(modules.Sine, "__init__"), "Sine class should have __init__ method"

    # Get the source code of Sine class
    source = inspect.getsource(modules.Sine)

    # Check that __init__ is properly defined (with 2 underscores on each side)
    assert "def __init__(self):" in source, (
        "Sine class should define __init__ with proper naming (2 underscores each side)"
    )

    # Make sure it's not the buggy version
    assert "def __init(self):" not in source, (
        "Sine class should NOT have misspelled __init method"
    )

    print("✓ Sine class has properly named __init__ method")


def test_sine_init_is_called():
    """Test that Sine.__init__ is actually called during instantiation."""
    # Create a test subclass to verify __init__ is called
    init_called = []

    class TestSine(modules.Sine):
        def __init__(self):
            init_called.append(True)
            super().__init__()

    # Create instance
    sine = TestSine()

    # Verify __init__ was called
    assert len(init_called) == 1, "Sine.__init__ should be called during instantiation"
    assert init_called[0] is True, "Sine.__init__ should be called"

    print("✓ Sine.__init__ is called during instantiation")


def test_sine_functionality():
    """Test that Sine module still works correctly after fix."""
    sine = modules.Sine()

    # Test forward pass
    x = torch.tensor([0.0, 3.14159 / 60, 3.14159 / 30])
    output = sine(x)

    # sin(30 * 0) = sin(0) = 0
    # sin(30 * pi/60) = sin(pi/2) = 1
    # sin(30 * pi/30) = sin(pi) = 0
    expected = torch.tensor([0.0, 1.0, 0.0])

    assert torch.allclose(output, expected, atol=1e-5), (
        f"Sine forward pass incorrect: got {output}, expected {expected}"
    )

    print("✓ Sine module functionality works correctly")


def test_sine_is_nn_module():
    """Test that Sine is still a proper nn.Module."""
    sine = modules.Sine()

    assert isinstance(sine, torch.nn.Module), "Sine should be an instance of nn.Module"

    # Should be able to call .parameters() even if no parameters
    params = list(sine.parameters())
    assert isinstance(params, list), "Should be able to get parameters"

    print("✓ Sine is a proper nn.Module")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Sine Module Initialization Fix")
    print("=" * 70)
    print()

    try:
        test_sine_has_proper_init()
        test_sine_init_is_called()
        test_sine_functionality()
        test_sine_is_nn_module()

        print()
        print("=" * 70)
        print("All tests PASSED! ✓")
        print("The Sine module initialization bug is fixed.")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test FAILED: {e}")
        print("The bug still exists or fix is incomplete.")
        print("=" * 70)
        sys.exit(1)
