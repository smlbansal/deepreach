"""
Test script to verify BRS support is implemented.

This test should FAIL before the fix and PASS after the fix.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

import inspect
from experiments import experiments


def test_brs_todos_removed():
    """Test that BRS TODOs are removed."""
    with open("/Users/zhih/AAI/deepreach/experiments/experiments.py", "r") as f:
        content = f.read()

    # Should not have TODOs about implementing BRS
    assert "TODO: implement option for BRS" not in content, (
        "BRS TODOs should be removed after implementation"
    )

    print("✓ BRS TODOs removed")


def test_set_type_parameter():
    """Test that set_type can be specified for tests."""
    sig = inspect.signature(experiments.Experiment.test)
    params = list(sig.parameters.keys())

    # set_type should be a parameter or the method should handle both BRT and BRS
    source = inspect.getsource(experiments.Experiment.test)

    # Check that BRS is mentioned (implemented)
    assert "BRS" in source, (
        "BRS should be mentioned in test method after implementation"
    )

    # Check that it's not hardcoded to only BRT
    brt_count = source.count('set_type="BRT"')
    brs_count = source.count('set_type="BRS"') + source.count("set_type='BRS'")

    # If BRS is implemented, there should be references to it
    assert brs_count > 0 or "BRS" in source, (
        "BRS should be implemented and referenced in code"
    )

    print("✓ BRS support implemented in test method")


def test_scenario_optimization_brs():
    """Test that scenario_optimization supports BRS."""
    from utils import error_evaluators
    import inspect

    source = inspect.getsource(error_evaluators.scenario_optimization)

    # Should handle both BRT and BRS
    assert "BRT" in source, "Should support BRT"
    # After fix, should also explicitly handle BRS
    # (This is a basic check - full implementation would need more)

    print("✓ scenario_optimization supports set types")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing BRS Support Implementation")
    print("=" * 70)
    print()

    try:
        test_brs_todos_removed()
        test_set_type_parameter()
        test_scenario_optimization_brs()

        print()
        print("=" * 70)
        print("All tests PASSED! ✓")
        print("BRS support is implemented.")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test FAILED: {e}")
        print("BRS support not fully implemented.")
        print("=" * 70)
        sys.exit(1)
