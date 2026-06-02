"""
Reproduction script for missing BRS support.

This script demonstrates that BRS (Backward Reachable Set) is not
fully implemented in the experiments code.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

print("=" * 70)
print("Reproducing Missing BRS Support")
print("=" * 70)
print()

# Check for TODOs in experiments.py
print("Test 1: Checking for BRS TODOs in experiments.py...")
print("-" * 70)

with open("/Users/zhih/AAI/deepreach/experiments/experiments.py", "r") as f:
    content = f.read()
    lines = content.split("\n")

    brs_todos = []
    for i, line in enumerate(lines, 1):
        if "BRS" in line and "TODO" in line:
            brs_todos.append((i, line.strip()))

if brs_todos:
    print(f"Found {len(brs_todos)} TODOs related to BRS:")
    for line_num, line in brs_todos:
        print(f"  Line {line_num}: {line}")
    print()
    print("❌ BRS support is incomplete (TODOs found)")
    bug_exists = True
else:
    print("No BRS TODOs found")
    bug_exists = False

print()

# Check if set_type parameter is exposed in test method
print("Test 2: Checking if BRS can be selected in tests...")
print("-" * 70)

import inspect
from experiments import experiments

test_sig = inspect.signature(experiments.Experiment.test)
params = list(test_sig.parameters.keys())

print(f"Experiment.test parameters: {params}")

if "set_type" in params:
    print("✓ set_type parameter exists")
    # Check if it's actually used for BRS
    source = inspect.getsource(experiments.Experiment.test)
    if 'set_type="BRT"' in source and "BRS" not in source:
        print("❌ set_type is hardcoded to BRT, BRS not implemented")
        bug_exists = True
    else:
        print("✓ BRS appears to be implemented")
        bug_exists = False
else:
    print("❌ set_type parameter not found in test method")
    bug_exists = True

print()
print("=" * 70)
if bug_exists:
    print("RESULT: Bug reproduced!")
    print("BRS support is not fully implemented.")
else:
    print("RESULT: BRS support appears to be implemented.")
print("=" * 70)
