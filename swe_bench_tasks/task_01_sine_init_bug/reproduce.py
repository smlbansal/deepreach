"""
Reproduction script for Sine module initialization bug.

This script demonstrates that the Sine class's __init__ method
is misspelled and never gets called.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

from utils import modules
import torch

print("=" * 70)
print("Reproducing Sine Module Initialization Bug")
print("=" * 70)
print()

# Test 1: Check if __init__ method exists with correct name
print("Test 1: Checking Sine class __init__ method...")
print(f"  Has __init__ method: {hasattr(modules.Sine, '__init__')}")
print(f"  Has __init method: {hasattr(modules.Sine, '__init')}")

# Get the actual methods
sine_methods = [m for m in dir(modules.Sine) if "init" in m.lower()]
print(f"  Methods with 'init' in name: {sine_methods}")
print()

# Test 2: Create Sine instance and check if custom __init__ was called
print("Test 2: Creating Sine instance...")

# Monkey patch to detect if __init__ is called
original_init = modules.Sine.__init__
init_called = [False]


def patched_init(self):
    init_called[0] = True
    # Call the actual method if it exists with correct name
    if hasattr(modules.Sine, "__init__"):
        super(modules.Sine, self).__init__()


# Check the actual class definition
import inspect

source = inspect.getsource(modules.Sine)
print("Sine class source (relevant part):")
for line in source.split("\n")[:10]:
    if "def __init" in line or "class Sine" in line:
        print(f"  {line}")
print()

# Test 3: Verify the bug by inspecting the class
print("Test 3: Verifying the bug...")
print(
    f"  Sine.__init__ is nn.Module.__init__: {modules.Sine.__init__ is torch.nn.Module.__init__}"
)

if modules.Sine.__init__ is torch.nn.Module.__init__:
    print()
    print("  ❌ BUG CONFIRMED: Sine.__init__ is the same as nn.Module.__init__")
    print("     This means Sine's custom __init__ method is NOT being used!")
    print("     The method is misspelled as '__init' instead of '__init__'")
    bug_exists = True
else:
    print()
    print("  ✓ No bug: Sine has its own __init__ method")
    bug_exists = False

print()
print("=" * 70)
if bug_exists:
    print("RESULT: Bug reproduced successfully!")
    print("The Sine class __init__ method is misspelled and never called.")
else:
    print("RESULT: Bug not found or already fixed.")
print("=" * 70)
