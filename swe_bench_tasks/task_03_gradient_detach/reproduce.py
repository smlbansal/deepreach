"""
Reproduction script for gradient computation bug in SingleBVPNet.

This script demonstrates that gradients don't flow back to input coordinates
because of .detach() in the forward method.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

import torch
from utils import modules

print("=" * 70)
print("Reproducing Gradient Computation Bug in SingleBVPNet")
print("=" * 70)
print()

# Create model
print("Test 1: Creating SingleBVPNet...")
net = modules.SingleBVPNet(
    in_features=4,
    out_features=1,
    type="sine",
    mode="mlp",
    hidden_features=32,
    num_hidden_layers=2,
)
print("✓ Model created")
print()

# Test 2: Check if gradients flow to input
print("Test 2: Testing gradient flow to input coordinates...")
print("-" * 70)

# Create input with requires_grad=True
coords = torch.randn(5, 4, requires_grad=True)
print(f"Input coords.requires_grad: {coords.requires_grad}")
print(f"Input coords.grad before forward: {coords.grad}")

# Forward pass
model_input = {"coords": coords}
output = net(model_input)

print(f"Output keys: {output.keys()}")
print(f"model_in is coords: {output['model_in'] is coords}")
print(f"model_in.requires_grad: {output['model_in'].requires_grad}")

# Backward pass
loss = output["model_out"].sum()
loss.backward()

print(f"Input coords.grad after backward: {coords.grad}")
print(f"model_in.grad after backward: {output['model_in'].grad}")

print()

if coords.grad is None:
    print("❌ BUG CONFIRMED: Input coords.grad is None!")
    print("   Gradients did NOT flow back to the original input.")
    print("   This is because .detach() in forward() breaks the computation graph.")
    bug_exists = True
else:
    print("✓ Gradients flowed back to input (bug may be fixed)")
    bug_exists = False

print()
print("Test 3: Checking if model_in is detached...")
print("-" * 70)

# Check if model_in shares storage with coords (it shouldn't if detached)
coords2 = torch.randn(5, 4, requires_grad=True)
model_input2 = {"coords": coords2}
output2 = net(model_input2)

# If detached, model_in will be a different tensor
if output2["model_in"] is not coords2:
    print("❌ model_in is a different tensor from input coords")
    print("   This confirms .clone().detach() is being used.")
    print("   The computation graph is broken!")
    bug_exists = True
else:
    print("✓ model_in is the same tensor as input coords")

print()
print("=" * 70)
if bug_exists:
    print("RESULT: Bug reproduced successfully!")
    print("Gradients cannot flow back to input due to .detach() in forward().")
    print("This prevents end-to-end differentiability.")
else:
    print("RESULT: Bug not reproduced or already fixed.")
    print("Gradients flow correctly to input coordinates.")
print("=" * 70)
