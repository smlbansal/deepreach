"""
Test script to verify the gradient computation bug is fixed.

This test should FAIL before the fix and PASS after the fix.
"""

import sys

sys.path.insert(0, "/Users/zhih/AAI/deepreach")

import torch
from utils import modules


def test_gradient_flows_to_input():
    """Test that gradients flow back to input coordinates."""
    net = modules.SingleBVPNet(
        in_features=4,
        out_features=1,
        type="sine",
        mode="mlp",
        hidden_features=32,
        num_hidden_layers=2,
    )

    # Create input with requires_grad=True
    coords = torch.randn(5, 4, requires_grad=True)
    model_input = {"coords": coords}

    # Forward pass
    output = net(model_input)

    # Backward pass
    loss = output["model_out"].sum()
    loss.backward()

    # Check that gradients flowed back to input
    assert coords.grad is not None, (
        "Gradients should flow back to input coords. "
        "If None, the computation graph is broken by .detach()."
    )

    # Check that gradients are non-zero
    assert torch.any(coords.grad != 0), "Input gradients should be non-zero"

    print("✓ Gradients flow back to input coordinates")


def test_model_in_is_input():
    """Test that model_in is the same tensor as input (not detached)."""
    net = modules.SingleBVPNet(
        in_features=4,
        out_features=1,
        type="sine",
        mode="mlp",
        hidden_features=32,
        num_hidden_layers=2,
    )

    coords = torch.randn(5, 4, requires_grad=True)
    model_input = {"coords": coords}

    output = net(model_input)

    # model_in should be the same tensor or a view of coords
    # If .detach() is used, it will be a different tensor
    assert (
        output["model_in"] is coords
        or output["model_in"].storage().data_ptr() == coords.storage().data_ptr()
    ), "model_in should share storage with input coords (not detached)"

    print("✓ model_in shares computation graph with input")


def test_end_to_end_differentiability():
    """Test end-to-end differentiability with a perception module."""

    # Simulate a perception module that outputs coordinates
    class PerceptionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 4)

        def forward(self, x):
            return self.linear(x)

    perception = PerceptionModule()
    deepreach = modules.SingleBVPNet(
        in_features=4,
        out_features=1,
        type="sine",
        mode="mlp",
        hidden_features=32,
        num_hidden_layers=2,
    )

    # Forward pass through perception then DeepReach
    x = torch.randn(5, 10)
    coords = perception(x)
    coords = coords.requires_grad_(True)
    coords.retain_grad()

    model_input = {"coords": coords}
    output = deepreach(model_input)

    loss = output["model_out"].sum()
    loss.backward()

    # Check that gradients flowed back through both networks
    assert coords.grad is not None, "Gradients should flow back to perception output"

    # Check perception module received gradients
    for param in perception.parameters():
        assert param.grad is not None, "Perception module should receive gradients"

    print("✓ End-to-end differentiability works")


def test_no_detach_in_forward():
    """Test that forward method doesn't use detach."""
    import inspect

    source = inspect.getsource(modules.SingleBVPNet.forward)

    # Should not contain .detach() call
    assert ".detach()" not in source, (
        "SingleBVPNet.forward should not use .detach() as it breaks gradient flow. "
        "Use .retain_grad() on inputs instead."
    )

    print("✓ Forward method doesn't use .detach()")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Gradient Computation Fix in SingleBVPNet")
    print("=" * 70)
    print()

    try:
        test_gradient_flows_to_input()
        test_model_in_is_input()
        test_end_to_end_differentiability()
        test_no_detach_in_forward()

        print()
        print("=" * 70)
        print("All tests PASSED! ✓")
        print("The gradient computation bug is fixed.")
        print("End-to-end differentiability works correctly.")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test FAILED: {e}")
        print("The bug still exists or fix is incomplete.")
        print("=" * 70)
        sys.exit(1)
