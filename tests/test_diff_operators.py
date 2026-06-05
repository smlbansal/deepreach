"""Tests for diff_operators module."""

import torch
import pytest
from utils import diff_operators


class TestJacobian:
    """Tests for jacobian function."""

    def test_jacobian_simple(self):
        """Test jacobian of simple linear function."""
        # y = 2*x, dy/dx = 2
        x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        y = 2 * x
        jac, status = diff_operators.jacobian(y, x)

        assert jac.shape == (3, 1, 1)
        expected = torch.tensor([[[2.0]], [[2.0]], [[2.0]]])
        assert torch.allclose(jac, expected)
        assert status == 0

    def test_jacobian_quadratic(self):
        """Test jacobian of quadratic function."""
        # y = x^2, dy/dx = 2*x
        x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        y = x**2
        jac, status = diff_operators.jacobian(y, x)

        assert jac.shape == (3, 1, 1)
        expected = torch.tensor([[[2.0]], [[4.0]], [[6.0]]])
        assert torch.allclose(jac, expected)
        assert status == 0

    def test_jacobian_multivariate(self):
        """Test jacobian with multiple inputs and outputs."""
        # y1 = x1 + x2, y2 = x1 * x2
        # dy/dx = [[1, 1], [x2, x1]]
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y1 = x[:, 0] + x[:, 1]
        y2 = x[:, 0] * x[:, 1]
        y = torch.stack([y1, y2], dim=1)

        jac, status = diff_operators.jacobian(y, x)

        assert jac.shape == (2, 2, 2)
        # First sample: x=[1, 2], jac = [[1, 1], [2, 1]]
        expected_0 = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        assert torch.allclose(jac[0], expected_0)
        # Second sample: x=[3, 4], jac = [[1, 1], [4, 3]]
        expected_1 = torch.tensor([[1.0, 1.0], [4.0, 3.0]])
        assert torch.allclose(jac[1], expected_1)
        assert status == 0

    def test_jacobian_batch(self):
        """Test jacobian with batch dimension."""
        x = torch.randn(5, 3, requires_grad=True)
        y = torch.sum(x**2, dim=1, keepdim=True)
        jac, status = diff_operators.jacobian(y, x)

        assert jac.shape == (5, 1, 3)
        # dy/dx_i = 2*x_i
        expected = 2 * x.unsqueeze(1)
        assert torch.allclose(jac, expected)
        assert status == 0

    def test_jacobian_preserves_graph(self):
        """Test that jacobian computation preserves computation graph."""
        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        y = x**2
        jac, status = diff_operators.jacobian(y, x)

        # Should be able to backprop through jacobian
        loss = jac.sum()
        loss.backward()
        assert x.grad is not None
        assert status == 0
