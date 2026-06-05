"""Tests for modules module."""

import torch
import torch.nn as nn
import pytest
from utils import modules


class TestSine:
    """Tests for Sine activation module."""

    def test_sine_forward(self):
        """Test sine activation forward pass."""
        sine = modules.Sine()
        # Sine module multiplies by 30: sin(30 * x)
        x = torch.tensor([0.0, torch.pi / 60, torch.pi / 30])
        output = sine(x)

        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(output, expected, atol=1e-5)

    def test_sine_zero(self):
        """Test sine at zero."""
        sine = modules.Sine()
        x = torch.tensor([0.0])
        output = sine(x)

        expected = torch.tensor([0.0])
        assert torch.allclose(output, expected, atol=1e-6)


class TestBatchLinear:
    """Tests for BatchLinear module."""

    def test_batch_linear_shape(self):
        """Test BatchLinear output shape."""
        layer = modules.BatchLinear(10, 5)
        x = torch.randn(3, 10)
        output = layer(x)

        assert output.shape == (3, 5)

    def test_batch_linear_batch(self):
        """Test BatchLinear with batch dimension."""
        layer = modules.BatchLinear(4, 3)
        x = torch.randn(7, 4)
        output = layer(x)

        assert output.shape == (7, 3)
        # Output should be different for different inputs
        assert not torch.allclose(output[0], output[1])


class TestFCBlock:
    """Tests for FCBlock module."""

    def test_fcblock_sine(self):
        """Test FCBlock with sine activation."""
        block = modules.FCBlock(
            in_features=4,
            out_features=2,
            num_hidden_layers=2,
            hidden_features=16,
            outermost_linear=True,
            nonlinearity="sine",
        )
        x = torch.randn(5, 4)
        output = block(x)

        assert output.shape == (5, 2)

    def test_fcblock_relu(self):
        """Test FCBlock with ReLU activation."""
        block = modules.FCBlock(
            in_features=3,
            out_features=1,
            num_hidden_layers=1,
            hidden_features=8,
            outermost_linear=True,
            nonlinearity="relu",
        )
        x = torch.randn(4, 3)
        output = block(x)

        assert output.shape == (4, 1)

    def test_fcblock_tanh(self):
        """Test FCBlock with tanh activation."""
        block = modules.FCBlock(
            in_features=5,
            out_features=3,
            num_hidden_layers=3,
            hidden_features=10,
            outermost_linear=True,
            nonlinearity="tanh",
        )
        x = torch.randn(6, 5)
        output = block(x)

        assert output.shape == (6, 3)

    def test_fcblock_no_outermost_linear(self):
        """Test FCBlock without outermost linear layer."""
        block = modules.FCBlock(
            in_features=4,
            out_features=2,
            num_hidden_layers=2,
            hidden_features=8,
            outermost_linear=False,
            nonlinearity="sine",
        )
        x = torch.randn(3, 4)
        output = block(x)

        assert output.shape == (3, 2)


class TestSingleBVPNet:
    """Tests for SingleBVPNet module."""

    def test_singlebvpnet_creation(self):
        """Test SingleBVPNet creation."""
        net = modules.SingleBVPNet(
            in_features=4,
            out_features=1,
            type="sine",
            mode="mlp",
            hidden_features=64,
            num_hidden_layers=3,
        )
        assert net is not None
        assert isinstance(net.net, modules.FCBlock)

    def test_singlebvpnet_forward(self):
        """Test SingleBVPNet forward pass."""
        net = modules.SingleBVPNet(
            in_features=4,
            out_features=1,
            type="sine",
            mode="mlp",
            hidden_features=32,
            num_hidden_layers=2,
        )
        coords = torch.randn(10, 4)
        model_input = {"coords": coords}

        output = net(model_input)

        assert "model_in" in output
        assert "model_out" in output
        assert output["model_out"].shape == (10, 1)
        assert output["model_in"].shape == (10, 4)

    def test_singlebvpnet_gradients(self):
        """Test that gradients flow through SingleBVPNet."""
        net = modules.SingleBVPNet(
            in_features=3,
            out_features=1,
            type="sine",
            mode="mlp",
            hidden_features=16,
            num_hidden_layers=2,
        )
        coords = torch.randn(5, 3, requires_grad=True)
        model_input = {"coords": coords}

        output = net(model_input)
        loss = output["model_out"].sum()
        loss.backward()

        # Check that model parameters have gradients
        # Note: coords.grad may be None because model detaches input
        # but model_in should have gradients
        assert output["model_in"].grad is not None
        for param in net.parameters():
            assert param.grad is not None

    def test_singlebvpnet_different_types(self):
        """Test SingleBVPNet with different activation types."""
        for activation in ["sine", "tanh", "relu", "sigmoid"]:
            net = modules.SingleBVPNet(
                in_features=3,
                out_features=1,
                type=activation,
                mode="mlp",
                hidden_features=16,
                num_hidden_layers=2,
            )
            coords = torch.randn(4, 3)
            model_input = {"coords": coords}
            output = net(model_input)
            assert output["model_out"].shape == (4, 1)


class TestInitialization:
    """Tests for weight initialization functions."""

    def test_sine_init(self):
        """Test sine initialization."""
        layer = nn.Linear(10, 5)
        modules.sine_init(layer)
        # Just check it runs without error
        assert layer.weight is not None

    def test_first_layer_sine_init(self):
        """Test first layer sine initialization."""
        layer = nn.Linear(10, 5)
        modules.first_layer_sine_init(layer)
        assert layer.weight is not None

    def test_init_weights_normal(self):
        """Test normal weight initialization."""
        layer = nn.Linear(10, 5)
        modules.init_weights_normal(layer)
        assert layer.weight is not None

    def test_init_weights_xavier(self):
        """Test Xavier weight initialization."""
        layer = nn.Linear(10, 5)
        modules.init_weights_xavier(layer)
        assert layer.weight is not None
