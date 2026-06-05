"""Tests for losses module."""

import torch
import pytest
from utils import losses
from dynamics import dynamics


class TestBRT_HJIVI_Loss:
    """Tests for BRT HJIVI loss function."""

    def test_loss_creation(self):
        """Test loss function creation."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )
        dubins.deepreach_model = "exact"

        loss_fn = losses.init_brt_hjivi_loss(
            dubins, minWith="target", dirichlet_loss_divisor=1.0
        )
        assert loss_fn is not None
        assert callable(loss_fn)

    def test_loss_pretraining(self):
        """Test loss in pretraining mode (all dirichlet)."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )
        dubins.deepreach_model = "exact"

        loss_fn = losses.init_brt_hjivi_loss(
            dubins, minWith="target", dirichlet_loss_divisor=1.0
        )

        # All dirichlet points (pretraining)
        batch_size = 5
        state = torch.randn(batch_size, 3)
        value = torch.randn(batch_size)
        dvdt = torch.randn(batch_size)
        dvds = torch.randn(batch_size, 3)
        boundary_value = torch.randn(batch_size)
        dirichlet_mask = torch.ones(batch_size, dtype=torch.bool)
        output = torch.randn(batch_size, 1)

        result = loss_fn(
            state, value, dvdt, dvds, boundary_value, dirichlet_mask, output
        )

        assert "dirichlet" in result
        assert isinstance(result["dirichlet"], torch.Tensor)

    def test_loss_with_minwith_zero(self):
        """Test loss with minWith='zero'."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )
        dubins.deepreach_model = "vanilla"

        loss_fn = losses.init_brt_hjivi_loss(
            dubins, minWith="zero", dirichlet_loss_divisor=1.0
        )

        batch_size = 5
        state = torch.randn(batch_size, 3)
        value = torch.randn(batch_size)
        dvdt = torch.randn(batch_size)
        dvds = torch.randn(batch_size, 3)
        boundary_value = torch.randn(batch_size)
        dirichlet_mask = torch.zeros(batch_size, dtype=torch.bool)
        output = torch.randn(batch_size, 1)

        result = loss_fn(
            state, value, dvdt, dvds, boundary_value, dirichlet_mask, output
        )

        assert "diff_constraint_hom" in result
        assert "dirichlet" in result

    def test_loss_with_minwith_target(self):
        """Test loss with minWith='target'."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )
        dubins.deepreach_model = "vanilla"

        loss_fn = losses.init_brt_hjivi_loss(
            dubins, minWith="target", dirichlet_loss_divisor=2.0
        )

        batch_size = 5
        state = torch.randn(batch_size, 3)
        value = torch.randn(batch_size)
        dvdt = torch.randn(batch_size)
        dvds = torch.randn(batch_size, 3)
        boundary_value = torch.randn(batch_size)
        dirichlet_mask = torch.zeros(batch_size, dtype=torch.bool)
        output = torch.randn(batch_size, 1)

        result = loss_fn(
            state, value, dvdt, dvds, boundary_value, dirichlet_mask, output
        )

        assert "diff_constraint_hom" in result
        assert "dirichlet" in result

    def test_loss_with_minwith_none(self):
        """Test loss with minWith='none'."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )
        dubins.deepreach_model = "vanilla"

        loss_fn = losses.init_brt_hjivi_loss(
            dubins, minWith="none", dirichlet_loss_divisor=1.0
        )

        batch_size = 5
        state = torch.randn(batch_size, 3)
        value = torch.randn(batch_size)
        dvdt = torch.randn(batch_size)
        dvds = torch.randn(batch_size, 3)
        boundary_value = torch.randn(batch_size)
        dirichlet_mask = torch.zeros(batch_size, dtype=torch.bool)
        output = torch.randn(batch_size, 1)

        result = loss_fn(
            state, value, dvdt, dvds, boundary_value, dirichlet_mask, output
        )

        assert "diff_constraint_hom" in result


class TestBRAT_HJIVI_Loss:
    """Tests for BRAT HJIVI loss function."""

    def test_brat_loss_creation(self):
        """Test BRAT loss function creation."""
        # Use NarrowPassage which has reach_fn and avoid_fn
        # avoid_only=False means it uses brat_hjivi loss type
        narrow = dynamics.NarrowPassage(avoid_fn_weight=1.0, avoid_only=False)
        narrow.deepreach_model = "exact"

        loss_fn = losses.init_brat_hjivi_loss(
            narrow, minWith="target", dirichlet_loss_divisor=1.0
        )
        assert loss_fn is not None
        assert callable(loss_fn)

    def test_brat_loss_pretraining(self):
        """Test BRAT loss in pretraining mode."""
        narrow = dynamics.NarrowPassage(avoid_fn_weight=1.0, avoid_only=False)
        narrow.deepreach_model = "exact"

        loss_fn = losses.init_brat_hjivi_loss(
            narrow, minWith="target", dirichlet_loss_divisor=1.0
        )

        batch_size = 5
        state = torch.randn(batch_size, 10)
        value = torch.randn(batch_size)
        dvdt = torch.randn(batch_size)
        dvds = torch.randn(batch_size, 10)
        boundary_value = torch.randn(batch_size)
        reach_value = torch.randn(batch_size)
        avoid_value = torch.randn(batch_size)
        dirichlet_mask = torch.ones(batch_size, dtype=torch.bool)
        output = torch.randn(batch_size, 1)

        result = loss_fn(
            state,
            value,
            dvdt,
            dvds,
            boundary_value,
            reach_value,
            avoid_value,
            dirichlet_mask,
            output,
        )

        assert "dirichlet" in result

    def test_brat_loss_with_minwith_zero(self):
        """Test BRAT loss with minWith='zero'."""
        narrow = dynamics.NarrowPassage(avoid_fn_weight=1.0, avoid_only=False)
        narrow.deepreach_model = "vanilla"

        loss_fn = losses.init_brat_hjivi_loss(
            narrow, minWith="zero", dirichlet_loss_divisor=1.0
        )

        batch_size = 5
        state = torch.randn(batch_size, 10)
        value = torch.randn(batch_size)
        dvdt = torch.randn(batch_size)
        dvds = torch.randn(batch_size, 10)
        boundary_value = torch.randn(batch_size)
        reach_value = torch.randn(batch_size)
        avoid_value = torch.randn(batch_size)
        dirichlet_mask = torch.zeros(batch_size, dtype=torch.bool)
        output = torch.randn(batch_size, 1)

        result = loss_fn(
            state,
            value,
            dvdt,
            dvds,
            boundary_value,
            reach_value,
            avoid_value,
            dirichlet_mask,
            output,
        )

        assert "diff_constraint_hom" in result
        assert "dirichlet" in result
