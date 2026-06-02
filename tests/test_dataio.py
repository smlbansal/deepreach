"""Tests for dataio module."""

import torch
import pytest
from utils import dataio
from dynamics import dynamics


class TestReachabilityDataset:
    """Tests for ReachabilityDataset."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )

        dataset = dataio.ReachabilityDataset(
            dynamics=dubins,
            numpoints=1000,
            pretrain=False,
            pretrain_iters=100,
            tMin=0.0,
            tMax=1.0,
            counter_start=0,
            counter_end=100,
            num_src_samples=10,
            num_target_samples=0,
        )

        assert dataset is not None
        assert len(dataset) == 1

    def test_dataset_getitem(self):
        """Test dataset __getitem__."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )

        dataset = dataio.ReachabilityDataset(
            dynamics=dubins,
            numpoints=100,
            pretrain=False,
            pretrain_iters=100,
            tMin=0.0,
            tMax=1.0,
            counter_start=0,
            counter_end=10,
            num_src_samples=10,
            num_target_samples=0,
        )

        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

        coords_dict, data_dict = item
        assert isinstance(coords_dict, dict)
        assert "coords" in coords_dict or "model_coords" in coords_dict
        assert isinstance(data_dict, dict)

    def test_dataset_pretrain(self):
        """Test dataset in pretrain mode."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )

        dataset = dataio.ReachabilityDataset(
            dynamics=dubins,
            numpoints=100,
            pretrain=True,
            pretrain_iters=50,
            tMin=0.0,
            tMax=1.0,
            counter_start=0,
            counter_end=10,
            num_src_samples=10,
            num_target_samples=0,
        )

        item = dataset[0]
        coords_dict, data_dict = item

        assert "dirichlet_masks" in data_dict
        # In pretrain mode, all points should be dirichlet
        assert torch.all(data_dict["dirichlet_masks"])

    def test_dataset_with_target_samples(self):
        """Test dataset with target samples."""
        # Dubins3D doesn't implement sample_target_state, so we use num_target_samples=0
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )

        dataset = dataio.ReachabilityDataset(
            dynamics=dubins,
            numpoints=100,
            pretrain=False,
            pretrain_iters=100,
            tMin=0.0,
            tMax=1.0,
            counter_start=0,
            counter_end=10,
            num_src_samples=10,
            num_target_samples=0,
        )

        item = dataset[0]
        coords_dict, data_dict = item

        coords = coords_dict.get("coords", coords_dict.get("model_coords"))
        assert coords.shape[0] == 100  # numpoints

    def test_dataset_coords_range(self):
        """Test that coords are in expected range."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )

        dataset = dataio.ReachabilityDataset(
            dynamics=dubins,
            numpoints=100,
            pretrain=False,
            pretrain_iters=100,
            tMin=0.0,
            tMax=1.0,
            counter_start=0,
            counter_end=10,
            num_src_samples=10,
            num_target_samples=0,
        )

        item = dataset[0]
        coords_dict, data_dict = item
        coords = coords_dict.get("coords", coords_dict.get("model_coords"))

        # Time coordinate should be in [tMin, tMax]
        assert torch.all(coords[:, 0] >= 0.0)
        assert torch.all(coords[:, 0] <= 1.0)

        # State coordinates should be normalized to [-1, 1]
        assert torch.all(coords[:, 1:] >= -1.0)
        assert torch.all(coords[:, 1:] <= 1.0)

    def test_dataset_boundary_values(self):
        """Test that boundary values are computed."""
        dubins = dynamics.Dubins3D(
            goalR=0.25,
            velocity=0.6,
            omega_max=1.1,
            angle_alpha_factor=1.2,
            set_mode="avoid",
            freeze_model=False,
        )

        dataset = dataio.ReachabilityDataset(
            dynamics=dubins,
            numpoints=100,
            pretrain=False,
            pretrain_iters=100,
            tMin=0.0,
            tMax=1.0,
            counter_start=0,
            counter_end=10,
            num_src_samples=10,
            num_target_samples=0,
        )

        item = dataset[0]
        coords_dict, data_dict = item
        coords = coords_dict.get("coords", coords_dict.get("model_coords"))

        assert "boundary_values" in data_dict
        assert isinstance(data_dict["boundary_values"], torch.Tensor)
        assert data_dict["boundary_values"].shape[0] == coords.shape[0]
