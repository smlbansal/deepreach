import torch
import torch.nn.functional as F

import diff_operators
import modules, utils

import math
import numpy as np


def initialize_hji_MultiVehicleCollisionNE(dataset, minWith):
    # Initialize the loss function for the multi-vehicle collision avoidance problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    numEvaders = dataset.numEvaders
    num_pos_states = dataset.num_pos_states
    alpha_angle = dataset.alpha_angle
    alpha_time = dataset.alpha_time

    def hji_MultiVehicleCollision(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            # Scale the costate for theta appropriately to align with the range of [-pi, pi]
            dudx[..., num_pos_states:] = dudx[..., num_pos_states:] / alpha_angle

            # Compute the hamiltonian for the ego vehicle
            ham = velocity*(torch.cos(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 0] + torch.sin(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 1]) - omega_max * torch.abs(dudx[..., num_pos_states])

            # Hamiltonian effect due to other vehicles
            for i in range(numEvaders):
                theta_index = num_pos_states+1+i+1
                xcostate_index = 2*(i+1)
                ycostate_index = 2*(i+1) + 1
                thetacostate_index = num_pos_states+1+i
                ham_local = velocity*(torch.cos(alpha_angle*x[..., theta_index]) * dudx[..., xcostate_index] + torch.sin(alpha_angle*x[..., theta_index]) * dudx[..., ycostate_index]) + omega_max * torch.abs(dudx[..., thetacostate_index])
                ham = ham + ham_local

            # Effect of time factor
            ham = ham * alpha_time

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_MultiVehicleCollision


def initialize_hji_air3D(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha_angle = dataset.alpha_angle

    def hji_air3D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_theta = x[..., 3] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 2] = dudx[..., 2] / alpha_angle
        # Scale the coordinates
        x_theta = alpha_angle * x_theta

        # Air3D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a

        # Compute the hamiltonian for the ego vehicle
        ham = omega_max * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])  # Control component
        ham = ham - omega_max * torch.abs(dudx[..., 2])  # Disturbance component
        ham = ham + (velocity * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (velocity * torch.sin(x_theta) * dudx[..., 1])  # Constant component

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_air3D
