from abc import ABC, abstractmethod
from utils import diff_operators, quaternion

import math
import torch
import numpy as np
from multiprocessing import Pool
import torch.nn as nn
import scipy.io as spio

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Dynamics device {}".format(device))
# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parametrized models


class Dynamics(ABC):
    def __init__(
        self,
        name: str,
        loss_type: str,
        set_mode: str,
        state_dim: int,
        input_dim: int,
        control_dim: int,
        disturbance_dim: int,
        state_mean: list,
        state_var: list,
        value_mean: float,
        value_var: float,
        value_normto: float,
        deepReach_model: bool,
    ):
        self.name = name
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean)
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepReach_model = deepReach_model

        assert self.loss_type in [
            "brt_hjivi",
            "brat_hjivi",
        ], f"loss type {self.loss_type} not recognized"
        if self.loss_type == "brat_hjivi":
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in [
            "reach",
            "avoid",
            "reach_avoid",
        ], f"set mode {self.set_mode} not recognized"
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, (
                "state descriptor dimension does not equal state dimension, "
                + str(len(state_descriptor))
                + " != "
                + str(self.state_dim)
            )

    # ALL METHODS ARE BATCH COMPATIBLE

    # set deepreach model. choices: "vanilla" (vanilla DeepReach V=NN(x,t)), diff (diff model V=NN(x,t) + l(x)), exact ( V=NN(x,t) + l(x))
    def set_model(self, deepreach_model):
        self.deepReach_model = deepreach_model

    # MODEL-UNIT CONVERSIONS
    # convert model input (normalized) to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (
            input[..., 1:] * self.state_var.to(device=input.device)
        ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord * 1.0
        input[..., 1:] = (
            coord[..., 1:] - self.state_mean.to(device=coord.device)
        ) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == "diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(
                self.input_to_coord(input)[..., 1:]
            )
        elif self.deepReach_model == "exact":
            return (
                output * input[..., 0] * self.value_var / self.value_normto
            ) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == "exact_diff":
            # Another way to impose exact BC: V(x,t)= l(x) + NN(x,t) - NN(x,0)
            output0 = output[0].squeeze(dim=-1)
            output1 = output[1].squeeze(dim=-1)
            return (
                output0 - output1
            ) * self.value_var / self.value_normto + self.boundary_fn(
                self.input_to_coord(input[0].detach())[..., 1:]
            )
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        if self.deepReach_model == "exact_diff":

            dodi1 = diff_operators.jacobian(output[0], input[0])[0].squeeze(dim=-2)
            dodi2 = diff_operators.jacobian(output[1], input[1])[0].squeeze(dim=-2)

            dvdt = (self.value_var / self.value_normto) * dodi1[..., 0]

            dvds_term1 = (
                self.value_var
                / self.value_normto
                / self.state_var.to(device=dodi1.device)
            ) * (dodi1[..., 1:] - dodi2[..., 1:])

            state = self.input_to_coord(input[0])[..., 1:]
            dvds_term2 = diff_operators.jacobian(
                self.boundary_fn(state).unsqueeze(dim=-1), state
            )[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
            return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(
            dim=-2
        )

        if self.deepReach_model == "diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (
                self.value_var
                / self.value_normto
                / self.state_var.to(device=dodi.device)
            ) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(
                self.boundary_fn(state).unsqueeze(dim=-1), state
            )[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        elif self.deepReach_model == "exact":

            dvdt = (self.value_var / self.value_normto) * (
                input[..., 0] * dodi[..., 0] + output
            )

            dvds_term1 = (
                (
                    self.value_var
                    / self.value_normto
                    / self.state_var.to(device=dodi.device)
                )
                * dodi[..., 1:]
                * input[..., 0].unsqueeze(-1)
            )
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(
                self.boundary_fn(state).unsqueeze(dim=-1), state
            )[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (
                self.value_var
                / self.value_normto
                / self.state_var.to(device=dodi.device)
            ) * dodi[..., 1:]

        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # convert model io to real dv
    def io_to_2nd_derivative(self, input, output):
        hes = diff_operators.batchHessian(output.unsqueeze(dim=-1), input)[0].squeeze(
            dim=-2
        )

        if self.deepReach_model == "diff":
            vis_term1 = (
                self.value_var
                / self.value_normto
                / self.state_var.to(device=hes.device)
            ) ** 2 * hes[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            vis_term2 = diff_operators.batchHessian(
                self.boundary_fn(state).unsqueeze(dim=-1), state
            )[0].squeeze(dim=-2)
            hes = vis_term1 + vis_term2

        else:
            hes = (
                self.value_var
                / self.value_normto
                / self.state_var.to(device=hes.device)
            ) ** 2 * hes[..., 1:]

        return hes

    def clamp_control(self, state, control):
        return control

    def bound_control(self, control):
        return torch.clamp(
            control, self.control_range_[..., 0], self.control_range_[..., 1]
        )

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def clamp_disturbance(self, state, disturbance):
        return disturbance

    def bound_disturbance(self, disturbance):
        return disturbance

    def clamp_state_input(self, state_input):
        return state_input

    def clamp_verification_state(self, state):
        return state

    # ALL FOLLOWING METHODS USE REAL UNITS
    @abstractmethod
    def periodic_transform_fn(self, input):
        raise NotImplementedError

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError


class VertDrone2D(Dynamics):
    def __init__(self):
        self.gravity = 9.8  # g
        self.input_multiplier = 12.0  # K
        self.input_magnitude_max = 1.0  # u_max
        self.state_range_ = torch.tensor([[-4, 4], [-0.5, 3.5]]).to(device)  # v, z, k
        self.control_range_ = torch.tensor(
            [[-self.input_magnitude_max, self.input_magnitude_max]]
        ).to(device)
        self.eps_var = torch.tensor([2]).to(device)
        self.control_init = (
            torch.ones(1).to(device) * self.gravity / self.input_multiplier
        )

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name="VertDrone2D",
            loss_type="brt_hjivi",
            set_mode="avoid",
            state_dim=2,
            input_dim=3,  # input_dim of the NN = state_dim + 1 (time dim)
            control_dim=1,
            disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,  # we estimate the ground-truth value function to be within [-0.5, 1.5] w.r.t. the state_range_ we used
            value_var=1,  # Then value_mean = 0.5*(-0.5 + 1.5) and value_max = 0.5*(1.5 - -0.5)
            value_normto=0.02,  # Don't need any changes
            deepReach_model="exact",  # chioce ['vanilla', 'exact'],
        )

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()
        # Here we verify the training results using the training range itself, we can verify on a smaller range for "stiff" systems

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def periodic_transform_fn(self, input):
        return input.to(device)

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.input_multiplier * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return (
            -torch.abs(state[..., 1] - 1.5) + 1.5
        )  # distance to ground (0m) and ceiling (3m)

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return (
            torch.abs(self.input_multiplier * dvds[..., 0]) * self.input_magnitude_max
            - dvds[..., 0] * self.gravity
            + dvds[..., 1] * state[..., 0]
        )

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            "state_slices": [0, 1.5],
            "state_labels": ["v", "z"],
            "x_axis_idx": 0,  # which dim you want it to be the
            "y_axis_idx": 1,
            "z_axis_idx": -1,  # because there is only 2D
        }


class ParameterizedVertDrone2D(Dynamics):
    def __init__(
        self, gravity: float, input_multiplier: float, input_magnitude_max: float
    ):
        self.gravity = gravity  # g
        self.input_multiplier = input_multiplier  # k_max
        self.input_magnitude_max = input_magnitude_max  # u_max
        self.state_range_ = torch.tensor(
            [[-4, 4], [-0.5, 3.5], [0, self.input_multiplier]]
        ).to(
            device
        )  # v, z, k
        self.control_range_ = torch.tensor(
            [[-self.input_magnitude_max, self.input_magnitude_max]]
        ).to(device)
        self.eps_var = torch.tensor([2]).to(device)
        self.control_init = torch.ones(1).to(device) * gravity / input_multiplier

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name="ParameterizedVertDrone2D",
            loss_type="brt_hjivi",
            set_mode="avoid",
            state_dim=3,
            input_dim=4,
            control_dim=1,
            disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model="exact",  # chioce ['vanilla', 'exact'],
        )

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def periodic_transform_fn(self, input):
        return input.to(device)

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        dsdt[..., 2] = 0
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return (
            torch.abs(state[..., 2] * dvds[..., 0]) * self.input_magnitude_max
            - dvds[..., 0] * self.gravity
            + dvds[..., 1] * state[..., 0]
        )

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            "state_slices": [0, 1.5, self.input_multiplier],
            "state_labels": ["v", "z", "k"],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 2,
        }


class Dubins3D(Dynamics):
    def __init__(self, set_mode: str):
        self.goalR = 0.5
        self.velocity = 1.0
        self.omega_max = 1.2
        self.state_range_ = torch.tensor([[-1, 1], [-1, 1], [-math.pi, math.pi]]).to(
            device
        )
        self.control_range_ = torch.tensor([[-self.omega_max, self.omega_max]]).to(
            device
        )
        self.eps_var = torch.tensor([1]).to(device)
        self.control_init = torch.zeros(1).to(device)
        self.set_mode = set_mode

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0
        super().__init__(
            name="Dubins3D",
            loss_type="brt_hjivi",
            set_mode=set_mode,
            state_dim=3,
            input_dim=5,
            control_dim=1,
            disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model="exact",
        )

    def control_range(self, state):
        return [[-self.omega_max, self.omega_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (
            2 * math.pi
        ) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] + 1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3] = torch.sin(input[..., 3] * self.state_var[-1])
        transformed_input[..., 4] = torch.cos(input[..., 3] * self.state_var[-1])
        return transformed_input.to(device)

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - 0.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == "avoid":
            return self.velocity * (
                torch.cos(state[..., 2]) * dvds[..., 0]
                + torch.sin(state[..., 2]) * dvds[..., 1]
            ) + self.omega_max * torch.abs(dvds[..., 2])
        elif self.set_mode == "reach":
            return self.velocity * (
                torch.cos(state[..., 2]) * dvds[..., 0]
                + torch.sin(state[..., 2]) * dvds[..., 1]
            ) - self.omega_max * torch.abs(dvds[..., 2])
        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        if self.set_mode == "avoid":
            return (self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == "reach":
            return -(self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        else:
            raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            "state_slices": [0, 0, 0],
            "state_labels": ["x", "y", r"$\theta$"],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 2,
        }


class Quadrotor(Dynamics):
    def __init__(
        self, collisionR: float, collective_thrust_max: float, set_mode: str
    ):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR
        self.reach_fn_weight = 1.0
        self.avoid_fn_weight = 0.3
        self.state_range_ = torch.tensor(
            [
                [-3.0, 3.0],
                [-3.0, 3.0],
                [-3.0, 3.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, 5.0],
            ]
        ).to(device)
        self.control_range_ = torch.tensor(
            [
                [-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max],
            ]
        ).to(device)
        self.eps_var = torch.tensor([20, 8, 8, 4]).to(device)
        self.control_init = torch.tensor([-self.Gz * 0.0, 0, 0, 0]).to(device)

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0
        if set_mode == "reach_avoid":
            l_type = "brat_hjivi"
        else:
            l_type = "brt_hjivi"
        super().__init__(
            name="Quadrotor",
            loss_type=l_type,
            set_mode=set_mode,
            state_dim=13,
            input_dim=14,
            control_dim=4,
            disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=(math.sqrt(3.0**2 + 3.0**2) - 2 * self.collisionR) / 2,
            value_var=math.sqrt(3.0**2 + 3.0**2) / 2,
            value_normto=0.02,
            deepReach_model="exact",
        )

    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x * 1.0
        q_tensor = x[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2, dim=-1
        )  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x

    def clamp_state_input(self, state_input):
        return self.normalize_q(state_input)

    def control_range(self, state):
        return [
            [-self.collective_thrust_max, self.collective_thrust_max],
            [-self.dwx_max, self.dwx_max],
            [-self.dwy_max, self.dwy_max],
            [-self.dwz_max, self.dwz_max],
        ]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def periodic_transform_fn(self, input):
        return input.to(device)

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / self.m * f
        dsdt[..., 9] = (
            self.Gz
            + (1 - 2 * torch.pow(qx, 2) - 2 * torch.pow(qy, 2)) * self.CT / self.m * f
        )
        dsdt[..., 10] = (control[..., 1]) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]) * 1.0

        return dsdt

    def dist_to_cylinder(self, state, a, b):
        """for cylinder with full body collision"""
        state_ = state * 1.0
        state_[..., 0] = state_[..., 0] - a
        state_[..., 1] = state_[..., 1] - b

        # create normal vector
        v = torch.zeros_like(state_[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state_[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state_[..., 0]
        py = state_[..., 1]

        # get full body distance
        dist = torch.norm(state_[..., :2], dim=-1)
        # return dist- self.collisionR
        dist = dist - torch.sqrt(
            (self.arm_l**2 * px**2 * vz**2)
            / (
                px**2 * vx**2
                + px**2 * vz**2
                + 2 * px * py * vx * vy
                + py**2 * vy**2
                + py**2 * vz**2
            )
            + (self.arm_l**2 * py**2 * vz**2)
            / (
                px**2 * vx**2
                + px**2 * vz**2
                + 2 * px * py * vx * vy
                + py**2 * vy**2
                + py**2 * vz**2
            )
        )
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR

    def reach_fn(self, state):
        state_ = state * 1.0
        state_[..., 0] = state_[..., 0] - 0.0
        state_[..., 1] = state_[..., 1]
        return (torch.norm(state[..., :2], dim=-1) - 0.3) * self.reach_fn_weight

    def avoid_fn(self, state):
        return self.avoid_fn_weight * torch.minimum(
            self.dist_to_cylinder(state, 0.0, 0.75),
            self.dist_to_cylinder(state, 0.0, -0.75),
        )

    def boundary_fn(self, state):
        if self.set_mode == "avoid":
            return self.dist_to_cylinder(state, 0.0, 0.0)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-1, 1]
        target_state_range[1] = [-0.25, 0.25]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim) * (
            target_state_range[:, 1] - target_state_range[:, 0]
        )

    def cost_fn(self, state_traj):
        if self.set_mode == "avoid":
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        else:
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(
                torch.clamp(
                    reach_values,
                    min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1),
                ),
                dim=-1,
            ).values

    def hamiltonian(self, state, dvds):
        if self.set_mode in ["reach", "reach_avoid"]:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 * torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += (
                -dvds[..., 10] * 5 * wy * wz / 9.0 + dvds[..., 11] * 5 * wx * wz / 9.0
            )

            ham -= (
                torch.abs(dvds[..., 7] * c1 + dvds[..., 8] * c2 + dvds[..., 9] * c3)
                * self.collective_thrust_max
            )

            ham -= (
                torch.abs(dvds[..., 10]) * self.dwx_max
                + torch.abs(dvds[..., 11]) * self.dwy_max
                + torch.abs(dvds[..., 12]) * self.dwz_max
            )

        elif self.set_mode == "avoid":
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 * torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += (
                -dvds[..., 10] * 5 * wy * wz / 9.0 + dvds[..., 11] * 5 * wx * wz / 9.0
            )

            ham += (
                torch.abs(dvds[..., 7] * c1 + dvds[..., 8] * c2 + dvds[..., 9] * c3)
                * self.collective_thrust_max
            )

            ham += (
                torch.abs(dvds[..., 10]) * self.dwx_max
                + torch.abs(dvds[..., 11]) * self.dwy_max
                + torch.abs(dvds[..., 12]) * self.dwz_max
            )

        else:
            raise NotImplementedError

        return ham

    def optimal_control(self, state, dvds):
        if self.set_mode in ["reach", "reach_avoid"]:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 * torch.pow(qy, 2)) * self.CT / self.m

            u1 = -self.collective_thrust_max * torch.sign(
                dvds[..., 7] * c1 + dvds[..., 8] * c2 + dvds[..., 9] * c3
            )
            u2 = -self.dwx_max * torch.sign(dvds[..., 10])
            u3 = -self.dwy_max * torch.sign(dvds[..., 11])
            u4 = -self.dwz_max * torch.sign(dvds[..., 12])
        elif self.set_mode == "avoid":
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 * torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * torch.sign(
                dvds[..., 7] * c1 + dvds[..., 8] * c2 + dvds[..., 9] * c3
            )
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat(
            (u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1
        )

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)

    def plot_config(self):
        return {
            "state_slices": [
                0.96,
                1.18,
                0.54,
                0.44,
                -0.45,
                0.27,
                -0.73,
                -2.83,
                -1.07,
                -3.34,
                3.19,
                -2.80,
                3.43,
            ],
            "state_labels": [
                "x",
                "y",
                "z",
                "qw",
                "qx",
                "qy",
                "qz",
                "vx",
                "vy",
                "vz",
                "wx",
                "wy",
                "wz",
            ],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 7,
        }


class F1tenth(Dynamics):
    def __init__(self):
        # variable for dynamics
        self.mu = 1.0489
        self.C_Sf = 4.718
        self.C_Sr = 5.4562
        self.lf = 0.15875
        self.lr = 0.17145
        self.h = 0.074
        self.m = 3.74
        self.I = 0.04712
        self.s_min = -0.4189
        self.s_max = 0.4189
        self.sv_min = -3.2
        self.sv_max = 3.2
        self.v_switch = 7.319
        self.a_max = 9.51
        self.v_min = 0.1
        self.v_max = 10.0
        self.omega_max = 6.0
        self.delta_t = 0.01
        self.g = 9.81
        self.lwb = self.lf + self.lr

        self.v_mean = (self.v_min + self.v_max) / 2
        self.v_var = (self.v_max - self.v_min) / 2

        # map info
        # self.dt = np.load(map_path)
        self.origin = [-78.21853769831466, -44.37590462453829]
        self.resolution = 0.062500
        self.width = 1600
        self.height = 1600

        # control constraints
        self.input_steering_v_max = self.sv_max
        self.input_acceleration_max = self.a_max

        self.xmean = 62.5 / 2
        self.xvar = 62.5 / 2
        self.ymean = 25
        self.yvar = 25

        self.x_min = self.xmean - self.xvar
        self.x_max = self.xmean + self.xvar
        self.y_min = self.ymean - self.yvar
        self.y_max = self.ymean + self.yvar

        self.state_range_ = torch.tensor(
            [
                [self.x_min, self.x_max],
                [self.y_min, self.y_max],
                [-0.4189, 0.4189],
                [self.v_min, self.v_max],
                [-math.pi, math.pi],
                [-self.omega_max, self.omega_max],
                [-1, 1],
            ]
        ).to(device)
        self.control_range_ = torch.tensor(
            [[self.sv_min, self.sv_max], [-self.a_max, self.a_max]]
        ).to(device)
        self.eps_var = torch.tensor([self.sv_max**2, self.a_max**2]).to(device)
        self.control_init = torch.tensor([0.0, 0.0]).to(device)

        # for the track
        self.obstaclemap_file = "dynamics/F1_map_obstaclemap.mat"
        self.pixel2world = 0.0625
        self.obstacle_map = spio.loadmat(self.obstaclemap_file)
        self.obstacle_map = self.obstacle_map["obs_map"]
        self.obstacle_map[self.obstacle_map == -0.0] = 1
        self.obstacle_map = self.obstacle_map[
            int(self.y_min / self.pixel2world) : int(self.y_max / self.pixel2world) + 1,
            int(self.x_min / self.pixel2world) : int(self.x_max / self.pixel2world) + 1,
        ]
        self.obstacle_map = torch.tensor(self.obstacle_map)

        self.world_range = self.state_range_.cpu().numpy()
        self.x_rangearray = torch.arange(self.obstacle_map.shape[0])
        self.y_rangearray = torch.arange(self.obstacle_map.shape[1])

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name="F1tenth",
            loss_type="brt_hjivi",
            set_mode="avoid",
            state_dim=7,
            input_dim=9,
            control_dim=2,
            disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,  # mean of expected value function
            value_var=1.5,  # (max - min)/2.0 of expected value function
            value_normto=0.02,
            deepReach_model="exact",
        )

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return [
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],  # y
            [-0.4189, 0.4189],  # steering angle
            [0.1, 8.0],  # velocity
            [-math.pi, math.pi],  # pose theta
            [-4.5, 4.5],  # pose theta rate
            [-0.8, 0.8],  # slip angle
        ]

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] + 1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :5] = input[..., :5]
        transformed_input[..., 5] = torch.sin(input[..., 5] * self.state_var[4])
        transformed_input[..., 6] = torch.cos(input[..., 5] * self.state_var[4])
        transformed_input[..., 7:] = input[..., 6:]
        return transformed_input.to(device)

    def dsdt(self, state, control, disturbance):
        # here the control is steering angle v and acceleration
        f = torch.zeros_like(state)
        current_vel = state[..., 3]  # [1, 65000]
        kinematic_mask = torch.abs(current_vel) < 0.5
        # switch to kinematic model for small velocities
        if torch.any(kinematic_mask):
            # print(f"kinematic_mask is {kinematic_mask.shape}")
            if len(kinematic_mask.shape) == 1:
                sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3] * torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3] * torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3] / self.lwb * torch.tan(x_ks[..., 2])
                f[sample_idx, :5] = f_ks
                f[sample_idx, 5] = (
                    u_ks[..., 1] / self.lwb * torch.tan(state[kinematic_mask][..., 2])
                    + state[kinematic_mask][..., 3]
                    / (self.lwb * torch.cos(state[kinematic_mask][..., 2]) ** 2)
                    * u_ks[..., 0]
                )
                f[sample_idx, 6] = 0.0
            else:
                batch_idx, sample_idx = kinematic_mask.nonzero(as_tuple=True)
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3] * torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3] * torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3] / self.lwb * torch.tan(x_ks[..., 2])
                f[batch_idx, sample_idx, :5] = f_ks
                f[batch_idx, sample_idx, 5] = (
                    u_ks[..., 1] / self.lwb * torch.tan(state[kinematic_mask][..., 2])
                    + state[kinematic_mask][..., 3]
                    / (self.lwb * torch.cos(state[kinematic_mask][..., 2]) ** 2)
                    * u_ks[..., 0]
                )
                f[batch_idx, sample_idx, 6] = 0.0

        dynamic_mask = ~kinematic_mask
        if torch.any(dynamic_mask):
            if len(kinematic_mask.shape) == 1:
                sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                f[sample_idx, 0] = state[dynamic_mask][..., 3] * torch.cos(
                    state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4]
                )
                f[sample_idx, 1] = state[dynamic_mask][..., 3] * torch.sin(
                    state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4]
                )
                f[sample_idx, 2] = control[dynamic_mask][..., 0]
                f[sample_idx, 3] = control[dynamic_mask][..., 1]
                f[sample_idx, 4] = state[dynamic_mask][..., 5]
                f[sample_idx, 5] = (
                    -self.mu
                    * self.m
                    / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))
                    * (
                        self.lf**2
                        * self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                        + self.lr**2
                        * self.C_Sr
                        * (self.g * self.lf + control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 5]
                    + self.mu
                    * self.m
                    / (self.I * (self.lr + self.lf))
                    * (
                        self.lr
                        * self.C_Sr
                        * (self.g * self.lf + control[dynamic_mask][..., 1] * self.h)
                        - self.lf
                        * self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 6]
                    + self.mu
                    * self.m
                    / (self.I * (self.lr + self.lf))
                    * self.lf
                    * self.C_Sf
                    * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    * state[dynamic_mask][..., 2]
                )
                f[sample_idx, 6] = (
                    (
                        self.mu
                        / (state[dynamic_mask][..., 3] ** 2 * (self.lr + self.lf))
                        * (
                            self.C_Sr
                            * (
                                self.g * self.lf
                                + control[dynamic_mask][..., 1] * self.h
                            )
                            * self.lr
                            - self.C_Sf
                            * (
                                self.g * self.lr
                                - control[dynamic_mask][..., 1] * self.h
                            )
                            * self.lf
                        )
                        - 1
                    )
                    * state[dynamic_mask][..., 5]
                    - self.mu
                    / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                    * (
                        self.C_Sr
                        * (self.g * self.lf + control[dynamic_mask][..., 1] * self.h)
                        + self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 6]
                    + self.mu
                    / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                    * (
                        self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 2]
                )
            else:
                batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)
                f[batch_idx, sample_idx, 0] = state[dynamic_mask][..., 3] * torch.cos(
                    state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4]
                )
                f[batch_idx, sample_idx, 1] = state[dynamic_mask][..., 3] * torch.sin(
                    state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4]
                )
                f[batch_idx, sample_idx, 2] = control[dynamic_mask][..., 0]
                f[batch_idx, sample_idx, 3] = control[dynamic_mask][..., 1]
                f[batch_idx, sample_idx, 4] = state[dynamic_mask][..., 5]
                f[batch_idx, sample_idx, 5] = (
                    -self.mu
                    * self.m
                    / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))
                    * (
                        self.lf**2
                        * self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                        + self.lr**2
                        * self.C_Sr
                        * (self.g * self.lf + control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 5]
                    + self.mu
                    * self.m
                    / (self.I * (self.lr + self.lf))
                    * (
                        self.lr
                        * self.C_Sr
                        * (self.g * self.lf + control[dynamic_mask][..., 1] * self.h)
                        - self.lf
                        * self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 6]
                    + self.mu
                    * self.m
                    / (self.I * (self.lr + self.lf))
                    * self.lf
                    * self.C_Sf
                    * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    * state[dynamic_mask][..., 2]
                )
                f[batch_idx, sample_idx, 6] = (
                    (
                        self.mu
                        / (state[dynamic_mask][..., 3] ** 2 * (self.lr + self.lf))
                        * (
                            self.C_Sr
                            * (
                                self.g * self.lf
                                + control[dynamic_mask][..., 1] * self.h
                            )
                            * self.lr
                            - self.C_Sf
                            * (
                                self.g * self.lr
                                - control[dynamic_mask][..., 1] * self.h
                            )
                            * self.lf
                        )
                        - 1
                    )
                    * state[dynamic_mask][..., 5]
                    - self.mu
                    / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                    * (
                        self.C_Sr
                        * (self.g * self.lf + control[dynamic_mask][..., 1] * self.h)
                        + self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 6]
                    + self.mu
                    / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                    * (
                        self.C_Sf
                        * (self.g * self.lr - control[dynamic_mask][..., 1] * self.h)
                    )
                    * state[dynamic_mask][..., 2]
                )
        # ------------------------------OPT CTRL--------------------------------

        return f

    def clamp_state_input(self, state_input):
        full_input = torch.cat(
            (torch.ones(state_input.shape[0], 1).to(state_input), state_input), dim=-1
        )
        state = self.input_to_coord(full_input)[..., 1:]
        lx = self.boundary_fn(state)
        return state_input[lx >= -1]

    def clamp_verification_state(self, state):
        lx = self.boundary_fn(state)
        return state[lx >= 0]

    def clamp_control(self, state, control):
        control_clamped = control * 1.0

        smax_mask = state[..., 2] > self.s_max - 0.01
        smin_mask = state[..., 2] < -self.s_max + 0.01
        if len(smax_mask.shape) == 1:
            max_sample_idx = smax_mask.nonzero(as_tuple=True)[0]
            control_clamped[max_sample_idx, 0] = torch.clamp(
                control_clamped[max_sample_idx, 0], max=0
            )
            min_sample_idx = smin_mask.nonzero(as_tuple=True)[0]
            control_clamped[min_sample_idx, 0] = torch.clamp(
                control_clamped[min_sample_idx, 0], min=0
            )

        else:
            max_batch_idx, max_sample_idx = smax_mask.nonzero(as_tuple=True)
            control_clamped[max_batch_idx, max_sample_idx, 0] = torch.clamp(
                control_clamped[max_batch_idx, max_sample_idx, 0], max=0
            )
            min_batch_idx, min_sample_idx = smin_mask.nonzero(as_tuple=True)
            control_clamped[min_batch_idx, min_sample_idx, 0] = torch.clamp(
                control_clamped[min_batch_idx, min_sample_idx, 0], min=0
            )

        accelerate_upper = (
            torch.ones(state.shape[:-1], device=state.device)
            * self.input_acceleration_max
        )
        accelerate_upper[state[..., 3] > self.v_switch] = (
            self.input_acceleration_max
            * self.v_switch
            / state[state[..., 3] > self.v_switch][..., 3]
        )

        acc_mask = control_clamped[..., 1] > accelerate_upper
        if len(acc_mask.shape) == 1:
            sample_idx = acc_mask.nonzero(as_tuple=True)[0]
            control_clamped[sample_idx, 1] = accelerate_upper[sample_idx]
        else:
            batch_idx, sample_idx = acc_mask.nonzero(as_tuple=True)
            control_clamped[batch_idx, sample_idx, 1] = accelerate_upper[
                batch_idx, sample_idx
            ]

        assert ((accelerate_upper - control_clamped[..., 1]) >= 0.0).all()
        return control_clamped

    def interpolation(self, state_pixel_coords):
        self.obstacle_map = self.obstacle_map.to(state_pixel_coords)
        # Find the indices surrounding the query points
        x0 = torch.floor(state_pixel_coords[..., 0]).long()
        x1 = x0 + 1
        y0 = torch.floor(state_pixel_coords[..., 1]).long()
        y1 = y0 + 1
        # Ensure indices are within bounds
        x0 = torch.clamp(x0, 0, self.x_rangearray.size(0) - 1)
        x1 = torch.clamp(x1, 0, self.x_rangearray.size(0) - 1)
        y0 = torch.clamp(y0, 0, self.y_rangearray.size(0) - 1)
        y1 = torch.clamp(y1, 0, self.y_rangearray.size(0) - 1)

        # Gather the values at the corner points for each query point
        v00 = self.obstacle_map[x0, y0]
        v01 = self.obstacle_map[x0, y1]
        v10 = self.obstacle_map[x1, y0]
        v11 = self.obstacle_map[x1, y1]
        # Compute the fractional part for each query point
        x_frac = state_pixel_coords[..., 0] - x0.float()
        y_frac = state_pixel_coords[..., 1] - y0.float()
        # Bilinear interpolation for each query point
        v0 = v00 * (1 - x_frac) + v10 * x_frac
        v1 = v01 * (1 - x_frac) + v11 * x_frac
        # Interpolated value
        interp_values = v0 * (1 - y_frac) + v1 * y_frac
        return interp_values

    def boundary_fn(self, state):
        # MPC: state = B * N * H * 7
        # DeepReach: state = B * 7
        # Takes the cordinates in the real world and returns the lx for the obstacles at those coords
        # shift the origin so that the min is 0
        shiftedCoords = state - torch.tensor(
            self.world_range[..., 0].reshape(
                self.state_dim,
            ),
            device=state.device,
        )  # num states involve time as well
        # extract and flip the x and y pos for image space query
        if shiftedCoords.shape[0] == 1:
            shiftedCoords_pos_world = np.squeeze(shiftedCoords[..., 0:2])
        else:
            shiftedCoords_pos_world = shiftedCoords[..., 0:2]
        if len(shiftedCoords_pos_world.shape) == 2:
            shiftedCoords_pos_image = torch.fliplr(
                shiftedCoords_pos_world
            )  # B*2 for deepreach, B*N*H*2 for MPC
        else:
            shiftedCoords_pos_image = torch.flip(shiftedCoords_pos_world, [-1])
        # convert the world coordinates to pixel coordinates
        shiftedCoords_pos_pixel = (
            shiftedCoords_pos_image / self.pixel2world
        )  # note this does not have to be integers due to the regularGridInterpolator
        # query the generator
        obstacle_value = self.interpolation(
            shiftedCoords_pos_pixel
        )  # obstacle value only depends on pos
        # obstacle_value = obstacle_value.reshape([obstacle_value.shape[0],1]) # this should be the lx
        return obstacle_value

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == "reach":
            raise NotImplementedError

        elif self.set_mode == "avoid":
            opt_control = self.optimal_control(state, dvds)
            dsdt_ = self.dsdt(state, opt_control, None)
            ham = torch.sum(dvds * dsdt_, dim=-1)
        return ham

    def optimal_control(self, state, dvds):

        if self.set_mode == "reach":
            raise NotImplementedError
        elif self.set_mode == "avoid":
            unsqueeze_u = False
            if state.shape[0] == 1:
                state = state.squeeze(0)
                dvds = dvds.squeeze(0)
                unsqueeze_u = True
            batch_dims = state.shape[:-1]
            u = torch.zeros(*batch_dims, 2, device=state.device)

            kinematic_mask = torch.abs(state[..., 3]) < 0.5
            # if the number of kinematic_mask's dimensional is greater than 1, print

            if torch.any(kinematic_mask):
                if len(kinematic_mask.shape) == 1:
                    sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * torch.sign(
                        dvds[kinematic_mask][..., 2]
                        + dvds[kinematic_mask][..., 5]
                        * state[kinematic_mask][..., 3]
                        / (self.lwb * torch.cos(state[kinematic_mask][..., 2]) ** 2)
                    )
                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[kinematic_mask][..., 3]
                        + dvds[kinematic_mask][..., 5]
                        / self.lwb
                        * torch.tan(state[kinematic_mask][..., 2])
                    )
                else:
                    batch_idx, sample_idx = kinematic_mask.nonzero(as_tuple=True)
                    u[batch_idx, sample_idx, 0] = (
                        self.input_steering_v_max
                        * torch.sign(
                            dvds[kinematic_mask][..., 2]
                            + dvds[kinematic_mask][..., 5]
                            * state[kinematic_mask][..., 3]
                            / (self.lwb * torch.cos(state[kinematic_mask][..., 2]) ** 2)
                        )
                    )
                    u[batch_idx, sample_idx, 1] = (
                        self.input_acceleration_max
                        * torch.sign(
                            dvds[kinematic_mask][..., 3]
                            + dvds[kinematic_mask][..., 5]
                            / self.lwb
                            * torch.tan(state[kinematic_mask][..., 2])
                        )
                    )

            dynamic_mask = ~kinematic_mask
            if torch.any(dynamic_mask):
                if len(kinematic_mask.shape) == 1:
                    sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * torch.sign(
                        dvds[dynamic_mask][..., 2]
                    )

                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[dynamic_mask][..., 3]
                        + dvds[dynamic_mask][..., 5]
                        * (
                            (
                                -self.mu
                                * self.m
                                / (
                                    state[dynamic_mask][..., 3]
                                    * self.I
                                    * (self.lr + self.lf)
                                )
                                * (
                                    -self.lf**2 * self.C_Sf * self.h
                                    + self.lr**2 * self.C_Sr * self.h
                                )
                            )
                            * state[dynamic_mask][..., 5]
                            + self.mu
                            * self.m
                            / (self.I * (self.lr + self.lf))
                            * (
                                self.lr * self.C_Sr * self.h
                                + self.lf * self.C_Sf * self.h
                            )
                            * state[dynamic_mask][..., 6]
                            - self.mu
                            * self.m
                            / (self.I * (self.lr + self.lf))
                            * self.lf
                            * self.C_Sf
                            * self.h
                            * state[dynamic_mask][..., 2]
                        )
                        + dvds[dynamic_mask][..., 6]
                        * (
                            (
                                self.mu
                                / (
                                    state[dynamic_mask][..., 3] ** 2
                                    * (self.lr + self.lf)
                                )
                                * (
                                    self.C_Sr * self.h * self.lr
                                    + self.C_Sf * self.h * self.lf
                                )
                            )
                            * state[dynamic_mask][..., 5]
                            - self.mu
                            / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                            * (self.C_Sr * self.h - self.C_Sf * self.h)
                            * state[dynamic_mask][..., 6]
                            - self.mu
                            / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                            * self.C_Sf
                            * self.h
                            * state[dynamic_mask][..., 2]
                        )
                    )
                else:
                    batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)

                    u[batch_idx, sample_idx, 0] = (
                        self.input_steering_v_max
                        * torch.sign(dvds[dynamic_mask][..., 2])
                    )

                    u[batch_idx, sample_idx, 1] = (
                        self.input_acceleration_max
                        * torch.sign(
                            dvds[dynamic_mask][..., 3]
                            + dvds[dynamic_mask][..., 5]
                            * (
                                (
                                    -self.mu
                                    * self.m
                                    / (
                                        state[dynamic_mask][..., 3]
                                        * self.I
                                        * (self.lr + self.lf)
                                    )
                                    * (
                                        -self.lf**2 * self.C_Sf * self.h
                                        + self.lr**2 * self.C_Sr * self.h
                                    )
                                )
                                * state[dynamic_mask][..., 5]
                                + self.mu
                                * self.m
                                / (self.I * (self.lr + self.lf))
                                * (
                                    self.lr * self.C_Sr * self.h
                                    + self.lf * self.C_Sf * self.h
                                )
                                * state[dynamic_mask][..., 6]
                                - self.mu
                                * self.m
                                / (self.I * (self.lr + self.lf))
                                * self.lf
                                * self.C_Sf
                                * self.h
                                * state[dynamic_mask][..., 2]
                            )
                            + dvds[dynamic_mask][..., 6]
                            * (
                                (
                                    self.mu
                                    / (
                                        state[dynamic_mask][..., 3] ** 2
                                        * (self.lr + self.lf)
                                    )
                                    * (
                                        self.C_Sr * self.h * self.lr
                                        + self.C_Sf * self.h * self.lf
                                    )
                                )
                                * state[dynamic_mask][..., 5]
                                - self.mu
                                / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                                * (self.C_Sr * self.h - self.C_Sf * self.h)
                                * state[dynamic_mask][..., 6]
                                - self.mu
                                / (state[dynamic_mask][..., 3] * (self.lr + self.lf))
                                * self.C_Sf
                                * self.h
                                * state[dynamic_mask][..., 2]
                            )
                        )
                    )
            u = self.clamp_control(state, u)
            if unsqueeze_u:
                u = u[None, ...]
        return u

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 4] = (wrapped_state[..., 4] + math.pi) % (
            2 * math.pi
        ) - math.pi
        return wrapped_state

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            "state_slices": [0, 0, 0, 8.0, 0, 0, 0],
            "state_labels": [
                "x",
                "y",
                "sangle",
                "v",
                "posetheta",
                "poserate",
                "slipangle",
            ],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 4,
        }


class LessLinearND(Dynamics):
    def __init__(self, N: int, gamma: float, mu: float, alpha: float, goalR: float):
        u_max, set_mode = 0.5, "reach"  # TODO: unfix

        self.N = N
        self.u_max = u_max
        self.input_center = torch.zeros(N - 1)
        self.input_shape = "box"
        self.game = set_mode

        self.A = (
            -0.5 * torch.eye(N)
            - torch.cat(
                (
                    torch.cat((torch.zeros(1, 1), torch.ones(N - 1, 1)), 0),
                    torch.zeros(N, N - 1),
                ),
                1,
            )
        ).to(device)
        self.B = torch.cat((torch.zeros(1, N - 1), 0.4 * torch.eye(N - 1)), 0).to(
            device
        )
        self.Bumax = u_max * torch.matmul(
            self.B, torch.ones(self.N - 1).to(device)
        ).unsqueeze(0).unsqueeze(0).to(device)
        self.C = torch.cat((torch.zeros(1, N - 1), 0.1 * torch.eye(N - 1)), 0)
        self.gamma, self.mu, self.alpha = gamma, mu, alpha
        self.gamma_orig, self.mu_orig, self.alpha_orig = gamma, mu, alpha

        self.goalR_2d = goalR
        self.goalR = (
            (N - 1) ** 0.5
        ) * self.goalR_2d  # accounts for N-dimensional combination
        self.ellipse_params = torch.cat(
            (((N - 1) ** 0.5) * torch.ones(1), torch.ones(N - 1) / 1.0), 0
        )  # accounts for N-dimensional combination

        self.state_range_ = torch.tensor([[-1, 1] for _ in range(self.N)]).to(device)
        self.control_range_ = torch.tensor(
            [[-u_max, u_max] for _ in range(self.N - 1)]
        ).to(device)
        self.eps_var = torch.tensor([u_max for _ in range(self.N - 1)]).to(device)
        self.control_init = torch.tensor([0.0 for _ in range(self.N - 1)]).to(device)

        super().__init__(
            name="50D system",
            loss_type="brt_hjivi",
            set_mode=set_mode,
            state_dim=N,
            input_dim=N + 1,
            control_dim=N - 1,
            disturbance_dim=N - 1,
            state_mean=[0 for _ in range(N)],
            state_var=[1 for _ in range(N)],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model="exact",
        )

    def vary_nonlinearity(self, epsilon):
        self.gamma = epsilon * self.gamma_orig
        self.mu = epsilon * self.mu_orig
        # self.alpha = epsilon * self.alpha_orig #shouldn't be varied since its not a scalar (1-\lambda) l(\cdot) +  \lambda f(\cdot)

    def state_test_range(self):
        return [[-1, 1] for _ in range(self.N)]

    def state_verification_range(self):
        return [[-1, 1] for _ in range(self.N)]

    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # LessLinear dynamics
    # \dot xN    = (aN \cdot x) + (no ctrl or dist) + mu * sin(alpha * xN) * xN^2
    # \dot xi    = (ai \cdot x) + bi * ui + ci * di - gamma * xi * xN^2
    # i.e.
    # \dot x = Ax + Bu + Cd + NLterm(x, gamma, mu, alpha)
    # def dsdt(self, state, control, disturbance):
    #     dsdt = torch.zeros_like(state)
    #     nl_term_N = self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]
    #     nl_term_i = torch.multiply(-self.gamma * state[..., 0] * state[..., 0], state[..., 1:])
    #     dsdt[..., :] = torch.matmul(self.A, state[..., :]) + torch.matmul(self.B, control[..., :]) + torch.cat((nl_term_N, nl_term_i), 0)
    #     return dsdt
    def dsdt(self, state, control, disturbance):
        x0 = state[..., 0]  # shape: (...)
        x_rest = state[..., 1:]  # shape: (..., n-1)

        # Nonlinear terms
        nl_term_N = self.mu * torch.sin(self.alpha * x0) * x0 * x0  # shape: (...)
        nl_term_N = nl_term_N.unsqueeze(-1)  # shape: (..., 1)

        x0_squared = (x0**2).unsqueeze(-1)  # shape: (..., 1)
        nl_term_i = -self.gamma * x0_squared * x_rest  # broadcasted: (..., n-1)

        nl_term = torch.cat([nl_term_N, nl_term_i], dim=-1)  # shape: (..., n)

        # Linear terms
        linear_term = torch.matmul(state, self.A.T) + torch.matmul(control, self.B.T)

        return linear_term + nl_term

    def periodic_transform_fn(self, input):
        return input.to(device)

    def boundary_fn(self, state):
        if (
            self.ellipse_params.device != state.device
        ):  # FIXME: Patch to cover de/attached state bug
            if state.device.type == "cuda":
                self.ellipse_params = self.ellipse_params.to(device)
            else:
                self.ellipse_params = self.ellipse_params.cpu()
        return 0.5 * (
            torch.square(torch.norm(self.ellipse_params * state[..., :], dim=-1))
            - (self.goalR**2)
        )
        # return 0.5 * (torch.square(torch.norm(torch.cat((((self.N-1)**0.5)*torch.ones(1),torch.ones(self.N-1)),0) * state[..., :], dim=-1)) - (self.goalR ** 2))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):

        nl_term_N = (
            self.mu
            * torch.sin(self.alpha * state[..., 0])
            * state[..., 0]
            * state[..., 0]
        ).unsqueeze(-1)
        nl_term_i = (-self.gamma * state[..., 0] * state[..., 0]).t() * state[..., 1:]
        pAx = (
            dvds
            * (torch.matmul(state, self.A.t()) + torch.cat((nl_term_N, nl_term_i), 2))
        ).sum(2)
        pBumax = (torch.abs(dvds) * self.Bumax).sum(2)

        if self.set_mode == "reach":
            return pAx - pBumax
        elif self.set_mode == "avoid":
            return pAx + pBumax

    def optimal_control(self, state, dvds):
        if self.set_mode == "reach":
            return -self.u_max * torch.sign(dvds[..., 1:])
        elif self.set_mode == "avoid":
            return self.u_max * torch.sign(dvds[..., 1:])

    def optimal_disturbance(self, state, dvds):
        return 0.0

    def plot_config(self):  # FIXME
        return {
            "state_slices": [0 for _ in range(self.N)],
            "state_labels": ["xN"] + ["x" + str(i) for i in range(1, self.N)],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 2,
        }


class VertDrone2DWithDist(VertDrone2D):
    def __init__(self, max_disturbance: float):
        self.disturbance_magnitude_max = max_disturbance
        self.disturbance_range_ = torch.tensor(
            [-self.disturbance_magnitude_max, self.disturbance_magnitude_max]
        ).to(device)
        self.disturbance_init = torch.zeros(1).to(device)
        super().__init__()
        self.disturbance_dim = 1

    def disturbance_range(self, state):
        return [[-self.disturbance_magnitude_max, self.disturbance_magnitude_max]]

    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.input_multiplier * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0] + disturbance[..., 0]
        return dsdt

    def hamiltonian(self, state, dvds):
        return (
            torch.abs(self.input_multiplier * dvds[..., 0]) * self.input_magnitude_max
            - dvds[..., 0] * self.gravity
            + dvds[..., 1] * state[..., 0]
            - torch.abs(dvds[..., 1]) * self.disturbance_magnitude_max
        )

    def optimal_disturbance(self, state, dvds):
        return -(torch.sign(dvds[..., 1]) * self.disturbance_magnitude_max)[..., None]


class Air3D(Dynamics):
    def __init__(self, set_mode: str):
        self.goalR = 0.25
        self.evader_velocity = 0.6
        self.pursuer_velocity = 0.6
        self.omega_max = 2.0
        self.state_max = 1.5

        self.state_range_ = torch.tensor(
            [
                [-self.state_max, self.state_max],
                [-self.state_max, self.state_max],
                [-math.pi, math.pi],
            ]
        ).to(device)
        self.control_range_ = torch.tensor([[-self.omega_max, self.omega_max]]).to(
            device
        )
        self.disturbance_range_ = torch.tensor([[-self.omega_max, self.omega_max]]).to(
            device
        )
        self.control_init = torch.zeros(1).to(device)
        self.disturbance_init = torch.zeros(1).to(device)
        self.set_mode = set_mode
        self.eps_var = torch.tensor([1]).to(device)

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name="Air3D",
            loss_type="brt_hjivi",
            set_mode=set_mode,
            state_dim=3,
            input_dim=5,
            control_dim=1,
            disturbance_dim=1,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model="exact",
        )

    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()

    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def bound_control(self, control):
        return torch.clamp(
            control, self.control_range_[..., 0], self.control_range_[..., 1]
        )

    def clamp_control(self, state, control):
        return self.bound_control(control)

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (
            2 * math.pi
        ) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] + 1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3] = torch.sin(input[..., 3] * self.state_var[-1])
        transformed_input[..., 4] = torch.cos(input[..., 3] * self.state_var[-1])
        return transformed_input.to(device)

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        x1, x2, x3 = state[..., 0], state[..., 1], state[..., 2]
        a = control[..., 0]
        b = disturbance[..., 0]
        va = self.evader_velocity
        vb = self.pursuer_velocity
        dsdt[..., 0] = -va + vb * torch.cos(x3) + a * x2
        dsdt[..., 1] = vb * torch.sin(x3) - a * x1
        dsdt[..., 2] = b - a
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        x1, x2, x3 = state[..., 0], state[..., 1], state[..., 2]
        dVdx1, dVdx2, dVdx3 = dvds[..., 0], dvds[..., 1], dvds[..., 2]
        va, vb, w_max = self.evader_velocity, self.pursuer_velocity, self.omega_max

        base = -va * dVdx1 + vb * torch.cos(x3) * dVdx1 + vb * torch.sin(x3) * dVdx2
        a_terms = x2 * dVdx1 - x1 * dVdx2 - dVdx3
        b_term = dVdx3

        if self.set_mode == "avoid":
            return base + w_max * torch.abs(a_terms) - w_max * torch.abs(b_term)
        elif self.set_mode == "reach":
            return base - w_max * torch.abs(a_terms) + w_max * torch.abs(b_term)
        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        x1, x2 = state[..., 0], state[..., 1]
        dVdx1, dVdx2, dVdx3 = dvds[..., 0], dvds[..., 1], dvds[..., 2]
        term = x2 * dVdx1 - x1 * dVdx2 - dVdx3
        if self.set_mode == "avoid":
            return (self.omega_max * torch.sign(term))[..., None]
        elif self.set_mode == "reach":
            return -(self.omega_max * torch.sign(term))[..., None]
        else:
            raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        dVdx3 = dvds[..., 2]
        if self.set_mode == "avoid":
            return -(self.omega_max * torch.sign(dVdx3))[..., None]
        elif self.set_mode == "reach":
            return (self.omega_max * torch.sign(dVdx3))[..., None]
        else:
            raise NotImplementedError

    def plot_config(self):
        return {
            "state_slices": [0, 0, 0],
            "state_labels": ["x₁", "x₂", r"$x_3$"],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 2,
        }


class Dubins6D(Dynamics):
    def __init__(self, collisionR: float, set_mode: str, velocity: float = 0.6):
        self.evader_velocity = velocity
        self.pursuer_velocity = velocity
        self.omega_e_max = 1.9
        self.omega_p_max = 1.9
        self.x_state_max = 3.5
        self.y_state_max = 2.5
        self.goalR = collisionR

        self.state_range_ = torch.tensor(
            [
                [-self.x_state_max, self.x_state_max],
                [-self.y_state_max, self.y_state_max],
                [-math.pi, math.pi],
                [-self.x_state_max, self.x_state_max],
                [-self.y_state_max, self.y_state_max],
                [-math.pi, math.pi],
            ]
        ).to(device)
        self.control_range_ = torch.tensor([[-self.omega_e_max, self.omega_e_max]]).to(
            device
        )
        self.disturbance_range_ = torch.tensor(
            [[-self.omega_p_max, self.omega_p_max]]
        ).to(device)
        self.box_bounds = torch.tensor([[-3.0, 3.0], [-2.0, 2.0]]).to(device)
        self.control_init = torch.zeros(1).to(device)
        self.disturbance_init = torch.zeros(1).to(device)
        self.set_mode = set_mode
        self.eps_var = torch.tensor([2 * self.omega_e_max]).to(device)

        if set_mode in ["avoid", "reach"]:
            loss_type = "brt_hjivi"
        elif set_mode == "reach_avoid":
            loss_type = "brat_hjivi"
        else:
            raise NotImplementedError(f"Unknown set_mode: {set_mode}")

        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_ = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        super().__init__(
            name="Dubins6D",
            loss_type=loss_type,
            set_mode=set_mode,
            state_dim=6,
            input_dim=9,
            control_dim=1,
            disturbance_dim=1,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model="exact",
        )

    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def bound_control(self, control):
        return torch.clamp(
            control, self.control_range_[..., 0], self.control_range_[..., 1]
        )

    def clamp_control(self, state, control):
        return self.bound_control(control)

    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (
            2 * math.pi
        ) - math.pi
        wrapped_state[..., 5] = (wrapped_state[..., 5] + math.pi) % (
            2 * math.pi
        ) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = input.shape[-1] + 2
        transformed_input = torch.zeros(output_shape, device=input.device)
        transformed_input[..., 0] = input[..., 0]  # time
        transformed_input[..., 1] = input[..., 1]  # x_e
        transformed_input[..., 2] = input[..., 2]  # y_e
        theta_e = input[..., 3]
        transformed_input[..., 3] = torch.sin(theta_e * self.state_var[-1])
        transformed_input[..., 4] = torch.cos(theta_e * self.state_var[-1])
        transformed_input[..., 5] = input[..., 4]  # x_p
        transformed_input[..., 6] = input[..., 5]  # y_p
        theta_p = input[..., 6]
        transformed_input[..., 7] = torch.sin(theta_p * self.state_var[-1])
        transformed_input[..., 8] = torch.cos(theta_p * self.state_var[-1])
        return transformed_input

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        xe, ye, theta_e = state[..., 0], state[..., 1], state[..., 2]
        xp, yp, theta_p = state[..., 3], state[..., 4], state[..., 5]
        ue = control[..., 0]
        up = disturbance[..., 0]
        dsdt[..., 0] = self.evader_velocity * torch.cos(theta_e)
        dsdt[..., 1] = self.evader_velocity * torch.sin(theta_e)
        dsdt[..., 2] = ue
        dsdt[..., 3] = self.pursuer_velocity * torch.cos(theta_p)
        dsdt[..., 4] = self.pursuer_velocity * torch.sin(theta_p)
        dsdt[..., 5] = up
        return dsdt

    def reach_fn(self, state):
        xe, ye = state[..., 0], state[..., 1]
        box_bounds = self.box_bounds.to(state.device)
        dx_min = xe - box_bounds[0, 0]
        dx_max = box_bounds[0, 1] - xe
        dy_min = ye - box_bounds[1, 0]
        dy_max = box_bounds[1, 1] - ye
        return torch.min(
            torch.stack([dx_min, dx_max, dy_min, dy_max], dim=-1), dim=-1
        ).values

    def avoid_fn(self, state):
        xe, ye = state[..., 0], state[..., 1]
        xp, yp = state[..., 3], state[..., 4]
        return torch.sqrt((xe - xp) ** 2 + (ye - yp) ** 2) - self.goalR

    def boundary_fn(self, state):
        if self.set_mode == "avoid":
            return torch.minimum(self.avoid_fn(state), self.reach_fn(state))
        elif self.set_mode == "reach":
            return self.reach_fn(state)
        elif self.set_mode == "reach_avoid":
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))
        else:
            raise NotImplementedError

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        if self.set_mode == "avoid":
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        elif self.set_mode == "reach":
            return torch.min(self.reach_fn(state_traj), dim=-1).values
        elif self.set_mode == "reach_avoid":
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            worst_avoid = torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)
            return torch.min(torch.clamp(reach_values, min=worst_avoid), dim=-1).values
        else:
            raise NotImplementedError

    def hamiltonian(self, state, dvds):
        theta_e, theta_p = state[..., 2], state[..., 5]
        base = (
            self.evader_velocity * torch.cos(theta_e) * dvds[..., 0]
            + self.evader_velocity * torch.sin(theta_e) * dvds[..., 1]
            + self.pursuer_velocity * torch.cos(theta_p) * dvds[..., 3]
            + self.pursuer_velocity * torch.sin(theta_p) * dvds[..., 4]
        )
        a_term = dvds[..., 2]
        b_term = dvds[..., 5]
        if self.set_mode == "avoid":
            return (
                base
                + self.omega_e_max * torch.abs(a_term)
                - self.omega_p_max * torch.abs(b_term)
            )
        elif self.set_mode in ["reach", "reach_avoid"]:
            return (
                base
                - self.omega_e_max * torch.abs(a_term)
                + self.omega_p_max * torch.abs(b_term)
            )
        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        dVdtheta_e = dvds[..., 2]
        if self.set_mode == "avoid":
            return self.omega_e_max * torch.sign(dVdtheta_e)[..., None]
        elif self.set_mode in ["reach", "reach_avoid"]:
            return -self.omega_e_max * torch.sign(dVdtheta_e)[..., None]
        else:
            raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        dVdtheta_p = dvds[..., 5]
        if self.set_mode == "avoid":
            return -self.omega_p_max * torch.sign(dVdtheta_p)[..., None]
        elif self.set_mode in ["reach", "reach_avoid"]:
            return self.omega_p_max * torch.sign(dVdtheta_p)[..., None]
        else:
            raise NotImplementedError

    def plot_config(self):
        return {
            "state_slices": [0, 0, 0, 0, 0, 0],
            "state_labels": ["x_e", "y_e", r"$\theta_e$", "x_p", "y_p", r"$\theta_p$"],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 2,
        }


class QuadrotorWithDist(Quadrotor):
    def __init__(
        self,
        collisionR: float,
        collective_thrust_max: float,
        set_mode: str,
        disturbance_max: float,
    ):
        super().__init__(collisionR, collective_thrust_max, set_mode)
        self.disturbance_max = disturbance_max
        self.wind_force_max = disturbance_max
        self.wind_torque_max = disturbance_max
        self.disturbance_dim = 6  # 3 forces + 3 torques
        self.disturbance_range_ = torch.tensor(
            [[-self.wind_force_max, self.wind_force_max]] * 3
            + [[-self.wind_torque_max, self.wind_torque_max]] * 3
        ).to(device)
        self.disturbance_init = torch.zeros(self.disturbance_dim).to(device)
        self.name = "QuadrotorWithDist"

    def disturbance_range(self, state):
        return self.disturbance_range_.cpu().tolist()

    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def bound_disturbance(self, disturbance):
        return torch.clamp(
            disturbance,
            self.disturbance_range_[..., 0],
            self.disturbance_range_[..., 1],
        )

    def dsdt(self, state, control, disturbance):
        dsdt = super().dsdt(state, control, disturbance)
        dsdt[..., 7] += disturbance[..., 0]
        dsdt[..., 8] += disturbance[..., 1]
        dsdt[..., 9] += disturbance[..., 2]
        dsdt[..., 10] += disturbance[..., 3]
        dsdt[..., 11] += disturbance[..., 4]
        dsdt[..., 12] += disturbance[..., 5]
        return dsdt

    def hamiltonian(self, state, dvds):
        ham = super().hamiltonian(state, dvds)
        dist_terms = (
            torch.abs(dvds[..., 7]) * self.wind_force_max
            + torch.abs(dvds[..., 8]) * self.wind_force_max
            + torch.abs(dvds[..., 9]) * self.wind_force_max
            + torch.abs(dvds[..., 10]) * self.wind_torque_max
            + torch.abs(dvds[..., 11]) * self.wind_torque_max
            + torch.abs(dvds[..., 12]) * self.wind_torque_max
        )
        if self.set_mode in ["avoid", "reach_avoid"]:
            ham -= dist_terms
        elif self.set_mode == "reach":
            ham += dist_terms
        else:
            raise NotImplementedError
        return ham

    def optimal_disturbance(self, state, dvds):
        grad_disturb = dvds[..., 7:13]
        if self.set_mode == "avoid":
            sign = torch.sign(grad_disturb)
        elif self.set_mode == "reach":
            sign = -torch.sign(grad_disturb)
        else:
            raise ValueError(f"Unknown set_mode: {self.set_mode}")
        force = sign[..., :3] * self.wind_force_max
        torque = sign[..., 3:] * self.wind_torque_max
        return torch.cat((force, torque), dim=-1)

class DronePursuitEvasion20D(Dynamics):
    """
    20D Drone pursuit-evasion system: 1 evader, 1 pursuer.
    State: [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z,  # Drone 1 (evader)
            x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z] # Drone 2 (pursuer)
    Control: [S1_x, S1_y, T1_z] (evader control)
    Disturbance: [S2_x, S2_y, T2_z] (pursuer control)
    """
    disturbance_dim = 3

    def __init__(self, thrust_max: float, max_angle: float, max_torque: float, capture_radius: float, set_mode: str, capture_shape: str = 'ellipse'):
        self.state_dim = 20  # 10D for each drone
        self.control_dim = 3  # 3 controls for evader
        self.disturbance_dim = 3  # 3 controls for pursuer

        self.control_max = 1.0  # u_max (normalized control bound)
        self.max_torque = max_torque
        self.Gz = -9.81
        self.max_v = 2.0
        self.max_omega = 2.0  # Maximum angular velocity
        self.max_theta = max_angle  # Maximum angle (radians)
        self.capture_radius = capture_radius
        self.capture_shape = capture_shape  # 'cylinder', 'ellipse', or 'cone'

        # Drone dynamics parameters
        self.d0 = 20.0
        self.d1 = 4.5
        self.n0 = 18.0
        self.k_T = 0.83
        self.thrust_max = thrust_max
        self.mass = 1.0
        self.c_x = 0.3  # Drag coefficient for x direction
        self.c_y = 0.3  # Drag coefficient for y direction

        # State: [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z,  # Drone 1 (evader)
        #         x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z] # Drone 2 (pursuer)
        
        # State ranges for both drones (same as Drone10DWithDist)
        drone_state_range = torch.tensor([
            [-4.5, 4.5], [-self.max_v, self.max_v], [-self.max_theta, self.max_theta], [-self.max_omega, self.max_omega],  # x, v_x, θ_x, ω_x
            [-2.5, 2.5], [-self.max_v, self.max_v], [-self.max_theta, self.max_theta], [-self.max_omega, self.max_omega],  # y, v_y, θ_y, ω_y
            [0.0, 2.2], [-self.max_v, self.max_v],  # z, v_z
        ])

        # Combine state ranges for both drones
        state_range_ = torch.cat([drone_state_range, drone_state_range], dim=0)
        
        control_range_ = torch.tensor([
            [-self.control_max, self.control_max],  # S1_x
            [-self.control_max, self.control_max],  # S1_y
            [0.25, self.control_max],  # T1_z
        ])
        disturbance_range_ = torch.tensor([
            [-self.control_max, self.control_max],  # S2_x
            [-self.control_max, self.control_max],  # S2_y
            [0.25, self.control_max],  # T2_z
        ])

        box_bounds_ = torch.tensor([
            [-4.0, 4.0], [-self.max_v, self.max_v],
            [-2.0, 2.0], [-self.max_v, self.max_v],
            [0.2, 2.0],  [-self.max_v, self.max_v],
        ])
        self.set_mode = set_mode
        if self.set_mode in ["avoid", "reach"]:
            loss_type = "brt_hjivi"
        elif self.set_mode == "avoid_flipped":
            loss_type = "brt_hjivi_inversed"
        elif self.set_mode == "reach_avoid":
            loss_type = "brat_hjivi"
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

        state_mean_ = (state_range_[:, 0] + state_range_[:, 1]) / 2.0
        state_var_ = (state_range_[:, 1] - state_range_[:, 0]) / 2.0

        super().__init__(
            name="DronePursuitEvasion20D", loss_type=loss_type, set_mode=set_mode,
            state_dim=20, input_dim=25, control_dim=self.control_dim, disturbance_dim=self.disturbance_dim,
            state_mean=state_mean_.tolist(),
            state_var=state_var_.tolist(),
            value_mean=0.2,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model='exact'
        )
        
        self.box_bounds_ = box_bounds_.to(device)

        self.state_range_ = state_range_.to(device)
        self.control_range_ = control_range_.to(device)
        self.disturbance_range_ = disturbance_range_.to(device)

        self.control_init = torch.tensor([0, 0, (-self.Gz * self.mass) / (self.thrust_max * self.k_T)]).to(device)
        self.disturbance_init = torch.tensor([0, 0, (-self.Gz * self.mass) / (self.thrust_max * self.k_T)]).to(device)

        # self.eps_var_control = torch.tensor([self.max_torque, self.max_torque, self.thrust_max]).to(device)  
        # self.eps_var_disturbance = torch.tensor([self.max_torque, self.max_torque, self.thrust_max]).to(device)  

        self.eps_var_control = torch.tensor([self.control_max, self.control_max, self.thrust_max]).to(device)  
        self.eps_var_disturbance = torch.tensor([self.control_max, self.control_max, self.thrust_max]).to(device)  

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        
        # State: [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z,  # Drone 1 (evader)
        #         x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z] # Drone 2 (pursuer)
        # Control: [S1_x, S1_y, T1_z] (evader)
        # Disturbance: [S2_x, S2_y, T2_z] (pursuer)
        
        # Drone 1 (evader) dynamics - indices 0-9
        # Position derivatives
        dsdt[..., 0] = state[..., 1]  # x1_dot = v1_x
        dsdt[..., 4] = state[..., 5]  # y1_dot = v1_y
        dsdt[..., 8] = state[..., 9]  # z1_dot = v1_z
        
        # Velocity derivatives (with evader control, disturbance, and drag terms)
        dsdt[..., 1] = -self.Gz * torch.tan(state[..., 2]) - self.c_x * state[..., 1]  # v̇1_x = g * tan(θ1_x) - c_x * v1_x
        dsdt[..., 5] = -self.Gz * torch.tan(state[..., 6]) - self.c_y * state[..., 5]  # v̇1_y = g * tan(θ1_y) - c_y * v1_y
        dsdt[..., 9] = (self.k_T / self.mass) * self.thrust_max * control[..., 2] + self.Gz  # v1_z_dot = T1_z - g
        
        # Angle derivatives
        dsdt[..., 2] = state[..., 3] - self.d1 * state[..., 2]  # θ1_x_dot = ω1_x - d1 * θ1_x
        dsdt[..., 6] = state[..., 7] - self.d1 * state[..., 6]  # θ1_y_dot = ω1_y - d1 * θ1_y
        
        # Angular velocity derivatives
        dsdt[..., 3] = -self.d0 * state[..., 2] + self.n0 * self.max_torque * control[..., 0]  # ω̇1_x
        dsdt[..., 7] = -self.d0 * state[..., 6] + self.n0 * self.max_torque * control[..., 1]  # ω̇1_y
        
        # Drone 2 (pursuer) dynamics - indices 10-19
        # Position derivatives
        dsdt[..., 10] = state[..., 11]  # x2_dot = v2_x
        dsdt[..., 14] = state[..., 15]  # y2_dot = v2_y
        dsdt[..., 18] = state[..., 19]  # z2_dot = v2_z
        
        # Velocity derivatives (with pursuer control, disturbance, and drag terms)
        dsdt[..., 11] = -self.Gz * torch.tan(state[..., 12]) - self.c_x * state[..., 11]  # v̇2_x = g * tan(θ2_x) - c_x * v2_x
        dsdt[..., 15] = -self.Gz * torch.tan(state[..., 16]) - self.c_y * state[..., 15]  # v̇2_y = g * tan(θ2_y) - c_y * v2_y
        dsdt[..., 19] = (self.k_T / self.mass) * self.thrust_max * disturbance[..., 2] + self.Gz  # v2_z_dot = T2_z - g
        
        # Angle derivatives
        dsdt[..., 12] = state[..., 13] - self.d1 * state[..., 12]  # θ2_x_dot = ω2_x - d1 * θ2_x
        dsdt[..., 16] = state[..., 17] - self.d1 * state[..., 16]  # θ2_y_dot = ω2_y - d1 * θ2_y
        
        # Angular velocity derivatives
        dsdt[..., 13] = -self.d0 * state[..., 12] + self.n0 * self.max_torque * disturbance[..., 0]  # ω̇2_x
        dsdt[..., 17] = -self.d0 * state[..., 16] + self.n0 * self.max_torque * disturbance[..., 1]  # ω̇2_y
        
        return dsdt

    def hamiltonian(self, state, dvds):
        # Extract velocities and gradients for both drones
        v1 = state[..., [1, 5, 9]]  # [v1_x, v1_y, v1_z]
        v2 = state[..., [11, 15, 19]]  # [v2_x, v2_y, v2_z]
        omega1 = state[..., [3, 7]]  # [ω1_x, ω1_y]
        omega2 = state[..., [13, 17]]  # [ω2_x, ω2_y]
        theta1 = state[..., [2, 6]]  # [θ1_x, θ1_y]
        theta2 = state[..., [12, 16]]  # [θ2_x, θ2_y]

        # Gradients for drone 1 (evader)
        dVdp1 = dvds[..., [0, 4, 8]]  # [dV/dx1, dV/dy1, dV/dz1]
        dVdv1 = dvds[..., [1, 5, 9]]  # [dV/dv1_x, dV/dv1_y, dV/dv1_z]
        dVdtheta1 = dvds[..., [2, 6]]  # [dV/dθ1_x, dV/dθ1_y]
        dVdomega1 = dvds[..., [3, 7]]  # [dV/dω1_x, dV/dω1_y]
        
        # Gradients for drone 2 (pursuer)
        dVdp2 = dvds[..., [10, 14, 18]]  # [dV/dx2, dV/dy2, dV/dz2]
        dVdv2 = dvds[..., [11, 15, 19]]  # [dV/dv2_x, dV/dv2_y, dV/dv2_z]
        dVdtheta2 = dvds[..., [12, 16]]  # [dV/dθ2_x, dV/dθ2_y]
        dVdomega2 = dvds[..., [13, 17]]  # [dV/dω2_x, dV/dω2_y]
        
        # Drone 1 (evader) terms
        ham = (v1 * dVdp1).sum(-1)  # Position derivatives
        ham += dVdv1[..., 0] * (-self.Gz * torch.tan(theta1[..., 0]) - self.c_x * v1[..., 0])  # v1_x term with drag
        ham += dVdv1[..., 1] * (-self.Gz * torch.tan(theta1[..., 1]) - self.c_y * v1[..., 1])  # v1_y term with drag
        ham += dVdv1[..., 2] * self.Gz  # v1_z gravity term
        ham += (omega1 * dVdtheta1).sum(-1)  # Angle derivatives
        ham += dVdtheta1[..., 0] * (-self.d1 * theta1[..., 0])  # θ1_x damping
        ham += dVdtheta1[..., 1] * (-self.d1 * theta1[..., 1])  # θ1_y damping
        ham += dVdomega1[..., 0] * (-self.d0 * theta1[..., 0])  # ω1_x damping
        ham += dVdomega1[..., 1] * (-self.d0 * theta1[..., 1])  # ω1_y damping
        
        # Drone 2 (pursuer) terms
        ham += (v2 * dVdp2).sum(-1)  # Position derivatives
        ham += dVdv2[..., 0] * (-self.Gz * torch.tan(theta2[..., 0]) - self.c_x * v2[..., 0])  # v2_x term with drag
        ham += dVdv2[..., 1] * (-self.Gz * torch.tan(theta2[..., 1]) - self.c_y * v2[..., 1])  # v2_y term with drag
        ham += dVdv2[..., 2] * self.Gz  # v2_z gravity term
        ham += (omega2 * dVdtheta2).sum(-1)  # Angle derivatives
        ham += dVdtheta2[..., 0] * (-self.d1 * theta2[..., 0])  # θ2_x damping
        ham += dVdtheta2[..., 1] * (-self.d1 * theta2[..., 1])  # θ2_y damping
        ham += dVdomega2[..., 0] * (-self.d0 * theta2[..., 0])  # ω2_x damping
        ham += dVdomega2[..., 1] * (-self.d0 * theta2[..., 1])  # ω2_y damping
        
        # Control and disturbance terms
        if self.set_mode in ['avoid', 'avoid_flipped']:
            # Evader tries to avoid capture (minimize value function)
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 0])  # S1_x
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 1])  # S1_y
            ham += (self.k_T / self.mass) * self.thrust_max * torch.where(dVdv1[..., 2] > 0, self.control_max, 0.25) * torch.abs(dVdv1[..., 2])  # T1_z
            
            # Pursuer tries to capture (maximize value function)
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 0])  # S2_x
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 1])  # S2_y
            ham -= (self.k_T / self.mass) * self.thrust_max * torch.where(dVdv2[..., 2] < 0, self.control_max, 0.25) * torch.abs(dVdv2[..., 2])  # T2_z

        elif self.set_mode == 'reach':
            # Evader tries to reach target (maximize value function)
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 0])  # S1_x
            ham -= self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega1[..., 1])  # S1_y
            ham -= (self.k_T / self.mass) * self.thrust_max * torch.where(dVdv1[..., 2] > 0, self.control_max, 0.25) * torch.abs(dVdv1[..., 2])  # T1_z
            
            # Pursuer tries to prevent reaching (minimize value function)
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 0])  # S2_x
            ham += self.n0 * self.max_torque * self.control_max * torch.abs(dVdomega2[..., 1])  # S2_y
            ham += (self.k_T / self.mass) * self.thrust_max * torch.where(dVdv2[..., 2] > 0, self.control_max, 0.25) * torch.abs(dVdv2[..., 2])  # T2_z
        
        return ham

    def boundary_fn(self, state):
        
        # Drone 1 position: [x1, y1, z1] - indices 0, 4, 8
        p1 = torch.stack([state[..., 0], state[..., 4], state[..., 8]], dim=-1)
        # Drone 2 position: [x2, y2, z2] - indices 10, 14, 18
        p2 = torch.stack([state[..., 10], state[..., 14], state[..., 18]], dim=-1)

        height = 0.75

        if self.capture_shape == 'cylinder':
            # Original cylinder implementation
            horizontal_dist = torch.sqrt((p1[..., 0] - p2[..., 0])**2 + (p1[..., 1] - p2[..., 1])**2) - self.capture_radius
            
            # Vertical distance: evader is above pursuer (positive) or below (negative)
            # Collision if evader is within height below pursuer
            z_diff = p1[..., 2] - p2[..., 2]  # positive if evader above pursuer
            vertical_dist = torch.where(z_diff > 0, z_diff, (p2[..., 2] - p1[..., 2]) - height)

            # Case 1: Outside in both directions
            outside_both = (horizontal_dist > 0) & (vertical_dist > 0)
            dist_outside = torch.sqrt(horizontal_dist**2 + vertical_dist**2)

            # Case 2: Outside horizontally
            outside_horiz = (horizontal_dist > 0) & (vertical_dist <= 0)

            # Case 3: Inside horizontally, outside vertically
            outside_vert = (horizontal_dist <= 0) & (vertical_dist > 0)

            # Case 4: Inside both (inside the cylinder)
            inside_both = (horizontal_dist <= 0) & (vertical_dist <= 0)
            dist_inside = torch.maximum(horizontal_dist, vertical_dist)  # least negative

            # Combine all cases
            inter_drone_dist = torch.where(
                outside_both, dist_outside,
                torch.where(
                    outside_horiz, horizontal_dist,
                    torch.where(
                        outside_vert, vertical_dist,
                        dist_inside
                    )
                )
            )

        elif self.capture_shape == 'ellipse':
            horizontal_radius = self.capture_radius  # a
            vertical_radius = height                 # c

            # Relative position
            dx = p1[..., 0] - p2[..., 0]
            dy = p1[..., 1] - p2[..., 1]
            dz = p1[..., 2] - p2[..., 2]

            # Euclidean distance to pursuer (for points above)
            dist_center = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

            # Approximate ellipsoid SDF (first-order, smooth)
            inv_a2 = 1.0 / (horizontal_radius * horizontal_radius)
            inv_c2 = 1.0 / (vertical_radius * vertical_radius)
            F = (dx * dx) * inv_a2 + (dy * dy) * inv_a2 + (dz * dz) * inv_c2 - 1.0
            G = torch.sqrt((dx * inv_a2) ** 2 + (dy * inv_a2) ** 2 + (dz * inv_c2) ** 2 + 1e-8) * 2.0
            d_ellip = F / (G + 1e-8)  # approximate signed distance to ellipsoid

            d_plane = dz - 0.50  # Cut off at z = 0.5 above pursuer
            
            m = torch.maximum(d_ellip, d_plane)
            sharpness = 8.0
            signed_dist = m + torch.log(
                torch.exp((d_ellip - m) * sharpness) +
                torch.exp((d_plane - m) * sharpness)
            ) / sharpness
            
            above_factor = torch.sigmoid(dz * 10.0)
            inter_drone_dist = above_factor * dist_center + (1 - above_factor) * signed_dist

        elif self.capture_shape == 'cone':
            # Smooth SDF for a truncated cone:
            # - Apex at z = 0.5 above pursuer (virtual apex)
            # - Truncated at z = 0.25 above pursuer (top cap)
            # - Base at z = -height
            # Negative inside, positive outside.
            dx = p1[..., 0] - p2[..., 0]
            dy = p1[..., 1] - p2[..., 1]
            dz = p1[..., 2] - p2[..., 2]

            horizontal_dist = torch.sqrt(dx**2 + dy**2 + 1e-8)

            # Linear radius shrink for cone (apex at z = 0.5)
            # At z = 0.5: radius = 0 (apex)
            # At z = 0.25: radius = R * 0.25 / (height + 0.25)
            # At z = -height: radius = R * (height + 0.5) / (height + 0.25)
            cone_radius = self.capture_radius * (0.5 - dz) / (height + 0.25)

            # Signed distance to lateral cone surface (negative inside)
            d_lateral = horizontal_dist - cone_radius

            # SDF for top plane (z <= 0.25) - truncation plane
            d_top = dz - 0.25  # positive above truncation plane

            # SDF for bottom plane (z >= -height)
            d_bottom = -(dz + height)  # positive below base

            # Combine using smooth max for outside
            # (soft union: distance = max(d_lateral, d_top, d_bottom))
            sharpness = 16.0
            m = torch.maximum(torch.maximum(d_lateral, d_top), d_bottom)
            inter_drone_dist = m + torch.log(
                torch.exp((d_lateral - m) * sharpness) +
                torch.exp((d_top - m) * sharpness) +
                torch.exp((d_bottom - m) * sharpness)
            ) / sharpness

        else:
            raise ValueError(f"Unknown capture shape: {self.capture_shape}. Must be 'cylinder', 'ellipse', or 'cone'")

        capture_constraint = inter_drone_dist

        # For each dimension, how far from the nearest boundary (positive inside, negative outside)
        # Use drone 1 position for box constraints
        px, py, pz = state[..., 0], state[..., 4], state[..., 8]

        box_bounds = self.box_bounds_.to(state.device)
        

        x_min, x_max = box_bounds[0, 0], box_bounds[0, 1]
        y_min, y_max = box_bounds[2, 0], box_bounds[2, 1]
        z_min, z_max = box_bounds[4, 0], box_bounds[4, 1]

        # Compute per-dimension signed distances to box faces
        dx_min = px - x_min
        dx_max = x_max - px
        dy_min = py - y_min
        dy_max = y_max - py
        dz_min = pz - z_min
        dz_max = z_max - pz

        # Inside: minimum distance to any face (negative inside, zero on surface)
        inside_dist = torch.min(torch.stack([dx_min, dx_max, dy_min, dy_max, dz_min, dz_max], dim=-1), dim=-1).values

        # For outside: compute the per-dimension "over" (how far outside the box in each dim)
        over_x = torch.clamp(px - x_max, min=0) + torch.clamp(x_min - px, min=0)
        over_y = torch.clamp(py - y_max, min=0) + torch.clamp(y_min - py, min=0)
        over_z = torch.clamp(pz - z_max, min=0) + torch.clamp(z_min - pz, min=0)
        # Norm of the "over" vector gives Euclidean distance outside
        outside_dist = torch.norm(torch.stack([over_x, over_y, over_z], dim=-1), dim=-1)

        # If all inside (all distances to faces > 0), use inside_dist; else use outside_dist
        is_inside = (dx_min > 0) & (dx_max > 0) & (dy_min > 0) & (dy_max > 0) & (dz_min > 0) & (dz_max > 0)
        inside_constraint = torch.where(is_inside, inside_dist, -outside_dist)
        
        if self.set_mode in ['avoid', 'avoid_flipped']:
            # Safe if outside capture radius AND inside bounds
            return torch.minimum(capture_constraint, inside_constraint)
        else:
            return torch.minimum(-capture_constraint, inside_constraint)

    def optimal_control(self, state, dvds):
        # Extract gradients for evader controls
        dVdomega1 = dvds[..., [3, 7]]  # [dV/dω1_x, dV/dω1_y] for torque controls S1_x, S1_y
        dVdv1 = dvds[..., [1, 5, 9]]  # [dV/dv1_x, dV/dv1_y, dV/dv1_z] for thrust control T1_z
        
        control = torch.zeros_like(dVdv1)

        if self.set_mode in ['avoid', 'avoid_flipped']:
            # Evader tries to avoid capture (minimize value function)
            control[..., 0] = self.control_max * torch.sign(dVdomega1[..., 0])  # S1_x
            control[..., 1] = self.control_max * torch.sign(dVdomega1[..., 1])  # S1_y
            control[..., 2] = torch.where(dVdv1[..., 2] > 0, self.control_max, 0.25)  # T1_z
        elif self.set_mode == 'reach':
            # Evader tries to reach target (maximize value function)
            control[..., 0] = -self.control_max * torch.sign(dVdomega1[..., 0])  # S1_x
            control[..., 1] = -self.control_max * torch.sign(dVdomega1[..., 1])  # S1_y
            control[..., 2] = torch.where(dVdv1[..., 2] < 0, self.control_max, 0.25)  # T1_z
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")
        return control

    def optimal_disturbance(self, state, dvds):
        # Extract gradients for pursuer controls
        dVdomega2 = dvds[..., [13, 17]]  # [dV/dω2_x, dV/dω2_y] for torque controls S2_x, S2_y
        dVdv2 = dvds[..., [11, 15, 19]]  # [dV/dv2_x, dV/dv2_y, dV/dv2_z] for thrust control T2_z
        
        disturbance = torch.zeros_like(dVdv2)

        if self.set_mode in ['avoid', 'avoid_flipped']:
            # Pursuer tries to capture (maximize value function)
            disturbance[..., 0] = -self.control_max * torch.sign(dVdomega2[..., 0])  # S2_x
            disturbance[..., 1] = -self.control_max * torch.sign(dVdomega2[..., 1])  # S2_y
            disturbance[..., 2] = torch.where(dVdv2[..., 2] < 0, self.control_max, 0.25)  # T2_z
        elif self.set_mode == 'reach':
            # Pursuer tries to prevent reaching (minimize value function)
            disturbance[..., 0] = self.control_max * torch.sign(dVdomega2[..., 0])  # S2_x
            disturbance[..., 1] = self.control_max * torch.sign(dVdomega2[..., 1])  # S2_y
            disturbance[..., 2] = torch.where(dVdv2[..., 2] > 0, self.control_max, 0.25)  # T2_z
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

        return disturbance

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # Wrap θ1_x, θ1_y, θ2_x, θ2_y (indices 2, 6, 12, 16)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 12] = (wrapped_state[..., 12] + math.pi) % (2 * math.pi) - math.pi
        wrapped_state[..., 16] = (wrapped_state[..., 16] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        # Transform periodic angles θ1_x, θ1_y, θ2_x, θ2_y to sin/cos components
        # Input: [..., 21] - [t, x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
        #                     x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z]
        # Output: [..., 25] - [t, x1, v1_x, sin(θ1_x), cos(θ1_x), ω1_x, y1, v1_y, sin(θ1_y), cos(θ1_y), ω1_y, z1, v1_z,
        #                      x2, v2_x, sin(θ2_x), cos(θ2_x), ω2_x, y2, v2_y, sin(θ2_y), cos(θ2_y), ω2_y, z2, v2_z]
        
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] + 4  # Add 4 dimensions for sin/cos transforms
        transformed_input = torch.zeros(output_shape, device=input.device)
        
        # Copy non-periodic variables for drone 1
        transformed_input[..., 0] = input[..., 0]  # t (time)
        transformed_input[..., 1] = input[..., 1]  # x1
        transformed_input[..., 2] = input[..., 2]  # v1_x
        transformed_input[..., 5] = input[..., 4]  # ω1_x
        transformed_input[..., 6] = input[..., 5]  # y1
        transformed_input[..., 7] = input[..., 6]  # v1_y
        transformed_input[..., 10] = input[..., 8]  # ω1_y
        transformed_input[..., 11] = input[..., 9]  # z1
        transformed_input[..., 12] = input[..., 10]  # v1_z
        
        # Transform θ1_x, θ1_y to sin/cos
        theta1_x = input[..., 3]
        transformed_input[..., 3] = torch.sin(theta1_x * self.state_var[2])  # sin(θ1_x)
        transformed_input[..., 4] = torch.cos(theta1_x * self.state_var[2])  # cos(θ1_x)
        
        theta1_y = input[..., 7]
        transformed_input[..., 8] = torch.sin(theta1_y * self.state_var[6])  # sin(θ1_y)
        transformed_input[..., 9] = torch.cos(theta1_y * self.state_var[6])  # cos(θ1_y)
        
        # Copy non-periodic variables for drone 2
        transformed_input[..., 13] = input[..., 11]  # x2
        transformed_input[..., 14] = input[..., 12]  # v2_x
        transformed_input[..., 17] = input[..., 14]  # ω2_x
        transformed_input[..., 18] = input[..., 15]  # y2
        transformed_input[..., 19] = input[..., 16]  # v2_y
        transformed_input[..., 22] = input[..., 18]  # ω2_y
        transformed_input[..., 23] = input[..., 19]  # z2
        transformed_input[..., 24] = input[..., 20]  # v2_z
        
        # Transform θ2_x, θ2_y to sin/cos
        theta2_x = input[..., 13]
        transformed_input[..., 15] = torch.sin(theta2_x * self.state_var[12])  # sin(θ2_x)
        transformed_input[..., 16] = torch.cos(theta2_x * self.state_var[12])  # cos(θ2_x)
        
        theta2_y = input[..., 17]
        transformed_input[..., 20] = torch.sin(theta2_y * self.state_var[16])  # sin(θ2_y)
        transformed_input[..., 21] = torch.cos(theta2_y * self.state_var[16])  # cos(θ2_y)
        
        return transformed_input

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()
    
    def control_range(self, state):
        return self.control_range_.tolist()

    def disturbance_range(self, state):
        return self.disturbance_range_.tolist()

    def bound_control(self, control):
        return torch.clamp(control, self.control_range_[:, 0], self.control_range_[:, 1])

    def bound_disturbance(self, disturbance):
        return torch.clamp(disturbance, self.disturbance_range_[:, 0], self.disturbance_range_[:, 1])
    
    def clamp_control(self, state, control):
        return self.bound_control(control)
    
    def clamp_disturbance(self, state, disturbance):
        return self.bound_disturbance(disturbance)

    def clip_state(self, state):
        return torch.clamp(state, self.state_range_[..., 0], self.state_range_[..., 1])

    def cost_fn(self, state_traj):
        # Use boundary function for consistency
        if self.set_mode == "avoid":
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        elif self.set_mode == "avoid_flipped":
            return torch.max(self.boundary_fn(state_traj), dim=-1).values
        else:
            raise NotImplementedError(f"Unknown set_mode: {self.set_mode}")

    def plot_config(self):
        return {
            'state_slices': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # Drone 1
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Drone 2
            'state_labels': ['x1', 'v1_x', 'θ1_x', 'ω1_x', 'y1', 'v1_y', 'θ1_y', 'ω1_y', 'z1', 'v1_z',
                           'x2', 'v2_x', 'θ2_x', 'ω2_x', 'y2', 'v2_y', 'θ2_y', 'ω2_y', 'z2', 'v2_z'],
            'x_axis_idx': 0,  # x1
            'y_axis_idx': 4,  # y1
            'z_axis_idx': 8,  # z1
        }