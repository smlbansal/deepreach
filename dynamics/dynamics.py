from abc import ABC, abstractmethod
from utils import diff_operators

import math
import torch

# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parametrized models
class Dynamics(ABC):
    def __init__(self, 
    loss_type:str, set_mode:str, 
    state_dim:int, input_dim:int, 
    control_dim:int, disturbance_dim:int, 
    state_mean:list, state_var:list, 
    value_mean:float, value_var:float, value_normto:float, 
    diff_model:bool):
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
        self.diff_model = diff_model
        assert self.loss_type in ['brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in ['reach', 'avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + str(len(state_descriptor)) + ' != ' + str(self.state_dim)
    
    # ALL METHODS ARE BATCH COMPATIBLE

    # MODEL-UNIT CONVERSIONS (TODO: refactor into separate model-unit conversion class?)

    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.diff_model:
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.diff_model:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # ALL FOLLOWING METHODS USE REAL UNITS

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

class ParameterizedVertDrone2D(Dynamics):
    def __init__(self, gravity:float, input_multiplier_max:float, input_magnitude_max:float):
        self.gravity = gravity                             # g
        self.input_multiplier_max = input_multiplier_max   # k_max
        self.input_magnitude_max = input_magnitude_max     # u_max
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 1.5, self.input_multiplier_max/2], # v, z, k
            state_var=[4, 2, self.input_multiplier_max/2],    # v, z, k
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            diff_model=True,
        )

    def state_test_range(self):
        return [
            [-4, 4],                        # v
            [-0.5, 3.5],                    # z
            [0, self.input_multiplier_max], # k
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2]*control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        dsdt[..., 2] = 0
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        raise NotImplementedError

    def hamiltonian(self, state, dvds):
        return state[..., 2]*torch.abs(dvds[..., 0]*self.input_magnitude_max) \
                - dvds[..., 0]*self.gravity \
                + dvds[..., 1]*state[..., 0]
    
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    def plot_config(self):
        return {
            'state_slices': [0, 1.5, self.input_multiplier_max/2],
            'state_labels': ['v', 'z', 'k'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Air3D(Dynamics):
    def __init__(self, collisionR:float, velocity:float, omega_max:float, angle_alpha_factor:float):
        self.collisionR = collisionR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=1,
            state_mean=[0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            diff_model=True,
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    # Air3D dynamics
    # \dot x    = -v + v \cos \psi + u y
    # \dot y    = v \sin \psi - u x
    # \dot \psi = d - u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = -self.velocity + self.velocity*torch.cos(state[..., 2]) + control[..., 0]*state[..., 1]
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2]) - control[..., 0]*state[..., 0]
        dsdt[..., 2] = disturbance[..., 0] - control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        ham = self.omega_max * torch.abs(dvds[..., 0] * state[..., 1] - dvds[..., 1] * state[..., 0] - dvds[..., 2])  # Control component
        ham = ham - self.omega_max * torch.abs(dvds[..., 2])  # Disturbance component
        ham = ham + (self.velocity * (torch.cos(state[..., 2]) - 1.0) * dvds[..., 0]) + (self.velocity * torch.sin(state[..., 2]) * dvds[..., 1])  # Constant component
        return ham

    def optimal_control(self, state, dvds):
        det = dvds[..., 0]*state[..., 1] - dvds[..., 1]*state[..., 0]-dvds[..., 2]
        return (self.omega_max * torch.sign(det))[..., None]
    
    def optimal_disturbance(self, state, dvds):
        return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', 'theta'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Dubins3D(Dynamics):
    def __init__(self, goalR:float, velocity:float, omega_max:float, angle_alpha_factor:float, set_mode:str, freeze_model: bool):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            diff_model=True
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        if self.freeze_model:
            raise NotImplementedError
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        elif self.set_mode == 'avoid':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return (-self.omega_max*torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == 'avoid':
            return (self.omega_max*torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Dubins4D(Dynamics):
    def __init__(self, bound_mode:str):
        self.vMin = 0.2
        self.vMax = 14.8
        self.collisionR = 1.5
        self.bound_mode = bound_mode
        assert self.bound_mode in ['v1', 'v2']

        xMean = 0
        yMean = 0
        thetaMean = 0
        vMean = 7.5
        aMean = 0
        oMean = 0

        xVar = 10
        yVar = 10
        thetaVar = 1.2*math.pi
        vVar = 7.5
        aVar = 10
        oVar = 3*math.pi if self.bound_mode == 'v1' else 2.0
        
        super().__init__(
            loss_type='brt_hjivi',
            state_dim=14, input_dim=15,  control_dim=2, disturbance_dim=0,
            state_mean=[xMean, yMean, thetaMean, vMean, xMean, yMean, aMean, aMean, oMean, oMean, aMean, aMean, oMean, oMean],
            state_var=[xVar, yVar, thetaVar, vVar, xVar, yVar, aVar, aVar, oVar, oVar, aVar, aVar, oVar, oVar],
            value_mean=13,
            value_var=14,
            value_normto=0.02,
            diff_model=True,
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
            [self.vMin, self.vMax],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    def boundary_fn(self, state):
        return torch.norm(state[..., 0:2] - state[..., 4:6], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        raise NotImplementedError

    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    def optimal_control(self, state, dvds):
        raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    def plot_config(self):
        raise NotImplementedError

class NarrowPassage(Dynamics):
    def __init__(self, avoid_fn_weight:float, avoid_only:bool):
        self.L = 2.0

        # # Target positions
        self.goalX = [6.0, -6.0]
        self.goalY = [-1.4, 1.4]

        # State bounds
        self.vMin = 0.001
        self.vMax = 6.50
        self.phiMin = -0.3*math.pi + 0.001
        self.phiMax = 0.3*math.pi - 0.001

        # Control bounds
        self.aMin = -4.0
        self.aMax = 2.0
        self.psiMin = -3.0*math.pi
        self.psiMax = 3.0*math.pi

        # Lower and upper curb positions (in the y direction)
        self.curb_positions = [-2.8, 2.8]

        # Stranded car position
        self.stranded_car_pos = [0.0, -1.8]

        self.avoid_fn_weight = avoid_fn_weight

        self.avoid_only = avoid_only

        super().__init__(
            loss_type='brt_hjivi' if self.avoid_only else 'brat_hjivi', set_mode='avoid' if self.avoid_only else 'reach',
            state_dim=10, input_dim=11, control_dim=4, disturbance_dim=0,
            # state = [x1, y1, th1, v1, phi1, x2, y2, th2, v2, phi2]
            state_mean=[
                0, 0, 0, 3, 0, 
                0, 0, 0, 3, 0
            ],
            state_var=[
                8.0, 3.8, 1.2*math.pi, 4.0, 1.2*0.3*math.pi, 
                8.0, 3.8, 1.2*math.pi, 4.0, 1.2*0.3*math.pi,
            ],
            value_mean=0.25*8.0,
            value_var=0.5*8.0,
            value_normto=0.02,
            diff_model=True,
        )

    def state_test_range(self):
        return [
            [-8, 8],
            [-3.8, 3.8],
            [-math.pi, math.pi],
            [-1, 7],
            [-0.3*math.pi, 0.3*math.pi],
            [-8, 8],
            [-3.8, 3.8],
            [-math.pi, math.pi],
            [-1, 7],
            [-0.3*math.pi, 0.3*math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 4] = (wrapped_state[..., 4] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi 
        wrapped_state[..., 9] = (wrapped_state[..., 9] + math.pi) % (2*math.pi) - math.pi 
        return wrapped_state 

    # NarrowPassage dynamics
    # \dot x   = v * cos(th)
    # \dot y   = v * sin(th)
    # \dot th  = v * tan(phi) / L
    # \dot v   = u1
    # \dot phi = u2
    # \dot x   = ...
    # \dot y   = ...
    # \dot th  = ...
    # \dot v   = ...
    # \dot phi = ...
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]*torch.cos(state[..., 2])
        dsdt[..., 1] = state[..., 3]*torch.sin(state[..., 2])
        dsdt[..., 2] = state[..., 3]*torch.tan(state[..., 4]) / self.L
        dsdt[..., 3] = control[..., 0]
        dsdt[..., 4] = control[..., 1]
        dsdt[..., 5] = state[..., 8]*torch.cos(state[..., 7])
        dsdt[..., 6] = state[..., 8]*torch.sin(state[..., 7])
        dsdt[..., 7] = state[..., 8]*torch.tan(state[..., 9]) / self.L
        dsdt[..., 8] = control[..., 2]
        dsdt[..., 9] = control[..., 3]
        return dsdt

    def reach_fn(self, state):
        if self.avoid_only:
            raise RuntimeError
        # vehicle 1
        goal_tensor_R1 = torch.tensor([self.goalX[0], self.goalY[0]], device=state.device)
        dist_R1 = torch.norm(state[..., 0:2] - goal_tensor_R1, dim=-1) - self.L
        # vehicle 2
        goal_tensor_R2 = torch.tensor([self.goalX[1], self.goalY[1]], device=state.device)
        dist_R2 = torch.norm(state[..., 5:7] - goal_tensor_R2, dim=-1) - self.L
        return torch.maximum(dist_R1, dist_R2)
    
    def avoid_fn(self, state):
        # distance from lower curb
        dist_lc_R1 = state[..., 1] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state[..., 6] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.minimum(dist_lc_R1, dist_lc_R2)
        
        # distance from upper curb
        dist_uc_R1 = self.curb_positions[1] - state[..., 1] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state[..., 6] - 0.5*self.L
        dist_uc = torch.minimum(dist_uc_R1, dist_uc_R2)
        
        # distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos, device=state.device)
        dist_stranded_R1 = torch.norm(state[..., 0:2] - stranded_car_pos, dim=-1) - self.L
        dist_stranded_R2 = torch.norm(state[..., 5:7] - stranded_car_pos, dim=-1) - self.L
        dist_stranded = torch.minimum(dist_stranded_R1, dist_stranded_R2)

        # distance between the vehicles themselves
        dist_R1R2 = torch.norm(state[..., 0:2] - state[..., 5:7], dim=-1) - self.L

        return self.avoid_fn_weight * torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2)

    def boundary_fn(self, state):
        if self.avoid_only:
            return self.avoid_fn(state)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):    
        if self.avoid_only:
            return torch.min(self.avoid_fn(state_traj), dim=-1).values
        else:   
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn 
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, dvds):
        optimal_control = self.optimal_control(state, dvds)
        return state[..., 3] * torch.cos(state[..., 2]) * dvds[..., 0] + \
               state[..., 3] * torch.sin(state[..., 2]) * dvds[..., 1] + \
               state[..., 3] * torch.tan(state[..., 4]) * dvds[..., 2] / self.L + \
               optimal_control[..., 0] * dvds[..., 3] + \
               optimal_control[..., 1] * dvds[..., 4] + \
               state[..., 8] * torch.cos(state[..., 7]) * dvds[..., 5] + \
               state[..., 8] * torch.sin(state[..., 7]) * dvds[..., 6] + \
               state[..., 8] * torch.tan(state[..., 9]) * dvds[..., 7] / self.L + \
               optimal_control[..., 2] * dvds[..., 8] + \
               optimal_control[..., 3] * dvds[..., 9]

    def optimal_control(self, state, dvds):
        a1_min = self.aMin * (state[..., 3] > self.vMin)
        a1_max = self.aMax * (state[..., 3] < self.vMax)
        psi1_min = self.psiMin * (state[..., 4] > self.phiMin)
        psi1_max = self.psiMax * (state[..., 4] < self.phiMax)
        a2_min = self.aMin * (state[..., 8] > self.vMin)
        a2_max = self.aMax * (state[..., 8] < self.vMax)
        psi2_min = self.psiMin * (state[..., 9] > self.phiMin)
        psi2_max = self.psiMax * (state[..., 9] < self.phiMax)

        if self.avoid_only:
            a1 = torch.where(dvds[..., 3] < 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] < 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] < 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] < 0, psi2_min, psi2_max)

        else:
            a1 = torch.where(dvds[..., 3] > 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] > 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] > 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] > 0, psi2_min, psi2_max)

        return torch.cat((a1[..., None], psi1[..., None], a2[..., None], psi2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                -6.0, -1.4, 0.0, 6.5, 0.0, 
                -6.0, 1.4, -math.pi, 0.0, 0.0
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$', r'$\theta_1$', r'$v_1$', r'$\phi_1$',
                r'$x_2$', r'$y_2$', r'$\theta_2$', r'$v_2$', r'$\phi_2$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class ReachAvoidRocketLanding(Dynamics):
    def __init__(self):
        super().__init__(
            loss_type='brat_hjivi', set_mode='reach',
            state_dim=6, input_dim=7, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            diff_model=True,
        )

    def state_test_range(self):
        return [
            [-150, 150],
            [10, 150],
            [-math.pi, math.pi],
            [-200, 200],
            [-200, 200],
            [-10, 10],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state 

    # \dot x = v_x
    # \dot y = v_y
    # \dot th = w
    # \dot v_x = u1 * cos(th) - u2 sin(th)
    # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
    # \dot w = 0.3 * u1
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]
        dsdt[..., 1] = state[..., 4]
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = control[..., 0]*torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
        dsdt[..., 4] = control[..., 0]*torch.sin(state[..., 2]) + control[..., 1]*torch.cos(state[..., 2]) - 9.81
        dsdt[..., 5] = 0.3*control[..., 0]
        return dsdt

    def reach_fn(self, state):
        # Only target set in the xy direction
        # Target set position in x direction
        dist_x = torch.abs(state[..., 0]) - 20.0 #[-20, 150] boundary_fn range

        # Target set position in y direction
        dist_y = state[..., 1] - 20.0  #[-10, 130] boundary_fn range

        # First compute the target function as you normally would but then normalize it later.
        max_dist = torch.max(dist_x, dist_y)
        return torch.where((max_dist >= 0), max_dist/150.0, max_dist/10.0)

    def avoid_fn(self, state):
        # distance to floor
        dist_y = state[..., 1]

        # distance to wall
        wall_left = -30
        wall_right = -20
        wall_bottom = 0
        wall_top = 100
        dist_left = wall_left - state[..., 0]
        dist_right = state[..., 0] - wall_right
        dist_bottom = wall_bottom - state[..., 1]
        dist_top = state[..., 1] - wall_top
        dist_wall_x = torch.max(dist_left, dist_right)
        dist_wall_y = torch.max(dist_bottom, dist_top)
        dist_wall = torch.norm(torch.cat((torch.max(torch.tensor(0), dist_wall_x).unsqueeze(-1), torch.max(torch.tensor(0), dist_wall_y).unsqueeze(-1)), dim=-1), dim=-1) + torch.min(torch.tensor(0), torch.max(dist_wall_x, dist_wall_y))

        return torch.min(dist_y, dist_wall)

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-20, 20] # y in [-20, 20]
        target_state_range[1] = [10, 20]  # z in [10, 20]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn 
        reach_values = self.reach_fn(state_traj)
        avoid_values = self.avoid_fn(state_traj)
        return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, dvds):
        # Control Hamiltonian
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        ham_ctrl = -250.0 * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)
        # Constant Hamiltonian
        ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
                      dvds[..., 2] * state[..., 5]  - dvds[..., 4] * 9.81
        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant
        return ham_vehicle

    def optimal_control(self, state, dvds):
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
        return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [-100, 120, 0, 150, -5, 0.0],
            'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }

class RocketLanding(Dynamics):
    def __init__(self):
        super().__init__(
            loss_type='brt_hjivi', set_mode='reach',
            state_dim=6, input_dim=8, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            diff_model=True,
        )

    # convert model input to real coord
    def input_to_coord(self, input):
        input = input[..., :-1]
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        input = torch.cat((input, torch.zeros((*input.shape[:-1], 1), device=input.device)), dim=-1)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.diff_model:
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)[..., :-1]

        if self.diff_model:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)


    def state_test_range(self):
        return [
            [-150, 150],
            [10, 150],
            [-math.pi, math.pi],
            [-200, 200],
            [-200, 200],
            [-10, 10],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state 

    # \dot x = v_x
    # \dot y = v_y
    # \dot th = w
    # \dot v_x = u1 * cos(th) - u2 sin(th)
    # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
    # \dot w = 0.3 * u1
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]
        dsdt[..., 1] = state[..., 4]
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = control[..., 0]*torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
        dsdt[..., 4] = control[..., 0]*torch.sin(state[..., 2]) + control[..., 1]*torch.cos(state[..., 2]) - 9.81
        dsdt[..., 5] = 0.3*control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        # Only target set in the yz direction
        # Target set position in y direction
        dist_y = torch.abs(state[..., 0]) - 20.0 #[-20, 150] boundary_fn range

        # Target set position in z direction
        dist_z = state[..., 1] - 20.0  #[-10, 130] boundary_fn range

        # First compute the l(x) as you normally would but then normalize it later.
        lx = torch.max(dist_y, dist_z)
        return torch.where((lx >= 0), lx/150.0, lx/10.0)

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-20, 20] # y in [-20, 20]
        target_state_range[1] = [10, 20]  # z in [10, 20]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        # Control Hamiltonian
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        ham_ctrl = -250.0 * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)
        # Constant Hamiltonian
        ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
                      dvds[..., 2] * state[..., 5]  - dvds[..., 4] * 9.81
        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant
        return ham_vehicle
    
    def optimal_control(self, state, dvds):
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
        return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [-100, 120, 0, 150, -5, 0.0],
            'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }

class Quadrotor(Dynamics):
    def __init__(self, collisionR:float, thrust_max:float, set_mode:str):
        self.thrust_max = thrust_max
        self.m=1 #mass
        self.arm_l=0.17
        self.CT=1
        self.CM=0.016
        self.Gz=-9.8

        self.thrust_max = thrust_max
        self.collisionR = collisionR


        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=[0 for i in range(13)], 
            state_var=[1.5, 1.5, 1.5, 1, 1, 1, 1, 10, 10 ,10 ,10 ,10 ,10],
            value_mean=(math.sqrt(1.5**2+1.5**2+1.5**2)-2*self.collisionR)/2, 
            value_var=math.sqrt(1.5**2+1.5**2+1.5**2), 
            value_normto=0.02,
            diff_model=True
        )

    def state_test_range(self):
        return [
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-10, 10],
            [-10, 10],
            [-10, 10],
            [-10, 10],
            [-10, 10],
            [-10, 10],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
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
        u1 = control[...,0] * 1.0
        u2 = control[...,1] * 1.0
        u3 = control[...,2] * 1.0
        u4 = control[...,3] * 1.0


        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx*qx+wy*qy+wz*qz)/2.0 
        dsdt[..., 4] =  (wx*qw+wz*qy-wy*qz)/2.0
        dsdt[..., 5] = (wy*qw-wz*qx+wx*qz)/2.0
        dsdt[..., 6] = (wz*qw+wy*qx-wx*qy)/2.0
        dsdt[..., 7] = 2*(qw*qy+qx*qz)*self.CT/self.m*(u1+u2+u3+u4)
        dsdt[..., 8] =2*(-qw*qx+qy*qz)*self.CT/self.m*(u1+u2+u3+u4)
        dsdt[..., 9] =self.Gz+(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m*(u1+u2+u3+u4)
        dsdt[..., 10] = 4*math.sqrt(2)*self.CT*(u1-u2-u3+u4)/(3*self.arm_l*self.m)-5*wy*wz/9.0
        dsdt[..., 11] = 4*math.sqrt(2)*self.CT*(-u1-u2+u3+u4)/(3*self.arm_l*self.m)+5*wx*wz/9.0
        dsdt[..., 12] =12*self.CT*self.CM/(7*self.arm_l**2*self.m)*(u1-u2+u3-u4)
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :3], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
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


            C1=2*(qw*qy+qx*qz)*self.CT/self.m
            C2=2*(-qw*qx+qy*qz)*self.CT/self.m
            C3=(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m
            C4=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C5=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C6=12*self.CT*self.CM/(7*self.arm_l**2*self.m)

            # Compute the hamiltonian for the quadrotor
            ham= dvds[..., 0]*vx + dvds[..., 1]*vy+ dvds[..., 2]*vz
            ham+= -dvds[..., 3]* (wx*qx+wy*qy+wz*qz)/2.0 
            ham+= dvds[..., 4]*(wx*qw+wz*qy-wy*qz)/2.0
            ham+= dvds[..., 5]*(wy*qw-wz*qx+wx*qz)/2.0
            ham+= dvds[..., 6]*(wz*qw+wy*qx-wx*qy)/2.0
            ham+= dvds[..., 9]*-9.8
            ham+= -dvds[..., 10]*5*wy*wz/9.0+ dvds[..., 11]*5*wx*wz/9.0

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4-dvds[..., 11]*C5+dvds[..., 12]*C6)*self.thrust_max

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4-dvds[..., 11]*C5-dvds[..., 12]*C6)*self.thrust_max

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4+dvds[..., 11]*C5+dvds[..., 12]*C6)*self.thrust_max

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4+dvds[..., 11]*C5-dvds[..., 12]*C6)*self.thrust_max

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0


            C1=2*(qw*qy+qx*qz)*self.CT/self.m
            C2=2*(-qw*qx+qy*qz)*self.CT/self.m
            C3=(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m
            C4=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C5=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C6=12*self.CT*self.CM/(7*self.arm_l**2*self.m)


            u1=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4-dvds[..., 11]*C5+dvds[..., 12]*C6)
            u2=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4-dvds[..., 11]*C5-dvds[..., 12]*C6)
            u3=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4+dvds[..., 11]*C5+dvds[..., 12]*C6)
            u4=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4+dvds[..., 11]*C5-dvds[..., 12]*C6)

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 2,
            'z_axis_idx': 7,
        }

class MultiVehicleCollision(Dynamics):
    def __init__(self):
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=9, input_dim=10, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0, 
                0, 0,
                0, 0, 0,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi,
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            diff_model=True
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],           
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 8] = (wrapped_state[..., 8] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        return dsdt
    
    def boundary_fn(self, state):
        boundary_values = torch.norm(state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] - state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(state[..., 6]) * dvds[..., 0] + torch.sin(state[..., 6]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 6])
        # Hamiltonian effect due to other vehicles
        ham += self.velocity*(torch.cos(state[..., 7]) * dvds[..., 2] + torch.sin(state[..., 7]) * dvds[..., 3]) + self.omega_max * torch.abs(dvds[..., 7])
        ham += self.velocity*(torch.cos(state[..., 8]) * dvds[..., 4] + torch.sin(state[..., 8]) * dvds[..., 5]) + self.omega_max * torch.abs(dvds[..., 8])
        return ham

    def optimal_control(self, state, dvds):
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [
                0, 0, 
                -0.4, 0, 
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }
