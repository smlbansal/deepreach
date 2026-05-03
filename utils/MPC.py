import torch
from tqdm import tqdm
import math


class MPC:
    def __init__(self, dT, horizon, receding_horizon, num_samples, dynamics_, device, mode="MPC",
                 sample_mode="gaussian", lambda_=0.01, style="direct", num_iterative_refinement=1, nbr_action_repeats=1):
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.receding_horizon = receding_horizon
        self.dynamics_ = dynamics_

        self.dT = dT
        self.nbr_action_repeats = nbr_action_repeats
        self.dT_integration = self.dT / self.nbr_action_repeats

        self.lambda_ = lambda_

        self.mode = mode
        self.sample_mode = sample_mode
        self.style = style  # choice: receding, direct
        self.num_iterative_refinement = num_iterative_refinement
        self.num_effective_horizon_refinement = 0

    def get_batch_data(self, initial_condition_tensor, T, policy=None, t=0.0):
        '''
        Generate MPC dataset in a batch manner
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                T: MPC total horizon
                t: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 
                costs: best cost function for all initial_condition_tensor (A)
                state_trajs: best trajs for all initial_condition_tensor (A * Horizon * state_dim)
                coords: bootstrapped MPC coords after normalization (coords=[time,state])  (? * (state_dim+1))
                value_labels: bootstrapped MPC value labels  (?)
        '''
        self.T = T*1.0
        self.batch_size = initial_condition_tensor.shape[0]
        if self.dynamics_.set_mode in ['avoid', 'reach']:
            state_trajs, lxs, num_iters = self.get_opt_trajs(
                initial_condition_tensor, policy, t)
            costs, _ = torch.min(lxs, dim=-1)

        elif self.dynamics_.set_mode == 'reach_avoid':
            state_trajs, avoid_values, reach_values, num_iters = self.get_opt_trajs(
                initial_condition_tensor, policy, t)
            costs = torch.min(torch.maximum(
                reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values
        else:
            raise NotImplementedError

        # generating MPC dataset: {..., (t, x, J, u), ...} NEW
        coords = torch.empty(0, self.dynamics_.state_dim+1).to(self.device)
        value_labels = torch.empty(0).to(self.device)
        # bootstrapping will be accurate up until the min l(x) occur
        if self.dynamics_.set_mode in ['avoid', 'reach']:
            _, min_idx = torch.min(lxs, dim=-1)
        elif self.dynamics_.set_mode == 'reach_avoid':
            _, min_idx = torch.min(torch.clamp(
                reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)), dim=-1)
        for i in range(num_iters):
            coord_i = torch.zeros(
                self.batch_size, self.dynamics_.state_dim+1).to(self.device)
            coord_i[:, 0] = self.T - i * self.dT
            coord_i[:, 1:] = state_trajs[:, i, :]*1.0
            if self.dynamics_.set_mode in ['avoid', 'reach']:
                valid_idx = (min_idx > i).nonzero(as_tuple=True)
                value_labels_i = torch.min(
                    lxs[valid_idx[0], i:], dim=-1).values
                coord_i = coord_i[valid_idx]
            elif self.dynamics_.set_mode == 'reach_avoid':
                valid_idx = (min_idx > i).nonzero(as_tuple=True)
                value_labels_i = torch.min(torch.clamp(reach_values[valid_idx[0], i:], min=torch.max(
                    -avoid_values[valid_idx[0], i:], dim=-1).values.unsqueeze(-1)), dim=-1).values
                coord_i = coord_i[valid_idx]
            else:
                raise NotImplementedError
            # add to data
            coords = torch.cat((coords, coord_i), dim=0)
            value_labels = torch.cat((value_labels, value_labels_i), dim=0)

        ##################### only use in range labels ###################################################
        output1 = torch.all(coords[..., 1:] >= self.dynamics_.state_range_[
                            :, 0]-0.01, -1, keepdim=False)
        output2 = torch.all(coords[..., 1:] <= self.dynamics_.state_range_[
                            :, 1]+0.01, -1, keepdim=False)
        in_range_index = torch.logical_and(torch.logical_and(
            output1, output2), ~torch.isnan(value_labels))

        coords = coords[in_range_index]
        value_labels = value_labels[in_range_index]
        ###################################################################################################
        coords = self.dynamics_.coord_to_input(coords)

        return costs, state_trajs, coords.detach().cpu().clone(), value_labels.detach().cpu().clone()

    def get_opt_trajs(self, initial_condition_tensor, policy=None, t=0.0):
        '''
        Generate optimal trajs in a batch manner
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                t: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 

                best_trajs: best trajs for all initial_condition_tensor (A * Horizon * state_dim)
                lxs: l(x) along best trajs (A*H)
                num_iters: H 
        '''
        num_iters = math.ceil((self.T)/self.dT)
        self.horizon = math.ceil((self.T)/self.dT)

        self.incremental_horizon = math.ceil((self.T-t)/self.dT)
        if self.style == 'direct':

            self.init_control_tensors()
            if policy is not None:
                self.num_effective_horizon_refinement = int(
                    self.num_iterative_refinement*0.4)
                for i in range(self.num_effective_horizon_refinement):
                    # optimize on the effective horizon first
                    self.warm_start_with_policy(
                        initial_condition_tensor, policy, t)
            # optimize on the entire horizon for stability (in case that the current learned value function is not accurate)
            best_controls, best_trajs = self.get_control(
                initial_condition_tensor, self.num_iterative_refinement, policy, t_remaining=t)

            if self.dynamics_.set_mode in ['avoid', 'reach']:
                lxs = self.dynamics_.boundary_fn(best_trajs)
                return best_trajs, lxs, num_iters
            elif self.dynamics_.set_mode == 'reach_avoid':
                avoid_values = self.dynamics_.avoid_fn(best_trajs)
                reach_values = self.dynamics_.reach_fn(best_trajs)
                return best_trajs, avoid_values, reach_values, num_iters
            else:
                raise NotImplementedError

        elif self.style == 'receding':
            if self.dynamics_.set_mode == 'reach_avoid':
                raise NotImplementedError

            state_trajs = torch.zeros(
                (self.batch_size, num_iters+1, self.dynamics_.state_dim)).to(self.device)  # A*H*D
            state_trajs[:, 0, :] = initial_condition_tensor

            self.init_control_tensors()
            if policy is not None:
                self.num_effective_horizon_refinement = int(
                    self.num_iterative_refinement*0.4)
                for i in range(self.num_effective_horizon_refinement):
                    self.warm_start_with_policy(
                        initial_condition_tensor, policy, t)
            lxs = torch.zeros(self.batch_size, num_iters+1).to(self.device)

            for i in tqdm(range(int(num_iters/self.receding_horizon))):
                best_controls, _ = self.get_control(
                    state_trajs[:, i, :])
                for k in range(self.receding_horizon):
                    lxs[:, i*self.receding_horizon+k] = self.dynamics_.boundary_fn(
                        state_trajs[:, i*self.receding_horizon+k, :])
                    state_trajs[:, i*self.receding_horizon+1+k, :] = self.get_next_step_state(
                        state_trajs[:, i*self.receding_horizon+k, :], best_controls[:, k, :])
                    self.receiding_start += 1
            lxs[:, -1] = self.dynamics_.boundary_fn(state_trajs[:, -1, :])
            return state_trajs, lxs, num_iters
        else:
            return NotImplementedError

    def warm_start_with_policy(self, initial_condition_tensor, policy=None, t_remaining=None):
        '''
        Generate optimal trajs in a batch manner using the DeepReach value function as the terminal cost
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                t_remaining: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 
                None
                Internally update self.control_tensors (first H_R horizon with MPC and t_remaining with DeepReach policy)
                Internally update self.warm_start_traj (for debugging purpose)
        '''
        if self.incremental_horizon > 0:
            # Rollout with the incremental horizon
            state_trajs_H, permuted_controls_H = self.rollout_dynamics(
                initial_condition_tensor, start_iter=0, rollout_horizon=self.incremental_horizon)

            costs = self.dynamics_.cost_fn(state_trajs_H)  # A * N
            # Use the learned value function for terminal cost and compute the cost function
            if t_remaining > 0.0:
                traj_times = torch.ones(self.batch_size, self.num_samples, 1).to(
                    self.device)*t_remaining
                state_trajs_clamped = torch.clamp(state_trajs_H[:, :, -1, :], torch.tensor(self.dynamics_.state_test_range(
                )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])

                traj_coords = torch.cat(
                    (traj_times, state_trajs_clamped), dim=-1)
                traj_policy_results = policy(
                    {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
                terminal_values = self.dynamics_.io_to_value(traj_policy_results['model_in'].detach(
                ), traj_policy_results['model_out'].squeeze(dim=-1).detach())
                if self.incremental_horizon > 0:
                    costs = torch.minimum(costs, terminal_values)
                    if self.dynamics_.set_mode == 'reach_avoid':
                        avoid_value_max = torch.max(
                            -self.dynamics_.avoid_fn(state_trajs_H), dim=-1).values
                        costs = torch.maximum(costs, avoid_value_max)
                        # TODO: Fix the cost function computation for receding horizon MPC
                else:
                    costs = terminal_values*1.0
            # Pick the best control and correponding traj
            if self.dynamics_.set_mode == 'avoid':
                best_costs, best_idx = costs.max(1)
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                best_costs, best_idx = costs.min(1)
            else:
                raise NotImplementedError
            expanded_idx = best_idx[..., None, None, None].expand(
                -1, -1, permuted_controls_H.size(2), permuted_controls_H.size(3))

            best_controls_H = torch.gather(
                permuted_controls_H, dim=1, index=expanded_idx).squeeze(1)  # A * H * D_u
            expanded_idx_traj = best_idx[..., None, None, None].expand(
                -1, -1, state_trajs_H.size(2), state_trajs_H.size(3))
            best_traj_H = torch.gather(
                state_trajs_H, dim=1, index=expanded_idx_traj).squeeze(1)

            # Rollout the remaining horizon with the learned policy and update the nominal control traj
            self.control_tensors[:, :self.incremental_horizon,
                                 :] = best_controls_H*1.0
            self.warm_start_traj = self.rollout_with_policy(
                best_traj_H[:, -1, :], policy, self.horizon-self.incremental_horizon, self.incremental_horizon)
            self.warm_start_traj = torch.cat(
                [best_traj_H[:, :-1, :], self.warm_start_traj], dim=1)

        else:
            # Rollout using the learned policy and update the nominal control traj
            self.warm_start_traj = self.rollout_with_policy(
                initial_condition_tensor, policy, self.horizon)

    def get_control(self, initial_condition_tensor, num_iterative_refinement=1, policy=None, t_remaining=None):
        '''
        Update self.control_tensors using perturbations
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                num_iterative_refinement: number of iterative improvement steps (re-sampling steps) in MPC
                t_remaining: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        '''

        if self.style == 'direct':
            if num_iterative_refinement == -1:  # rollout using the policy
                best_traj = self.rollout_with_policy(
                    initial_condition_tensor, policy, self.horizon)
            for i in range(num_iterative_refinement+1 - self.num_effective_horizon_refinement):
                state_trajs, permuted_controls = self.rollout_dynamics(
                    initial_condition_tensor, start_iter=0, rollout_horizon=self.horizon)
                self.all_state_trajs = state_trajs.detach().cpu()*1.0
                _, best_traj, best_costs = self.update_control_tensor(
                    state_trajs, permuted_controls)
            return self.control_tensors, best_traj
        elif self.style == 'receding':
            # initial_condition_tensor: A*D
            state_trajs, permuted_controls = self.rollout_dynamics(
                initial_condition_tensor, start_iter=self.receiding_start, rollout_horizon=self.horizon-self.receiding_start)

            current_controls, best_traj, _ = self.update_control_tensor(
                state_trajs, permuted_controls)

            return current_controls, best_traj

    def rollout_with_policy(self, initial_condition_tensor, policy, policy_horizon, policy_start_iter=0):
        '''
        Rollout traj with policy and update self.control_tensors (nominal control)
        Inputs: initial_condition_tensor A*D_N (Batch size * State dim)
                policy: Current DeepReach model
                policy_horizon: num steps correpond to t_remaining
                policy_start_iter: step num correpond to H_R
        '''
        state_trajs = torch.zeros(
            (self.batch_size, policy_horizon+1, self.dynamics_.state_dim))  # A * H * D
        # Move to GPU only when needed
        state_trajs = state_trajs.to(self.device, non_blocking=True)
        state_trajs[:, 0, :] = initial_condition_tensor*1.0
        state_trajs_clamped = state_trajs*1.0
        traj_times = torch.ones(self.batch_size, 1).to(
            self.device)*policy_horizon*self.dT
        # update control from policy_start_iter to policy_start_iter+ policy horizon
        for k in range(policy_horizon):

            traj_coords = torch.cat(
                (traj_times, state_trajs_clamped[:, k, :]), dim=-1)
            traj_policy_results = policy(
                {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
            traj_dvs = self.dynamics_.io_to_dv(
                traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

            self.control_tensors[:, k+policy_start_iter, :] = self.dynamics_.optimal_control(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device))
            self.control_tensors[:, k+policy_start_iter, :] = self.dynamics_.clamp_control(
                state_trajs[:, k, :], self.control_tensors[:, k+policy_start_iter, :])
            state_trajs[:, k+1, :] = self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k+policy_start_iter, :])

            state_trajs_clamped[:, k+1, :] = torch.clamp(state_trajs[:, k+1, :], torch.tensor(self.dynamics_.state_test_range(
            )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])
            traj_times = traj_times-self.dT
        return state_trajs

    def update_control_tensor(self, state_trajs, permuted_controls):
        '''
        Determine nominal controls (self.control_tensors) using permuted_controls and corresponding state trajs
        Inputs: 
                state_trajs: A*N*H*D_N (Batch size * Num perturbation * Horizon * State dim)
                permuted_controls: A*N*H*D_U (Batch size * Num perturbation * Horizon * Control dim)
        '''
        costs = self.dynamics_.cost_fn(state_trajs)  # A * N

        if self.mode == "MPC":
            # just use the best control
            if self.dynamics_.set_mode == 'avoid':
                best_costs, best_idx = costs.max(1)
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                best_costs, best_idx = costs.min(1)
            else:
                raise NotImplementedError
            expanded_idx = best_idx[..., None, None, None].expand(
                -1, -1, permuted_controls.size(2), permuted_controls.size(3))

            best_controls = torch.gather(
                permuted_controls, dim=1, index=expanded_idx).squeeze(1)  # A * H * D_u
            if self.style == 'direct':
                self.control_tensors = best_controls*1.0
            elif self.style == 'receding':
                self.control_tensors[:, self.receiding_start:,
                                     :] = best_controls*1.0
            else:
                raise NotImplementedError
            expanded_idx_traj = best_idx[..., None, None, None].expand(
                -1, -1, state_trajs.size(2), state_trajs.size(3))
            best_traj = torch.gather(
                state_trajs, dim=1, index=expanded_idx_traj).squeeze(1)
        elif self.mode == "MPPI":
            # use weighted average
            if self.dynamics_.set_mode == 'avoid':
                exp_terms = torch.exp((1/self.lambda_)*costs)  # A * N
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                exp_terms = torch.exp((1/self.lambda_)*-costs)  # A * N
            else:
                raise NotImplementedError

            den = torch.sum(exp_terms, dim=-1)  # A

            num = torch.sum(exp_terms[:, :, None, None].repeat(
                1, 1, self.horizon, self.dynamics_.control_dim) * permuted_controls, dim=1)  # A * H * D_u

            self.control_tensors = num/den[:, None, None]

            self.control_tensors = torch.clamp(
                self.control_tensors, self.dynamics_.control_range_[..., 0], self.dynamics_.control_range_[..., 1])
        else:
            raise NotImplementedError
        # update controls

        current_controls = self.control_tensors[:,
                                                self.receiding_start:self.receiding_start+self.receding_horizon, :]

        return current_controls, best_traj, best_costs

    def rollout_nominal_trajs(self, initial_state_tensor):
        '''
        Rollout trajs with nominal controls (self.control_tensors)
        '''
        # rollout trajs
        state_trajs = torch.zeros(
            (self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
        state_trajs[:, 0, :] = initial_state_tensor*1.0  # A * D

        for k in range(self.horizon):

            state_trajs[:, k+1, :] = self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k, :])
        return state_trajs

    def rollout_dynamics(self, initial_state_tensor, start_iter, rollout_horizon, eps_var_factor=1):
        '''
        Rollout trajs by generating perturbed controls
        Inputs: 
                initial_state_tensor A*D_N (Batch size * State dim)
                start_iter: from which step we start rolling out
                rollout_horizon: rollout for how many steps
                eps_var_factor: scaling factor for the sample variance (not being used in the paper but can be tuned if needed)
        Outputs: 
                state_trajs: A*N*H*D_N (Batch size * Num perturbation * Horizon * State dim)
                permuted_controls: A*N*H*D_U (Batch size * Num perturbation * Horizon * Control dim)
        '''
        # returns the state trajectory list and swith collision
        if self.sample_mode == "gaussian":
            epsilon_tensor = torch.randn(
                self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.control_dim).to(self.device)*torch.sqrt(self.dynamics_.eps_var_control if hasattr(self.dynamics_, 'eps_var_control') else self.dynamics_.eps_var)*eps_var_factor  # A * N * H * D_u

            # always include the nominal trajectory
            epsilon_tensor[:, 0, ...] = 0.0

            permuted_controls = self.control_tensors[:, start_iter:start_iter+rollout_horizon, :].unsqueeze(1).repeat(1,
                                                                                                                      self.num_samples, 1, 1) + epsilon_tensor * 1.0  # A * N * H * D_u
        elif self.sample_mode == "binary":
            permuted_controls = torch.sign(torch.empty(
                self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.control_dim).uniform_(-1, 1)).to(self.device)
            # always include the nominal trajectory
            permuted_controls[:, 0, ...] = self.control_tensors[:,
                                                                start_iter:start_iter+rollout_horizon, :]*1.0

        # clamp control
        permuted_controls = torch.clamp(permuted_controls, self.dynamics_.control_range_[
                                        ..., 0], self.dynamics_.control_range_[..., 1])

        # rollout trajs
        state_trajs = torch.zeros((self.batch_size, self.num_samples, rollout_horizon+1,
                                  self.dynamics_.state_dim)).to(self.device)  # A * N * H * D
        state_trajs[:, :, 0, :] = initial_state_tensor.unsqueeze(
            1).repeat(1, self.num_samples, 1)  # A * N * D

        for k in range(rollout_horizon):
            permuted_controls[:, :, k, :] = self.dynamics_.clamp_control(
                state_trajs[:, :, k, :], permuted_controls[:, :, k, :])
            state_trajs[:, :, k+1, :] = self.get_next_step_state(
                state_trajs[:, :, k, :], permuted_controls[:, :, k, :])

        return state_trajs, permuted_controls

    def init_control_tensors(self):
        self.receiding_start = 0
        self.control_init = self.dynamics_.control_init.unsqueeze(
            0).repeat(self.batch_size, 1)
        self.control_tensors = self.control_init.unsqueeze(
            1).repeat(1, self.horizon, 1)  # A * H * D_u

    def get_next_step_state(self, state, controls):
        curr_state = state
        for i in range(self.nbr_action_repeats):
            current_dsdt = self.dynamics_.dsdt(curr_state, controls, None)
            next_state = self.dynamics_.equivalent_wrapped_state(
                curr_state + current_dsdt * self.dT_integration)
            curr_state = next_state
        return curr_state


class RobustMPC(MPC):
    def __init__(
        self,
        dT,
        horizon,
        receding_horizon,
        num_samples,
        dynamics_,
        device,
        mode="MPC",
        control_sample_mode="gaussian",
        disturbance_sample_mode="policy",
        lambda_=0.01,
        style="direct",
        num_iterative_refinement=1,
        integration_method="euler",
        nbr_action_repeats=1,
    ):
        self.dT = dT
        self.nbr_action_repeats = nbr_action_repeats
        self.dT_integration = self.dT / self.nbr_action_repeats
        self.horizon = horizon
        self.receding_horizon = receding_horizon
        self.num_samples = num_samples
        self.device = device
        self.dynamics_ = dynamics_
        self.policy_dynamics_ = dynamics_

        self.integration_method = integration_method

        self.mode = mode
        self.control_sample_mode = control_sample_mode
        self.disturbance_sample_mode = disturbance_sample_mode

        self.lambda_ = lambda_

        self.style = style
        self.num_iterative_refinement = num_iterative_refinement
        self.num_effective_horizon_refinement = 0

    def get_batch_data(self, init_state, T, policy=None, t=0.0):
        return super().get_batch_data(init_state, T, policy=policy, t=t)

    def get_opt_trajs(self, init_state, policy=None, t=0.0):
        num_iters = math.ceil(self.T / self.dT)
        self.horizon = num_iters
        self.incremental_horizon = math.ceil((self.T - t) / self.dT)

        if self.style == "direct":
            self.init_tensors()
            if policy is not None:
                self.num_effective_horizon_refinement = int(self.num_iterative_refinement * 0.4)
                for i in range(self.num_effective_horizon_refinement):
                    self.warm_start_with_policy(init_state, policy, t)
            best_controls, best_dists, best_trajs = self.get_control_and_disturbance(
                init_state, self.num_iterative_refinement, policy, t_remaining=t
            )

            if self.dynamics_.set_mode in ["avoid", "reach"]:
                lxs = self.dynamics_.boundary_fn(best_trajs)
                return best_trajs, lxs, num_iters
            elif self.dynamics_.set_mode == "reach_avoid":
                avoid_values = self.dynamics_.avoid_fn(best_trajs)
                reach_values = self.dynamics_.reach_fn(best_trajs)
                return best_trajs, avoid_values, reach_values, num_iters
            else:
                raise NotImplementedError

    def warm_start_with_policy(self, init_state, policy=None, t_remaining=None):
        if self.incremental_horizon > 0:
            state_trajs_H, controls_H, disturbances_H = self.rollout_dynamics(
                init_state,
                start_iter=0,
                rollout_horizon=self.incremental_horizon,
                policy=policy,
            )

            costs = self.dynamics_.cost_fn(state_trajs_H)
            if t_remaining > 0.0:
                traj_times = (
                    torch.ones(self.batch_size, self.num_samples, 1).to(self.device) * t_remaining
                )
                final_state = state_trajs_H[:, :, -1, :]
                final_state_bounded = self.dynamics_.clip_state(final_state)
                state_coords = torch.cat((traj_times, final_state_bounded), dim=-1)
                results = policy(
                    {"coords": self.policy_dynamics_.coord_to_input(state_coords.to(self.device))}
                )
                terminal_values = self.policy_dynamics_.io_to_value(
                    results["model_in"].detach(),
                    results["model_out"].squeeze(dim=-1).detach(),
                )
                if self.incremental_horizon > 0:
                    costs = torch.minimum(costs, terminal_values)
                else:
                    costs = terminal_values * 1.0

            if self.dynamics_.set_mode == "avoid":
                if self.disturbance_sample_mode == "policy":
                    best_costs, best_idx = costs.max(1)
                else:
                    best_costs, best_idx = costs.min(1)
            else:
                if self.disturbance_sample_mode == "policy":
                    best_costs, best_idx = costs.min(1)
                else:
                    best_costs, best_idx = costs.max(1)

            H = controls_H.size(2)
            idx_controls = best_idx[..., None, None, None].expand(-1, -1, H, self.dynamics_.control_dim)
            idx_disturbances = best_idx[..., None, None, None].expand(-1, -1, H, self.dynamics_.disturbance_dim)
            idx_trajs = best_idx[..., None, None, None].expand(-1, -1, H + 1, self.dynamics_.state_dim)

            if self.disturbance_sample_mode == "policy":
                best_controls = torch.gather(controls_H, dim=1, index=idx_controls).squeeze(1)
                best_disturbances = torch.gather(disturbances_H, dim=1, index=idx_disturbances).squeeze(1)
                self.control_tensors[:, : self.incremental_horizon, :] = best_controls * 1.0
                self.disturbance_tensors[:, : self.incremental_horizon, :] = best_disturbances * 1.0
            else:
                best_disturbances = torch.gather(disturbances_H, dim=1, index=idx_disturbances).squeeze(1)
                self.disturbance_tensors[:, : self.incremental_horizon, :] = best_disturbances * 1.0
                best_controls = torch.gather(controls_H, dim=1, index=idx_controls).squeeze(1)
                self.control_tensors[:, : self.incremental_horizon, :] = best_controls * 1.0

            best_traj_H = torch.gather(state_trajs_H, dim=1, index=idx_trajs).squeeze(1)
            remaining_traj = self.rollout_with_policy(
                best_traj_H[:, -1, :],
                policy,
                policy_horizon=self.horizon - self.incremental_horizon,
                policy_start_iter=self.incremental_horizon,
            )
            self.warm_start_traj = torch.cat([best_traj_H[:, :-1, :], remaining_traj], dim=1)

        else:
            self.warm_start_traj = self.rollout_with_policy(
                init_state, policy, policy_horizon=self.horizon
            )

    def get_control_and_disturbance(
        self, init_state, num_iterative_refinement=1, policy=None, t_remaining=None
    ):
        assert self.style == "direct"
        if num_iterative_refinement == -1:
            best_traj = self.rollout_with_policy(init_state, policy, self.horizon)
        for i in range(num_iterative_refinement + 1 - self.num_effective_horizon_refinement):
            state_trajs, controls, disturbances = self.rollout_dynamics(
                init_state, start_iter=0, rollout_horizon=self.horizon, policy=policy
            )
            _, _, best_traj, best_costs = self.update_tensors(state_trajs, controls, disturbances)
        return self.control_tensors, self.disturbance_tensors, best_traj

    def rollout_with_policy(self, init_state, policy, policy_horizon, policy_start_iter=0):
        state_trajs = torch.zeros(
            (self.batch_size, policy_horizon + 1, self.dynamics_.state_dim)
        )
        state_trajs = state_trajs.to(self.device, non_blocking=True)
        state_trajs[:, 0, :] = init_state * 1.0
        traj_times = torch.ones(self.batch_size, 1).to(self.device) * policy_horizon * self.dT
        for k in range(policy_horizon):
            traj_coords = torch.cat(
                (traj_times, self.dynamics_.clip_state(state_trajs[:, k, :])), dim=-1
            )
            traj_policy_results = policy(
                {"coords": self.policy_dynamics_.coord_to_input(traj_coords.to(self.device))}
            )
            traj_dvs = self.policy_dynamics_.io_to_dv(
                traj_policy_results["model_in"],
                traj_policy_results["model_out"].squeeze(dim=-1),
            ).detach()

            self.control_tensors[:, k + policy_start_iter, :] = self.dynamics_.optimal_control(
                traj_coords[:, 1:].to(self.device),
                traj_dvs[..., 1:].to(self.device),
            )
            self.control_tensors[:, k + policy_start_iter, :] = self.dynamics_.clamp_control(
                state_trajs[:, k, :],
                self.control_tensors[:, k + policy_start_iter, :],
            )

            dist = self.dynamics_.optimal_disturbance(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device)
            )
            self.disturbance_tensors[:, k + policy_start_iter, :] = (
                self.dynamics_.clamp_disturbance(state_trajs[:, k, :], dist)
            )
            state_trajs[:, k + 1, :] = self.get_next_step_state(
                state_trajs[:, k, :],
                self.control_tensors[:, k + policy_start_iter, :],
                self.disturbance_tensors[:, k + policy_start_iter, :],
            )
            traj_times = traj_times - self.dT
        return state_trajs

    def update_tensors(self, state_trajs, controls, disturbances):
        costs = self.dynamics_.cost_fn(state_trajs)

        if self.mode == "MPC":
            if self.dynamics_.set_mode == "avoid":
                if self.disturbance_sample_mode == "policy":
                    best_costs, best_idx = costs.max(1)
                else:
                    best_costs, best_idx = costs.min(1)
            else:
                if self.disturbance_sample_mode == "policy":
                    best_costs, best_idx = costs.min(1)
                else:
                    best_costs, best_idx = costs.max(1)

            assert self.style == "direct"
            H = controls.size(2)
            idx_controls = best_idx[..., None, None, None].expand(-1, -1, H, self.dynamics_.control_dim)
            idx_disturbances = best_idx[..., None, None, None].expand(-1, -1, H, self.dynamics_.disturbance_dim)
            idx_trajs = best_idx[..., None, None, None].expand(-1, -1, H + 1, self.dynamics_.state_dim)

            if self.disturbance_sample_mode == "policy":
                best_controls = torch.gather(controls, dim=1, index=idx_controls).squeeze(1)
                self.control_tensors = best_controls * 1.0
                best_disturbances = torch.gather(disturbances, dim=1, index=idx_disturbances).squeeze(1)
                self.disturbance_tensors = best_disturbances * 1.0
            else:
                best_disturbances = torch.gather(disturbances, dim=1, index=idx_disturbances).squeeze(1)
                self.disturbance_tensors = best_disturbances * 1.0
                best_controls = torch.gather(controls, dim=1, index=idx_controls).squeeze(1)
                self.control_tensors = best_controls * 1.0

            best_traj = torch.gather(state_trajs, dim=1, index=idx_trajs).squeeze(1)

        elif self.mode == "MPPI":
            raise NotImplementedError
        else:
            raise NotImplementedError

        current_controls = self.control_tensors[
            :, self.receiding_start : self.receiding_start + self.receding_horizon, :
        ]
        current_dists = self.disturbance_tensors[
            :, self.receiding_start : self.receiding_start + self.receding_horizon, :
        ]
        return current_controls, current_dists, best_traj, best_costs

    def rollout_nominal_trajs(self, init_state):
        state_trajs = torch.zeros(self.batch_size, self.horizon + 1, self.dynamics_.state_dim)
        state_trajs[:, 0, :] = init_state
        for k in range(self.horizon):
            state_trajs[:, k + 1, :] = self.get_next_step_state(
                state_trajs[:, k, :],
                self.control_tensors[:, k, :],
                self.disturbance_tensors[:, k, :],
            )
        return state_trajs

    def rollout_dynamics(self, init_state, start_iter, rollout_horizon, eps_var_factor=1, policy=None):
        assert self.control_sample_mode == "policy" or self.disturbance_sample_mode == "policy"
        controls = torch.zeros(
            (self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.control_dim)
        ).to(self.device)
        disturbances = torch.zeros(
            (self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.disturbance_dim)
        ).to(self.device)

        if self.control_sample_mode == "gaussian":
            control_randn = torch.randn(
                self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.control_dim
            ).to(self.device)
            _eps_ctrl = self.dynamics_.eps_var_control if hasattr(self.dynamics_, 'eps_var_control') else self.dynamics_.eps_var
            eps_tensor_control = control_randn * torch.sqrt(_eps_ctrl) * eps_var_factor
            eps_tensor_control[:, 0, ...] = 0.0
            end_iter = start_iter + rollout_horizon
            offset = (
                self.control_tensors[:, start_iter:end_iter, :]
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1, 1)
            )
            controls = offset + eps_tensor_control
            controls = self.dynamics_.bound_control(controls)
        elif self.disturbance_sample_mode == "gaussian":
            dist_randn = torch.randn(
                self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.disturbance_dim
            ).to(self.device)
            _eps_dist = (self.dynamics_.eps_var_disturbance if hasattr(self.dynamics_, 'eps_var_disturbance')
                         else (self.dynamics_.eps_var_control if hasattr(self.dynamics_, 'eps_var_control')
                               else self.dynamics_.eps_var))
            eps_tensor_dist = dist_randn * torch.sqrt(_eps_dist) * eps_var_factor
            eps_tensor_dist[:, 0, ...] = 0.0
            end_iter = start_iter + rollout_horizon
            offset = (
                self.disturbance_tensors[:, start_iter:end_iter, :]
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1, 1)
            )
            disturbances = offset + eps_tensor_dist
            disturbances = self.dynamics_.bound_disturbance(disturbances)

        state_trajs = torch.zeros(
            (self.batch_size, self.num_samples, rollout_horizon + 1, self.dynamics_.state_dim)
        ).to(self.device)
        state_trajs[:, :, 0, :] = init_state.unsqueeze(1).repeat(1, self.num_samples, 1)

        traj_times = (
            torch.ones(self.batch_size, self.num_samples, 1).to(self.device)
            * rollout_horizon
            * self.dT
        )
        for k in range(rollout_horizon):
            traj_coords = torch.cat(
                (traj_times, self.dynamics_.clip_state(state_trajs[:, :, k, :])), dim=-1
            )
            traj_policy_results = policy(
                {"coords": self.policy_dynamics_.coord_to_input(traj_coords.to(self.device))}
            )
            traj_dvs = self.policy_dynamics_.io_to_dv(
                traj_policy_results["model_in"],
                traj_policy_results["model_out"].squeeze(dim=-1),
            ).detach()
            if self.disturbance_sample_mode == "policy":
                dist = self.policy_dynamics_.optimal_disturbance(
                    traj_coords[..., 1:].to(self.device),
                    traj_dvs[..., 1:].to(self.device),
                )
                disturbances[:, :, k, :] = self.policy_dynamics_.clamp_disturbance(
                    state_trajs[:, :, k, :], dist
                )
            if self.control_sample_mode == "policy":
                control = self.policy_dynamics_.optimal_control(
                    traj_coords[..., 1:].to(self.device),
                    traj_dvs[..., 1:].to(self.device),
                )
                controls[:, :, k, :] = self.policy_dynamics_.clamp_control(
                    state_trajs[:, :, k, :], control
                )
            state_trajs[:, :, k + 1, :] = self.get_next_step_state(
                state_trajs[:, :, k, :], controls[:, :, k, :], disturbances[:, :, k, :]
            )
            traj_times = traj_times - self.dT

        return state_trajs, controls, disturbances

    def init_tensors(self):
        self.receiding_start = 0
        self.control_init = self.dynamics_.control_init.unsqueeze(0).repeat(self.batch_size, 1)
        self.control_tensors = self.control_init.unsqueeze(1).repeat(1, self.horizon, 1)
        self.disturbance_init = self.dynamics_.disturbance_init.unsqueeze(0).repeat(self.batch_size, 1)
        self.disturbance_tensors = self.disturbance_init.unsqueeze(1).repeat(1, self.horizon, 1)

    def get_next_step_state(self, state, control, disturbance):
        if self.integration_method == "rk4":
            return self.get_next_step_state_rk4(state, control, disturbance)
        curr_state = state
        for i in range(self.nbr_action_repeats):
            current_dsdt = self.dynamics_.dsdt(curr_state, control, disturbance)
            next_state = self.dynamics_.equivalent_wrapped_state(
                curr_state + current_dsdt * self.dT_integration
            )
            curr_state = next_state
        return curr_state

    def get_next_step_state_rk4(self, state, control, disturbance):
        h = self.dT
        k1 = self.dynamics_.dsdt(state, control, disturbance)
        state_k2 = state + 0.5 * h * k1
        k2 = self.dynamics_.dsdt(state_k2, control, disturbance)
        state_k3 = state + 0.5 * h * k2
        k3 = self.dynamics_.dsdt(state_k3, control, disturbance)
        state_k4 = state + h * k3
        k4 = self.dynamics_.dsdt(state_k4, control, disturbance)
        next_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.dynamics_.equivalent_wrapped_state(next_state)
