import torch
from torch.utils.data import Dataset
import numpy as np
import math
from utils import MPC
import os
from tqdm import tqdm
import random
from bisect import bisect_right
import gc

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
            None, ...
        ].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[0], : sidelen[1], : sidelen[2]], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError("Not implemented for dim=%d" % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(
        self,
        dynamics,
        numpoints,
        pretrain,
        pretrain_iters,
        tMin,
        tMax,
        counter_start,
        counter_end,
        num_src_samples,
        num_target_samples,
        use_MPC,
        time_curr,
        MPC_data_path,
        num_MPC_perturbation_samples,
        MPC_dt,
        MPC_mode,
        MPC_sample_mode,
        MPC_style,
        MPC_lambda_,
        MPC_batch_size,
        MPC_receding_horizon,
        num_MPC_data_samples,
        num_iterative_refinement,
        time_till_refinement,
        refinement_schedule=None,
        num_MPC_batches=1,
        aug_with_MPC_data=0,
        policy=None,
        refine_dataset=True,
        initiate_with_MPC=True,
        add_disturbance_samples=False,
        MPC_integration_method="euler",
        MPC_nbr_action_repeats=1,
    ):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin
        self.tMax = tMax
        self.counter = counter_start
        self.counter_end = counter_end
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples

        self.use_MPC = use_MPC

        self.time_curr = time_curr
        self.num_MPC_data_samples = num_MPC_data_samples
        self.aug_with_MPC_data = aug_with_MPC_data
        self.MPC_batch_size = MPC_batch_size
        self.MPC_receding_horizon = MPC_receding_horizon
        self.MPC_dt = MPC_dt
        self.policy = policy
        self.num_MPC_batches = num_MPC_batches
        self.time_till_refinement = time_till_refinement
        self.MPC_data_path = MPC_data_path
        self.num_MPC_perturbation_samples = num_MPC_perturbation_samples
        self.MPC_mode = MPC_mode
        self.MPC_sample_mode = MPC_sample_mode

        self.epoch_till_refinement = 20000
        self.refine_dataset = refine_dataset
        self.MPC_lambda_ = MPC_lambda_
        self.MPC_style = MPC_style
        self.num_iterative_refinement = num_iterative_refinement
        self.refinement_schedule = refinement_schedule
        if refinement_schedule is not None:
            self.refinement_schedule_index = 0
        self.add_disturbance_samples = add_disturbance_samples
        self.MPC_integration_method = MPC_integration_method
        self.MPC_nbr_action_repeats = MPC_nbr_action_repeats

        if use_MPC and initiate_with_MPC:
            if MPC_data_path == "none":
                # initialize MPC dataset
                if refine_dataset:
                    if self.refinement_schedule is not None:
                        self.generate_MPC_dataset(self.refinement_schedule[0], 0.0)
                    else:
                        self.generate_MPC_dataset(self.time_till_refinement, 0.0)
                else:
                    self.generate_MPC_dataset(self.tMax, 0.0)

            else:
                self.MPC_inputs = torch.load(os.path.join(MPC_data_path, "inputs.pt"))
                self.MPC_values = torch.load(
                    os.path.join(MPC_data_path, "value_labels.pt")
                )
                self.mpc_time_sorted_indices = torch.argsort(self.MPC_inputs[:, 0])

    def get_next_refinement_time(self, current_time):
        if self.refinement_schedule is not None:
            for time in self.refinement_schedule:
                if time > current_time:
                    return time
            return None
        else:
            return current_time + self.time_till_refinement

    def use_terminal_MPC(self):
        # self.MPC_dt can be different
        self.MPC_dt = 0.005
        self.num_iterative_refinement = -1
        # self.MPC_batch_size=50000

    def __len__(self):
        return 1

    def sample_init_state(self):
        MPC_states = torch.empty((0, self.dynamics.state_dim))
        while MPC_states.shape[0] < self.MPC_batch_size:

            MPC_states_ = torch.zeros(
                self.MPC_batch_size, self.dynamics.state_dim
            ).uniform_(-1, 1)

            if self.dynamics.name == "Quadrotor":
                MPC_states_ = self.dynamics.normalize_q(MPC_states_)
            MPC_states = torch.cat(
                (MPC_states, self.dynamics.clamp_state_input(MPC_states_)), dim=0
            )
        MPC_states = MPC_states[: self.MPC_batch_size, ...] * 1.0
        MPC_states = self.dynamics.input_to_coord(
            torch.cat((torch.ones(MPC_states.shape[0], 1), MPC_states), dim=-1)
        )[..., 1:]

        if self.num_target_samples > 0:
            num_mpc_target_samples = min(
                int(self.MPC_batch_size / 5), self.num_target_samples
            )
            target_state_samples = self.dynamics.sample_target_state(
                num_mpc_target_samples
            )
            MPC_states[-num_mpc_target_samples:] = target_state_samples
        return MPC_states

    def collect_MPC_io(self, mpc_fn, T, t, style="random"):
        MPC_states = self.sample_init_state()
        _, _, MPC_inputs, MPC_values = mpc_fn.get_batch_data(
            MPC_states.to(device), T, self.policy, t=t
        )
        for i in tqdm(range(self.num_MPC_batches - 1)):
            MPC_states = self.sample_init_state()

            if mpc_fn.style == "direct":
                t_max = (
                    random.randint(1, math.floor((T * 1.1) / self.MPC_dt)) * self.MPC_dt
                )
            elif mpc_fn.style == "receding":
                t_max = (
                    random.randint(1, int(T / self.MPC_dt / mpc_fn.receding_horizon))
                    * mpc_fn.receding_horizon
                    * self.MPC_dt
                )
            else:
                raise NotImplementedError

            if style == "terminal" and i < self.num_MPC_batches / 2:
                t_max = T * 1.0
            _, _, MPC_inputs_, MPC_values_ = mpc_fn.get_batch_data(
                MPC_states.to(device), t_max, self.policy, t=t
            )
            MPC_inputs = torch.cat([MPC_inputs, MPC_inputs_], dim=0)
            MPC_values = torch.cat([MPC_values, MPC_values_], dim=0)

        if style == "terminal":
            MPC_inputs_ = MPC_inputs[MPC_inputs[:, 0] == T, ...]
            MPC_values_ = MPC_values[MPC_inputs[:, 0] == T, ...]
            idxs = torch.randperm(MPC_inputs[MPC_inputs[:, 0] < T].shape[0])[
                : int(MPC_inputs[MPC_inputs[:, 0] < T].shape[0] / 1.5)
            ]
            MPC_inputs = torch.cat([MPC_inputs[idxs, ...], MPC_inputs_], dim=0)
            MPC_values = torch.cat([MPC_values[idxs, ...], MPC_values_], dim=0)
        return MPC_inputs, MPC_values

    def generate_MPC_dataset(self, T, t, style="random"):
        print("Generating MPC dataset")
        mpc_fn = MPC.MPC(
            horizon=None,
            receding_horizon=self.MPC_receding_horizon,
            dT=self.MPC_dt,
            num_samples=self.num_MPC_perturbation_samples,
            dynamics_=self.dynamics,
            device=device,
            mode=self.MPC_mode,
            sample_mode=self.MPC_sample_mode,
            lambda_=self.MPC_lambda_,
            style=self.MPC_style,
            num_iterative_refinement=self.num_iterative_refinement,
            nbr_action_repeats=self.MPC_nbr_action_repeats,
        )
        MPC_inputs, MPC_values = self.collect_MPC_io(mpc_fn, T, t, style=style)

        # convert to memory-mapped tensor for faster query
        if not os.path.exists("./data"):
            os.makedirs("./data")
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        np.save(
            "./data/MPC_inputs_gpu%s.npy" % device_id,
            MPC_inputs.cpu().numpy(),
            allow_pickle=False,
        )
        np.save(
            "./data/MPC_values_gpu%s.npy" % device_id,
            MPC_values.cpu().numpy(),
            allow_pickle=False,
        )
        MPC_inputs_mmap = np.load(
            "./data/MPC_inputs_gpu%s.npy" % device_id, mmap_mode="r"
        )
        MPC_values_mmap = np.load(
            "./data/MPC_values_gpu%s.npy" % device_id, mmap_mode="r"
        )
        self.MPC_inputs = torch.from_numpy(MPC_inputs_mmap).detach()
        self.MPC_values = torch.from_numpy(MPC_values_mmap).detach()
        self.mpc_time_sorted_indices = torch.argsort(self.MPC_inputs[:, 0])
        print("Generated %d labels" % MPC_inputs.shape[0])

        del mpc_fn
        gc.collect()
        torch.cuda.empty_cache()

    def __getitem__(self, idx):
        # uniformly sample domain and include coordinates where source is non-zero
        model_states_normed = torch.empty((0, self.dynamics.state_dim))
        while model_states_normed.shape[0] < self.numpoints:
            model_states_normed_ = torch.zeros(
                self.numpoints, self.dynamics.state_dim
            ).uniform_(-1, 1)
            model_states_normed = torch.cat(
                (
                    model_states_normed,
                    self.dynamics.clamp_state_input(model_states_normed_),
                ),
                dim=0,
            )
        model_states_normed = model_states_normed[: self.numpoints, ...]

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((model_states_normed.shape[0], 1), self.tMin)
        else:
            # slowly grow time values from start time
            # make the curriculum slightly (1.1 times) longer to avoid innaccuracy on the boundary
            times = self.tMin + torch.zeros(model_states_normed.shape[0], 1).uniform_(
                0,
                (self.tMax * 1.1 - self.tMin)
                * min((self.counter + 1) / self.counter_end, 1.0),
            )

            # make sure we always have training samples at the initial time
            if self.dynamics.deepReach_model in ["vanilla", "diff"]:
                times[-self.num_src_samples :, 0] = self.tMin

        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(
                self.num_target_samples
            )
            model_states_normed[-self.num_target_samples :] = (
                self.dynamics.coord_to_input(
                    torch.cat(
                        (torch.zeros(self.num_target_samples, 1), target_state_samples),
                        dim=-1,
                    )
                )[:, 1 : self.dynamics.state_dim + 1]
            )
        model_inputs = torch.cat((times, model_states_normed), dim=1)

        # generating MPC inputs
        if self.use_MPC and hasattr(self, "MPC_inputs"):
            current_t = (self.tMax * 1.0 - self.tMin) * min(
                (self.counter) / self.counter_end, 1.0
            )
            if self.time_curr and not self.pretrain:
                if current_t > self.MPC_dt:
                    # Find upper bound index using precomputed sorted indices
                    max_idx = bisect_right(
                        self.MPC_inputs[self.mpc_time_sorted_indices][:, 0]
                        .cpu()
                        .numpy(),
                        current_t,
                    )

                    if max_idx >= self.num_MPC_data_samples:
                        idxs = torch.randint(0, max_idx, (self.num_MPC_data_samples,))
                        MPC_inputs_ = self.MPC_inputs[
                            self.mpc_time_sorted_indices[idxs]
                        ]
                        MPC_values_ = self.MPC_values[
                            self.mpc_time_sorted_indices[idxs]
                        ]
                    else:
                        MPC_inputs_ = self.MPC_inputs[
                            self.mpc_time_sorted_indices[:max_idx]
                        ]
                        MPC_values_ = self.MPC_values[
                            self.mpc_time_sorted_indices[:max_idx]
                        ]

                    # Augment with MPC data (around MPC datapoints)
                    if self.aug_with_MPC_data > 0:
                        num_available = min(max_idx, self.aug_with_MPC_data)
                        idxs = torch.randint(0, num_available, (num_available,))
                        state_around_MPC_inputs = (
                            self.MPC_inputs[self.mpc_time_sorted_indices[idxs]] * 1.0
                        )
                        state_around_MPC_inputs = torch.clip(
                            state_around_MPC_inputs
                            * (torch.randn_like(state_around_MPC_inputs) * 0.01 + 1.0),
                            min=-1.0,
                            max=1.0,
                        )
                        # state_around_MPC_inputs[...,1:]=self.dynamics.clamp_state_input(state_around_MPC_inputs[...,1:])
                        model_inputs[-num_available:, ...] = state_around_MPC_inputs

                else:
                    MPC_inputs_ = torch.Tensor([0])
                    MPC_values_ = torch.Tensor([0])

            else:
                idxs = torch.randint(
                    0, len(self.MPC_inputs), (self.num_MPC_data_samples,)
                )
                MPC_inputs_ = self.MPC_inputs[idxs]
                MPC_values_ = self.MPC_values[idxs]

                # Augment with MPC data
                if self.aug_with_MPC_data > 0 and not self.pretrain:
                    idxs = torch.randint(
                        0, len(self.MPC_inputs), (self.aug_with_MPC_data,)
                    )
                    model_inputs[-self.aug_with_MPC_data :, ...] = self.MPC_inputs[idxs]

        else:
            MPC_inputs_ = torch.Tensor([0])
            MPC_values_ = torch.Tensor([0])

        boundary_values = self.dynamics.boundary_fn(
            self.dynamics.input_to_coord(model_inputs)[..., 1:]
        )
        if self.dynamics.loss_type == "brat_hjivi":
            reach_values = self.dynamics.reach_fn(
                self.dynamics.input_to_coord(model_inputs)[..., 1:]
            )
            avoid_values = self.dynamics.avoid_fn(
                self.dynamics.input_to_coord(model_inputs)[..., 1:]
            )

        if self.pretrain:
            dirichlet_masks = torch.ones(model_inputs.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = model_inputs[:, 0] == self.tMin

        if self.pretrain:
            self.pretrain_counter += 1
        else:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.dynamics.loss_type == "brt_hjivi":
            return {"model_inputs": model_inputs, "MPC_inputs": MPC_inputs_}, {
                "boundary_values": boundary_values,
                "dirichlet_masks": dirichlet_masks,
                "MPC_values": MPC_values_,
            }
        elif self.dynamics.loss_type == "brat_hjivi":
            return {"model_inputs": model_inputs, "MPC_inputs": MPC_inputs_}, {
                "boundary_values": boundary_values,
                "reach_values": reach_values,
                "avoid_values": avoid_values,
                "dirichlet_masks": dirichlet_masks,
                "MPC_values": MPC_values_,
            }
        else:
            raise NotImplementedError


class RobustReachabilityDataset(ReachabilityDataset):
    def generate_MPC_dataset(self, T, t, style="random"):
        print("Generating robust MPC dataset")
        mpc_control_fn = MPC.RobustMPC(
            horizon=None,
            receding_horizon=self.MPC_receding_horizon,
            dT=self.MPC_dt,
            num_samples=self.num_MPC_perturbation_samples,
            dynamics_=self.dynamics,
            device=device,
            mode=self.MPC_mode,
            control_sample_mode=self.MPC_sample_mode,
            disturbance_sample_mode="policy",
            lambda_=self.MPC_lambda_,
            style=self.MPC_style,
            num_iterative_refinement=self.num_iterative_refinement,
            integration_method=self.MPC_integration_method,
            nbr_action_repeats=self.MPC_nbr_action_repeats,
        )
        mpc_input_control, MPC_values_control = self.collect_MPC_io(
            mpc_control_fn, T, t, style=style
        )

        if self.add_disturbance_samples:
            mpc_disturbance_fn = MPC.RobustMPC(
                horizon=None,
                receding_horizon=self.MPC_receding_horizon,
                dT=self.MPC_dt,
                num_samples=self.num_MPC_perturbation_samples,
                dynamics_=self.dynamics,
                device=device,
                mode=self.MPC_mode,
                control_sample_mode="policy",
                disturbance_sample_mode=self.MPC_sample_mode,
                lambda_=self.MPC_lambda_,
                style=self.MPC_style,
                num_iterative_refinement=self.num_iterative_refinement,
                integration_method=self.MPC_integration_method,
                nbr_action_repeats=self.MPC_nbr_action_repeats,
            )
            mpc_input_disturbance, MPC_values_disturbance = self.collect_MPC_io(
                mpc_disturbance_fn, T, t, style=style
            )
            MPC_inputs = torch.cat((mpc_input_control, mpc_input_disturbance), dim=0)
            MPC_values = torch.cat((MPC_values_control, MPC_values_disturbance), dim=0)
        else:
            MPC_inputs = mpc_input_control
            MPC_values = MPC_values_control

        if not os.path.exists("./data"):
            os.makedirs("./data")
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        np.save(
            "./data/MPC_inputs_gpu%s.npy" % device_id,
            MPC_inputs.cpu().numpy(),
            allow_pickle=False,
        )
        np.save(
            "./data/MPC_values_gpu%s.npy" % device_id,
            MPC_values.cpu().numpy(),
            allow_pickle=False,
        )
        MPC_inputs_mmap = np.load(
            "./data/MPC_inputs_gpu%s.npy" % device_id, mmap_mode="r"
        )
        MPC_values_mmap = np.load(
            "./data/MPC_values_gpu%s.npy" % device_id, mmap_mode="r"
        )
        self.MPC_inputs = torch.from_numpy(MPC_inputs_mmap).detach()
        self.MPC_values = torch.from_numpy(MPC_values_mmap).detach()
        self.mpc_time_sorted_indices = torch.argsort(self.MPC_inputs[:, 0])
        print("Generated %d labels" % MPC_inputs.shape[0])

        gc.collect()
        torch.cuda.empty_cache()
