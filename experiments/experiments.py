import wandb
import torch
import os
import shutil
import time
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.stats import beta as beta__dist
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
import seaborn as sns


def flatten_dict(d, parent_key='', sep='/'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        ## Dynamic Weighting
        self.loss_weights = {'dirichlet': 1., 'mpc_loss': 1., 'diff_constraint_hom': 1.}

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(
                self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path)['model'])
        else:
            model_path = os.path.join(
                self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])


    def train(
        self, batch_size, epochs, lr, steps_til_summary, epochs_til_checkpoint, loss_fn, clip_grad, use_lbfgs, adjust_relative_grads,
        val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution, MPC_importance_init, MPC_importance_final, MPC_decay_scheme
    ):
        was_eval = not self.model.training
        self.MPC_importance_init=MPC_importance_init
        self.MPC_importance_final=MPC_importance_final
        self.mpc_importance_coef=0.0
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(
            self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        self.optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optim, gamma=0.9997)  # by default running without scheduler, 0.9997
        # copy settings from Raissi et al. (2019) and here
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            self.optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                      history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        self.training_dir = training_dir
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0

        if self.dataset.refinement_schedule is not None:
            self.last_refine_time = 0.0
            self.refinement_schedule_index = 0
        else:
            self.last_refine_time = (
                math.floor(
                    min(1.0, self.dataset.counter / self.dataset.counter_end)
                    * (self.dataset.tMax - self.dataset.tMin)
                    / self.dataset.time_till_refinement
                )
                * self.dataset.time_till_refinement
            )
        self.use_MPC_terminal_loss=False

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            for epoch in range(0, epochs):
                # if current epochs exceed the counter end, then we train with t \in [tMin,tMax]
                time_interval_length = min(1.0,
                                           self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                
                if self.dataset.refine_dataset:
                    self.dataset_refinement(time_interval_length, epoch)
                    

                # semi-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()

                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}

                    model_results = self.model(
                        {'coords': model_input['model_inputs']})

                    states = self.dataset.dynamics.input_to_coord(
                        model_results['model_in'].detach())[..., 1:]

                    values = self.dataset.dynamics.io_to_value(
                        model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))

                    dvs = self.dataset.dynamics.io_to_dv(
                        model_results['model_in'], model_results['model_out'].squeeze(dim=-1))

                    # get optimal next value using discrete bellman eq and cost labels using rollouts
                    if len(model_input['MPC_inputs'].shape)>2: # When MPC data is available
                        MPC_results = self.model(
                            {'coords': model_input['MPC_inputs']})

                        MPC_values = self.dataset.dynamics.io_to_value(
                            MPC_results['model_in'].detach(), MPC_results['model_out'].squeeze(dim=-1))
   
                    else:
                        MPC_values = torch.Tensor([0]).to(device)
                        

                    # Compute losses
                    boundary_values = gt['boundary_values']

                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']

                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(
                            states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'],
                            MPC_values, gt['MPC_values'],self.use_MPC_terminal_loss)
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(
                            states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results[
                                'model_out'], MPC_values, gt['MPC_values'],self.use_MPC_terminal_loss)
                    else:
                        raise NotImplementedError

                    # Adjust the relative magnitude of the losses if required
                    if adjust_relative_grads:
                        self.adjust_rel_weight(losses, MPC_decay_scheme, epoch, epochs, model_input)
                            
                    # scale losses and step the optimizer
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        writer.add_scalar(
                                loss_name, single_loss * self.loss_weights[loss_name], total_steps)
                        train_loss +=   single_loss * self.loss_weights[loss_name]

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss",
                                      train_loss, total_steps)

                    if not use_lbfgs:
                        self.optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=clip_grad)

                        self.optim.step()
                        if epoch > (self.dataset.counter_end + self.dataset.pretrain_iters):
                            scheduler.step()

                    pbar.update(1)
                    # update wandb
                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (
                            epoch, train_loss, time.time() - start_time))
                        if self.use_wandb:
                            wandb.log({
                                'train_loss': train_loss,
                                'boundary_loss': losses['dirichlet'],
                                'pde_loss': losses['diff_constraint_hom']/self.dataset.numpoints,
                                'mpc_loss': losses['mpc_loss']/self.dataset.num_MPC_data_samples,
                                'mpc_importance': self.mpc_importance_coef,
                                'mpc_weight': self.loss_weights['mpc_loss'], 
                            })

                    total_steps += 1

                if not (epoch+1) % epochs_til_checkpoint:
                    # Saving the optimizer state is important to produce consistent results
                    checkpoint = {
                        'epoch': epoch+1,
                        'model': self.model.state_dict(),
                        'optimizer': self.optim.state_dict()}
                    torch.save(checkpoint,
                               os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                               np.array(train_losses))
                    self.validate(
                        epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        x_resolution=val_x_resolution, y_resolution=val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

        torch.save(checkpoint, os.path.join(
            checkpoints_dir, 'model_final.pth')) # save final model

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None,
             gt_data_path=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        if data_step in ["plot_basic_recovery", 'run_basic_recovery', 'plot_ND', 'run_robust_recovery', 'plot_robust_recovery','eval_w_gt']:
            testing_dir = self.experiment_dir
        else:
            testing_dir = os.path.join(
                self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
            if os.path.exists(testing_dir):
                overwrite = input(
                    "The testing directory %s already exists. Overwrite? (y/n)" % testing_dir)
                if not (overwrite == 'y'):
                    print('Exiting.')
                    quit()
                shutil.rmtree(testing_dir)
            os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            # checkpoint x simulation_time square matrices
            sidelen = 10
            assert (last_checkpoint /
                    checkpoint_dt) % sidelen == 0, 'checkpoints cannot be even divided by sidelen'
            BRT_volumes_matrix = np.zeros((sidelen, sidelen))
            BRT_errors_matrix = np.zeros((sidelen, sidelen))
            BRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            BRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            exBRT_volumes_matrix = np.zeros((sidelen, sidelen))
            exBRT_errors_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            checkpoints = np.linspace(0, last_checkpoint, num=sidelen+1)[1:]
            checkpoints[-1] = -1
            times = np.linspace(self.dataset.tMin,
                                self.dataset.tMax, num=sidelen+1)[1:]
            print('constructing matrices for')
            print('checkpoints:', checkpoints)
            print('times:', times)
            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                for j in tqdm(range(sidelen), desc='Simulation Time', leave=False):
                    # get BRT volume, error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        violation_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    BRT_volumes_matrix[i, j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        BRT_errors_matrix[i,
                                          j] = results['max_violation_error']
                        BRT_error_rates_matrix[i,
                                               j] = results['violation_rate']
                        BRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=float('-inf'), v_max=0.0),
                            target_validator=ValueThresholdValidator(
                                v_min=-results['max_violation_error'], v_max=0.0),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        BRT_errors_matrix[i, j] = np.NaN
                        BRT_error_rates_matrix[i, j] = np.NaN
                        BRT_error_region_fracs_matrix[i, j] = np.NaN

                    # get exBRT error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    exBRT_volumes_matrix[i,
                                         j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        exBRT_errors_matrix[i,
                                            j] = results['max_violation_error']
                        exBRT_error_rates_matrix[i,
                                                 j] = results['violation_rate']
                        exBRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=float('inf')),
                            target_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=results['max_violation_error']),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        exBRT_errors_matrix[i, j] = np.NaN
                        exBRT_error_rates_matrix[i, j] = np.NaN
                        exBRT_error_region_fracs_matrix[i, j] = np.NaN

            # save the matrices
            matrices = {
                'BRT_volumes_matrix': BRT_volumes_matrix,
                'BRT_errors_matrix': BRT_errors_matrix,
                'BRT_error_rates_matrix': BRT_error_rates_matrix,
                'BRT_error_region_fracs_matrix': BRT_error_region_fracs_matrix,
                'exBRT_volumes_matrix': exBRT_volumes_matrix,
                'exBRT_errors_matrix': exBRT_errors_matrix,
                'exBRT_error_rates_matrix': exBRT_error_rates_matrix,
                'exBRT_error_region_fracs_matrix': exBRT_error_region_fracs_matrix,
            }
            for name, arr in matrices.items():
                with open(os.path.join(testing_dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, arr)

            # plot the matrices
            matrices = {
                'BRT_volumes_matrix': [
                    BRT_volumes_matrix, 'BRT Fractions of Test State Space'
                ],
                'BRT_errors_matrix': [
                    BRT_errors_matrix, 'BRT Errors'
                ],
                'BRT_error_rates_matrix': [
                    BRT_error_rates_matrix, 'BRT Error Rates'
                ],
                'BRT_error_region_fracs_matrix': [
                    BRT_error_region_fracs_matrix, 'BRT Error Region Fractions'
                ],
                'exBRT_volumes_matrix': [
                    exBRT_volumes_matrix, 'exBRT Fractions of Test State Space'
                ],
                'exBRT_errors_matrix': [
                    exBRT_errors_matrix, 'exBRT Errors'
                ],
                'exBRT_error_rates_matrix': [
                    exBRT_error_rates_matrix, 'exBRT Error Rates'
                ],
                'exBRT_error_region_fracs_matrix': [
                    exBRT_error_region_fracs_matrix, 'exBRT Error Region Fractions'
                ],
            }
            for name, data in matrices.items():
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                ax.imshow(data[0], cmap=cmap)
                plt.title(data[1])
                for (y, x), label in np.ndenumerate(data[0]):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(testing_dir, name + '.png'), dpi=600)
                plt.clf()
                # log version
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                new_matrix = np.log(data[0])
                ax.imshow(new_matrix, cmap=cmap)
                plt.title('(Log) ' + data[1])
                for (y, x), label in np.ndenumerate(new_matrix):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(
                    testing_dir, name + '_log' + '.png'), dpi=600)
                plt.clf()

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics

            
            if data_step == "eval_w_gt":
                coords = torch.load(os.path.join(gt_data_path, "coords.pt")).to(device)
                gt_values = torch.load(os.path.join(gt_data_path, "gt_values.pt")).to(device)
                with torch.no_grad():
                    results = model(
                        {'coords': self.dataset.dynamics.coord_to_input(coords)})
                    pred_values = self.dataset.dynamics.io_to_value(results['model_in'].detach(
                        ), results['model_out'].squeeze(dim=-1).detach())
                mse = torch.pow(pred_values-gt_values,2).mean()


                # print(results['batch_state_trajs'].shape)
                gt_values=gt_values.cpu().numpy()
                pred_values=pred_values.cpu().numpy()
                fp=np.argwhere(np.logical_and(gt_values < 0, pred_values >= 0)).shape[0]/pred_values.shape[0]
                fn=np.argwhere(np.logical_and(gt_values >= 0, pred_values < 0)).shape[0]/pred_values.shape[0]
                np.save(os.path.join(
                    testing_dir, f"mse.npy"),mse.cpu().numpy())
                np.save(os.path.join(
                    testing_dir, f"fp.npy"),torch.tensor([fp]))
                np.save(os.path.join(
                    testing_dir, f"fn.npy"),torch.tensor([fn]))
                print("False positive: %0.4f, False negative: %0.4f"%(fp,
                        fn))
                print("MSE: ", mse)

            if data_step == 'plot_robust_recovery':
                epsilons=-np.load(os.path.join(testing_dir, f'epsilons.npy'))+1
                deltas=np.load(os.path.join(testing_dir, f'deltas.npy'))
                target_eps=0.01
                delta_level=deltas[np.argmin(np.abs(epsilons-target_eps))]
                fig,values_slices = self.plot_recovery_fig(
                    dataset, dynamics, model, delta_level)
                fig.savefig(os.path.join(
                    testing_dir, f'robust_BRTs_1e-2.png'), dpi=800)
                np.save(os.path.join(testing_dir, f'values_slices'),values_slices)

            if data_step == 'run_robust_recovery':
                logs = {}
                # rollout samples all over the state space
                beta_ = 1e-10
                N = 300000
                logs['beta_'] = beta_
                logs['N'] = N
                delta_level = float(
                    'inf') if dynamics.set_mode in ['reach','reach_avoid'] else float('-inf')

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                values_ = results['values'].cpu().numpy()
                unsafe_cost_safe_value_indeces = np.argwhere(
                    np.logical_and(costs_ < 0, values_ >= 0))

                print("k max: ", unsafe_cost_safe_value_indeces.shape[0])

                # determine delta_level_max
                delta_level_max = np.max(
                    values_[unsafe_cost_safe_value_indeces])
                print("delta_level_max: ", delta_level_max)

                # for each delta level, determine (1) the corresponding volume;
                # (2) k and and corresponding epsilon
                ks = []
                epsilons = []
                volumes = []

                for delta_level_ in np.arange(0, delta_level_max, delta_level_max/100):
                    k = int(np.argwhere(np.logical_and(
                        costs_ < 0, values_ >= delta_level_)).shape[0])
                    eps = beta__dist.ppf(beta_,  N-k, k+1)
                    volume = values_[values_ >= delta_level_].shape[0]/values_.shape[0]
                    
                    ks.append(k)
                    epsilons.append(eps)
                    volumes.append(volume)

                # plot epsilon volume graph
                fig1, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('volumes')
                ax1.set_ylabel('epsilons', color=color)
                ax1.plot(volumes, epsilons, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()

                color = 'tab:blue'
                ax2.set_ylabel('number of outliers', color=color)
                ax2.plot(volumes, ks, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                plt.title("beta_=1e-10, N =3e6")
                fig1.savefig(os.path.join(
                    testing_dir, f'robust_verification_results.png'), dpi=800)
                plt.close(fig1)
                np.save(os.path.join(testing_dir, f'epsilons'),
                        epsilons)
                np.save(os.path.join(testing_dir, f'volumes'),
                        volumes)
                np.save(os.path.join(testing_dir, f'deltas'),
                        np.arange(0, delta_level_max, delta_level_max/100))
                np.save(os.path.join(testing_dir, f'ks'),
                        ks)
                
            if data_step == 'run_basic_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for tMax
                # record state/learned_value/violation for each while loop iteration
                delta_level = float(
                    'inf') if dynamics.set_mode in ['reach','reach_avoid'] else float('-inf')
                algorithm_iters = []
                for i in range(M):
                    print('algorithm iter', str(i))
                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=N, max_samples=1000*min(N, 10000))
                    if not results['maxed_scenarios']:
                        delta_level = float(
                            '-inf') if dynamics.set_mode in ['reach','reach_avoid'] else float('inf')
                        break
                    algorithm_iters.append(
                        {
                            'states': results['states'],
                            'values': results['values'],
                            'violations': results['violations']
                        }
                    )
                    if results['violation_rate'] == 0:
                        break
                    violation_levels = results['values'][results['violations']]
                    delta_level_arg = np.argmin(
                        violation_levels) if dynamics.set_mode in ['reach','reach_avoid'] else np.argmax(violation_levels)
                    delta_level = violation_levels[delta_level_arg].item()

                    print('violation_rate:', str(results['violation_rate']))
                    print('delta_level:', str(delta_level))
                    print('valid_sample_fraction:', str(
                        results['valid_sample_fraction'].item()))
                    sns.set_style('whitegrid')
                    # density_plot=sns.kdeplot(results['costs'].cpu().numpy(), bw=0.5)
                    # density_plot=sns.displot(results['costs'].cpu().numpy(), x="cost function")
                    # fig1 = density_plot.get_figure()
                    fig1 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy(), bins=200)
                    fig1.savefig(os.path.join(
                        testing_dir, f'cost distribution.png'), dpi=800)
                    plt.close(fig1)
                    fig2 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy() -
                             results['values'].cpu().numpy(), bins=200)
                    fig2.savefig(os.path.join(
                        testing_dir, f'diff distribution.png'), dpi=800)
                    plt.close(fig1)

                logs['algorithm_iters'] = algorithm_iters
                logs['delta_level'] = delta_level

                # 2. record solution volume, recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['recovered_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000)
                ).item()

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0

                print('learned_volume', str(logs['learned_volume']))
                print('recovered_volume', str(logs['recovered_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['recovered_violation_rate'] = results['violation_rate']
                else:
                    logs['recovered_violation_rate'] = 0
                print('recovered_violation_rate', str(
                    logs['recovered_violation_rate']))

                with open(os.path.join(testing_dir, 'basic_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_basic_recovery':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                print('S:', str(logs['S']))
                print('delta level', str(logs['delta_level']))
                delta_level = logs['delta_level']
                print('learned volume', str(logs['learned_volume']))
                print('recovered volume', str(logs['recovered_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('recovered violation rate', str(
                    logs['recovered_violation_rate']))

                # fig, _ = self.plot_recovery_fig(
                #     dataset, dynamics, model, delta_level)
                plot_config = self.dataset.dynamics.plot_config()

                state_test_range = self.dataset.dynamics.state_test_range()
                
                times = [self.dataset.tMax]
                if plot_config['z_axis_idx'] == -1:
                    fig = self.plotSingleFig(
                        state_test_range, plot_config, 512, 512, times, delta_level)
                else:
                    fig= self.plotMultipleFigs(
                        state_test_range, plot_config, 512, 512, 5, times, delta_level)
                # plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'basic_BRTs.png'), dpi=800)
                np.save(os.path.join(
                    testing_dir, f'volumes.npy'), np.array([float(logs['learned_volume']),
                                                            float(logs['recovered_volume']),float(logs['theoretically_recoverable_volume'])]))

            if data_step == 'plot_ND':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                print('S:', str(logs['S']))
                print('delta level', str(logs['delta_level']))
                delta_level = logs['delta_level']
                print('learned volume', str(logs['learned_volume']))
                print('recovered volume', str(logs['recovered_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('recovered violation rate', str(
                    logs['recovered_violation_rate']))

                
                plot_config = self.dataset.dynamics.plot_config()

                state_test_range = self.dataset.dynamics.state_test_range()
                
                times = [self.dataset.tMax]
                if plot_config['z_axis_idx'] == -1:
                    fig = self.plotSingleFig(
                        state_test_range, plot_config, 512, 512, times, delta_level)
                else:
                    fig= self.plotMultipleFigs(
                        state_test_range, plot_config, 512, 512, 5, times, delta_level)
                    
                x_resolution=512
                y_resolution=512
                x_min, x_max = state_test_range[plot_config['x_axis_idx']]
                y_min, y_max = state_test_range[plot_config['y_axis_idx']]

                xs = torch.linspace(x_min, x_max, x_resolution)
                ys = torch.linspace(y_min, y_max, y_resolution)
                xys = torch.cartesian_prod(xs, ys)
                Xg, Yg = torch.meshgrid(xs, ys)
                
                ## Plot Set and Value Fn
                fig = plt.figure(figsize=(5*len(times), 2*5*1), facecolor='white')
                
                plt.rcParams['text.usetex'] = False

                # for i in range(3*len(times)):
                for i in range(2*len(times)):
                    
                    if i >= len(times):
                        ax = fig.add_subplot(2, len(times), 1+i)
                    else:
                        ax = fig.add_subplot(2, len(times), 1+i, projection='3d')
                    ax.set_title(r"t =" + "%0.2f" % (times[i % len(times)]))

                    ## Define Grid Slice to Plot

                    coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                    coords[:, 0] = times[i % len(times)]
                    coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)

                    # xN - (xi = xj) plane
                    if i >= len(times):
                        pad_label = 0
                        ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label)
                        ax.set_xticks([-1, 1])
                        ax.set_xticklabels([r'$-1$', r'$1$'])
                        ax.set_yticks([-1, 1])
                        ax.set_yticklabels([r'$-1$', r'$1$'])

                        ax_pad = 0
                        ax.xaxis.set_tick_params(pad=ax_pad)
                        ax.yaxis.set_tick_params(pad=ax_pad)

                    else:
                        pad_label = 6
                        ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); 
                        ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label); 
                        ax.set_zlabel(r"$V$", fontsize=12, labelpad=10) #, labelpad=pad_label)
                        ax.set_xticks([-1, 0, 1])
                        ax.set_xticklabels([r'$-1$', r'$0$', r'$1$'])
                        ax.set_yticks([-1, 0, 1])
                        ax.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
                        ax.set_zticks([])
                        ax.zaxis.label.set_position((-0.1, 0.5))
                        
                        ax.xaxis.pane.fill = False
                        ax.yaxis.pane.fill = False
                        ax.zaxis.pane.fill = False
                        
                        ax_pad = 0
                        ax.xaxis.set_tick_params(pad=ax_pad)
                        ax.yaxis.set_tick_params(pad=ax_pad)
                        ax.zaxis.set_tick_params(pad=ax_pad)

                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 2:] = (xys[:, 1] * torch.ones(self.dataset.dynamics.N-1, xys.size()[0])).t()

                    with torch.no_grad():
                        model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                        values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                    
                    learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

                    xig = torch.arange(-0.99, 1.01, 0.02) # 100 x 100
                    X1g, X2g = torch.meshgrid(xig, xig)

                    Vgt=np.load("./dynamics/vgt_40D.npy")
                    ## Make Value-Based Colormap
                    
                    cmap_name = "RdBu"

                    if learned_value.min() > 0:
                        # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                        scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., 256))))
                        RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

                    elif learned_value.max() < 0:
                        # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                        scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256))))
                        RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

                    else:
                        # n_bins_high = int(256 * (learned_value.max()/(learned_value.max() - learned_value.min())) // 1)
                        n_bins_high = round(256 * learned_value.max()/(learned_value.max() - learned_value.min()))


                        offset = 0
                        scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
                        RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)
                    
                    if i >= len(times):

                        ## Plot Zero-level Set of Learned Value
                        
                        # s = ax.imshow(1*(learned_value.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                        # s = ax.imshow(learned_value.T, cmap='coolwarm_r', origin='lower', extent=(-1., 1., -1., 1.))
                        s = ax.contourf(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, levels=256)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = fig.colorbar(s, cax=cax)
                        cbar.set_ticks([learned_value.min(), 0., learned_value.max()])  # Define custom tick locations
                        cbar.set_ticklabels([f'{learned_value.min():1.1f}', '0', f'{learned_value.max():1.1f}'])  # Define custom tick labels

                        ## Plot Ground-Truth Zero-Level Contour

                        ax.contour(X1g, X2g, Vgt, [0.], linewidths=4, alpha=0.7, colors='brown')
                        ax.contour(Xg, Yg, learned_value, [0.], linewidths=1, alpha=1, colors='k',linestyles="--")
                        ax.contour(Xg, Yg, learned_value, [-0.0744], linewidths=1, alpha=1, colors='k',linestyles="-")
                    
                    else:


                        ax.view_init(elev=15, azim=-60)
                        ax.set_facecolor((1, 1, 1, 1))
                        surf = ax.plot_surface(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')

                        cbar = fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.0)

                        ax.set_zlim(learned_value.min() - (learned_value.max() - learned_value.min())/5)

                        ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')

                fig.savefig(os.path.join(
                    testing_dir, f'basic_BRTs.png'), dpi=800)
                
                
        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def plotSingleFig(self, state_test_range, plot_config, x_resolution, y_resolution, times, delta_level = None):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys)
        fig = plt.figure(figsize=(6, 5*len(times)))
        X, Y = np.meshgrid(xs, ys)
        for i in range(len(times)):
            coords = torch.zeros(
                x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i]
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            with torch.no_grad():
                model_results = self.model(
                    {'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})

                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                ), model_results['model_out'].squeeze(dim=-1).detach())

            ax = fig.add_subplot(len(times), 1, 1 + i)
            ax.set_title('t = %0.2f' % (times[i]))
            BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
            max_value = np.amax(BRT_img)
            min_value = np.amin(BRT_img)
            imshow_kwargs = {
                'vmax': max_value,
                'vmin': min_value,
                'cmap': 'coolwarm_r',
                'extent': (x_min, x_max, y_min, y_max),
                'origin': 'lower',
            }
            ax.imshow(BRT_img, **imshow_kwargs)
            lx=self.dataset.dynamics.boundary_fn(coords.to(device)[...,1:]).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
            zero_contour = ax.contour(X, 
                                Y, 
                                BRT_img, 
                                levels=[0.0],  
                                colors="black",  
                                linewidths=2,    
                                linestyles='--')  
                
            failure_set_contour = ax.contour(X, 
                            Y, 
                            lx, 
                            levels=[0.0],  
                            colors="saddlebrown",  
                            linewidths=2,    
                            linestyles='-')  
            if delta_level is not None:
                delta_contour = ax.contour(X, 
                            Y, 
                            BRT_img, 
                            levels=[delta_level],  
                            colors="black",  
                            linewidths=2,    
                            linestyles='-')  
        return fig

    def plotMultipleFigs(self, state_test_range, plot_config, x_resolution, y_resolution, z_resolution, times, delta_level = None):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]


        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)

        fig = plt.figure(figsize=(6*len(zs),5*len(times)))
        X, Y = np.meshgrid(xs, ys)
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(
                    x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                lx=self.dataset.dynamics.boundary_fn(coords.to(device)[...,1:]).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
                with torch.no_grad():
                    model_results = self.model(
                        {'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                        ), model_results['model_out'].squeeze(dim=-1).detach())

                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (
                    times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))

                BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T

                max_value = np.amax(BRT_img)
                min_value = np.amin(BRT_img)
                imshow_kwargs = {
                    'vmax': max_value,
                    'vmin': min_value,
                    'cmap': 'coolwarm_r',
                    'extent': (x_min, x_max, y_min, y_max),
                    'origin': 'lower',
                }

                s1 = ax.imshow(BRT_img, **imshow_kwargs)
                fig.colorbar(s1)
                zero_contour = ax.contour(X, 
                                Y, 
                                BRT_img, 
                                levels=[0.0],  
                                colors="black",  
                                linewidths=2,    
                                linestyles='--')  
                
                failure_set_contour = ax.contour(X, 
                                Y, 
                                lx, 
                                levels=[0.0],  
                                colors="saddlebrown",  
                                linewidths=2,    
                                linestyles='-')  
                
                if delta_level is not None:
                    delta_contour = ax.contour(X, 
                                Y, 
                                BRT_img, 
                                levels=[delta_level],  
                                colors="black",  
                                linewidths=2,    
                                linestyles='-')  
   
        return fig

    def validate(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        if plot_config['z_axis_idx'] == -1:
            fig = self.plotSingleFig(
                state_test_range, plot_config, x_resolution, y_resolution, times)
        else:
            fig= self.plotMultipleFigs(
                state_test_range, plot_config, x_resolution, y_resolution, z_resolution, times)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()
        plt.close()
        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def adjust_rel_weight(self, losses, MPC_decay_scheme, epoch, epochs, model_input):
        if self.dataset.dynamics.deepReach_model in ['vanilla', 'diff'] and losses['diff_constraint_hom'] > 0.01:
            params = OrderedDict(self.model.named_parameters())
            # Gradients with respect to the PDE loss
            self.optim.zero_grad()
            losses['diff_constraint_hom'].backward(
                retain_graph=True)
            grads_PDE = []
            for key, param in params.items():
                grads_PDE.append(param.grad.view(-1))
            grads_PDE = torch.cat(grads_PDE)

            # Gradients with respect to the boundary loss
            self.optim.zero_grad()
            losses['dirichlet'].backward(retain_graph=True)
            grads_dirichlet = []
            for key, param in params.items():
                grads_dirichlet.append(param.grad.view(-1))
            grads_dirichlet = torch.cat(grads_dirichlet)

            # Set the new weight according to the paper
            # num = torch.max(torch.abs(grads_PDE))
            num = torch.mean(torch.abs(grads_PDE))
            den = torch.mean(torch.abs(grads_dirichlet))
            self.loss_weights['dirichlet'] = 0.9*self.loss_weights['dirichlet'] + 0.1*num/den
            losses['dirichlet'] = self.loss_weights['dirichlet'] * \
                losses['dirichlet']
            if (self.dataset.counter>=1) and len(model_input['MPC_inputs'].shape)>2:
                if MPC_decay_scheme=="exponential":
                    if self.MPC_importance_final<= self.MPC_importance_init:
                        self.mpc_importance_coef = self.MPC_importance_final * math.e**(math.log(self.MPC_importance_init/self.MPC_importance_final)*(1-(epoch - self.dataset.pretrain_iters)/(epochs - 1 - self.dataset.pretrain_iters))) 
                    else:
                        self.mpc_importance_coef = self.MPC_importance_final * math.e**(math.log(self.MPC_importance_init/self.MPC_importance_final)*(epoch - self.dataset.pretrain_iters)/(epochs - 1 - self.dataset.pretrain_iters))
                        self.mpc_importance_coef = self.MPC_importance_final + self.MPC_importance_init - self.mpc_importance_coef
                elif MPC_decay_scheme=="linear":
                    self.mpc_importance_coef=self.MPC_importance_init+(self.MPC_importance_final-self.MPC_importance_init)*(epoch - self.dataset.pretrain_iters)/(epochs - self.dataset.pretrain_iters)
                else: 
                    raise NotImplementedError
                
                # Gradients with respect to the mpc loss
                self.optim.zero_grad()
                losses['mpc_loss'].backward(retain_graph=True)
                grads_mpc = []
                for key, param in params.items():
                    grads_mpc.append(param.grad.view(-1))
                grads_mpc = torch.cat(grads_mpc)
                # Set the new weight according to the paper 
                den = torch.mean(torch.abs(grads_mpc))
                
                self.loss_weights['mpc_loss'] = 0.9*self.loss_weights['mpc_loss'] + 0.1*self.mpc_importance_coef*num/(den+1e-16)
                
        elif self.dataset.dynamics.deepReach_model == 'exact' and (self.dataset.counter>=1) and losses['mpc_loss'] > 1e-8:

            self.loss_weights['diff_constraint_hom']=1.0
            params = OrderedDict(self.model.named_parameters())
            # Gradients with respect to the PDE loss
            self.optim.zero_grad()
            losses['diff_constraint_hom'].backward(
                retain_graph=True)
            grads_PDE = []
            for key, param in params.items():
                grads_PDE.append(param.grad.view(-1))
            grads_PDE = torch.cat(grads_PDE)

            if MPC_decay_scheme=="exponential":
                if self.MPC_importance_final<= self.MPC_importance_init:
                    self.mpc_importance_coef = self.MPC_importance_final * math.e**(math.log(self.MPC_importance_init/self.MPC_importance_final)*(1-(epoch - self.dataset.pretrain_iters)/(epochs - 1 - self.dataset.pretrain_iters))) 
                else:
                    self.mpc_importance_coef = self.MPC_importance_final * math.e**(math.log(self.MPC_importance_init/self.MPC_importance_final)*(epoch - self.dataset.pretrain_iters)/(epochs - 1 - self.dataset.pretrain_iters))
                    self.mpc_importance_coef = self.MPC_importance_final + self.MPC_importance_init - self.mpc_importance_coef
            elif MPC_decay_scheme=="linear":
                self.mpc_importance_coef=self.MPC_importance_init+(self.MPC_importance_final-self.MPC_importance_init)*(epoch - self.dataset.pretrain_iters)/(epochs - self.dataset.pretrain_iters)
            else: 
                raise NotImplementedError
            
            # Gradients with respect to the mpc loss
            self.optim.zero_grad()
            losses['mpc_loss'].backward(retain_graph=True)
            grads_mpc = []
            for key, param in params.items():
                grads_mpc.append(param.grad.view(-1))
            grads_mpc = torch.cat(grads_mpc)
            # Set the new weight according to the paper
            # num = torch.max(torch.abs(grads_PDE))
            den = torch.mean(torch.abs(grads_mpc))
            num = torch.mean(torch.abs(grads_PDE))
            
            self.loss_weights['mpc_loss'] =min( 0.9*self.loss_weights['mpc_loss'] + 0.1*self.mpc_importance_coef*num/(den+1e-16), 1e5)

    def dataset_refinement(self, time_interval_length, epoch):
        should_refine = False
        next_refine_time = None

        if self.dataset.refinement_schedule is not None:
            if hasattr(self, "refinement_schedule_index") and self.refinement_schedule_index < len(
                self.dataset.refinement_schedule
            ):
                next_refine_time = self.dataset.refinement_schedule[self.refinement_schedule_index]
                should_refine = time_interval_length >= next_refine_time
            else:
                should_refine = False
        else:
            should_refine = time_interval_length >= (self.last_refine_time + self.dataset.time_till_refinement)
            if should_refine:
                next_refine_time = self.last_refine_time + self.dataset.time_till_refinement

        if should_refine and self.dataset.use_MPC:
            if self.dataset.refinement_schedule is not None:
                self.last_refine_time = next_refine_time
                self.refinement_schedule_index += 1
                if self.refinement_schedule_index < len(self.dataset.refinement_schedule):
                    refine_till_t = self.dataset.refinement_schedule[self.refinement_schedule_index]
                else:
                    refine_till_t = self.dataset.tMax
            else:
                self.last_refine_time += self.dataset.time_till_refinement
                refine_till_t = time_interval_length + self.dataset.time_till_refinement

            self.dataset.policy = self.model

            if refine_till_t < self.dataset.tMax:
                self.dataset.generate_MPC_dataset(refine_till_t, time_interval_length, style="random")
            else:
                self.dataset.use_terminal_MPC()
                for g in self.optim.param_groups:
                    g['lr'] = 1e-6
                self.use_MPC_terminal_loss=True
                self.MPC_importance_final=1.0
                self.MPC_importance_init=1.0
                self.dataset.policy=self.model
                refine_till_t=self.dataset.tMax
                self.dataset.generate_MPC_dataset(refine_till_t, refine_till_t, style="terminal")

        if time_interval_length>=self.dataset.tMax and epoch%self.dataset.epoch_till_refinement== 0 and self.dataset.use_MPC:
            for g in self.optim.param_groups:
                g['lr'] = 1e-6
            self.use_MPC_terminal_loss=True
            self.dataset.policy=self.model
            self.dataset.use_terminal_MPC()
            self.dataset.generate_MPC_dataset(
                            self.dataset.tMax , self.dataset.tMax, style="terminal")
            
    def plot_recovery_fig(self, dataset, dynamics, model, delta_level):
        # 1. for ground truth slices (if available), record (higher-res) grid of learned values
        # plot (with ground truth) learned BRTs, recovered BRTs
        z_res = 5
        plot_config = dataset.dynamics.plot_config()
        if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
            ground_truth = spio.loadmat(os.path.join(
                self.experiment_dir, 'ground_truth.mat'))
            if 'gmat' in ground_truth:
                ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                ground_truth_values = ground_truth['data']
                ground_truth_ts = np.linspace(
                    0, 1, ground_truth_values.shape[3])

            elif 'g' in ground_truth:
                ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                ground_truth_ts = ground_truth['tau'][0]
                ground_truth_values = ground_truth['data']

            # idxs to plot
            x_idxs = np.linspace(0, len(ground_truth_xs)-1,
                                 len(ground_truth_xs)).astype(dtype=int)
            y_idxs = np.linspace(0, len(ground_truth_ys)-1,
                                 len(ground_truth_ys)).astype(dtype=int)
            z_idxs = np.linspace(0, len(ground_truth_zs) -
                                 1, z_res).astype(dtype=int)
            t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # indexed ground truth to plot
            ground_truth_xs = ground_truth_xs[x_idxs]
            ground_truth_ys = ground_truth_ys[y_idxs]
            ground_truth_zs = ground_truth_zs[z_idxs]
            ground_truth_ts = ground_truth_ts[t_idxs]
            ground_truth_values = ground_truth_values[
                x_idxs[:, None, None, None],
                y_idxs[None, :, None, None],
                z_idxs[None, None, :, None],
                t_idxs[None, None, None, :]
            ]
            ground_truth_grids = ground_truth_values

            xs = ground_truth_xs
            ys = ground_truth_ys
            zs = ground_truth_zs
        else:
            ground_truth_grids = None
            resolution = 512
            xs = np.linspace(*dynamics.state_test_range()
                             [plot_config['x_axis_idx']], resolution)
            ys = np.linspace(*dynamics.state_test_range()
                             [plot_config['y_axis_idx']], resolution)
            zs = np.linspace(*dynamics.state_test_range()
                             [plot_config['z_axis_idx']], z_res)

        xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
        value_grids = np.zeros((len(zs), len(xs), len(ys)))
        for i in range(len(zs)):
            coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
            coords[:, 0] = dataset.tMax
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            if dataset.dynamics.state_dim > 2:
                coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

            model_results = model(
                {'coords': dataset.dynamics.coord_to_input(coords.to(device))})
            values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
            ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
            value_grids[i] = values.reshape(len(xs), len(ys))

        fig = plt.figure()
        fig.suptitle(plot_config['state_slices'], fontsize=8)
        x_min, x_max = dataset.dynamics.state_test_range()[
            plot_config['x_axis_idx']]
        y_min, y_max = dataset.dynamics.state_test_range()[
            plot_config['y_axis_idx']]

        for i in range(len(zs)):
            values = value_grids[i]

            # learned BRT and recovered BRT
            ax = fig.add_subplot(1, len(zs), (i+1))
            ax.set_title('%s = %0.2f' % (
                plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

            image = np.full((*values.shape, 3), 255, dtype=int)
            BRT = values < 0
            recovered_BRT = values < delta_level

            if dynamics.set_mode in ['reach','reach_avoid']:
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                image[recovered_BRT] = np.array([155, 241, 249])
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)
            else:
                image[recovered_BRT] = np.array([155, 241, 249])
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                # overlay recovered border over learned BRT
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)

            ax.imshow(image.transpose(1, 0, 2), origin='lower',
                      extent=(x_min, x_max, y_min, y_max))

            ax.set_xlabel(plot_config['state_labels']
                          [plot_config['x_axis_idx']])
            ax.set_ylabel(plot_config['state_labels']
                          [plot_config['y_axis_idx']])
            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.tick_params(labelsize=6)
            if i != 0:
                ax.set_yticks([])
        return fig, value_grids

    def overlay_ground_truth(self, image, z_idx, ground_truth_grids):
        thickness = max(0, image.shape[0] // 120 - 1)
        ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
        ground_truth_brts = ground_truth_grid < 0
        for x in range(ground_truth_brts.shape[0]):
            for y in range(ground_truth_brts.shape[1]):
                if not ground_truth_brts[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_brts.shape[0] and neighbor[1] < ground_truth_brts.shape[1]:
                        if not ground_truth_brts[neighbor]:
                            image[x-thickness:x+1, y-thickness:y +
                                  1+thickness] = np.array([50, 50, 50])
                            break

    def overlay_border(self, image, set, color):
        thickness = max(0, image.shape[0] // 120 - 1)
        for x in range(set.shape[0]):
            for y in range(set.shape[1]):
                if not set[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                        if not set[neighbor]:
                            image[x-thickness:x+1, y -
                                  thickness:y+1+thickness] = color
                            break

    
class DeepReach(Experiment):
    def init_special(self):
        pass
