import wandb
import configargparse
import inspect
import os
import torch
import shutil
import random
import numpy as np
import pickle

from datetime import datetime
from dynamics import dynamics 
from experiments import experiments
from utils import modules, dataio, losses

p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--mode', type=str, required=True, choices=['all', 'train', 'test'], help="Experiment mode to run (new experiments must choose 'all' or 'train').")

# save/load directory options
p.add_argument('--experiments_dir', type=str, default='./runs', help='Where to save the experiment subdirectory.')
p.add_argument('--experiment_name', type=str, required=True, help='Name of the experient subdirectory.')
p.add_argument('--wandb_project', type=str, required=True, help='wandb project')
p.add_argument('--wandb_group', type=str, required=True, help='wandb group')
p.add_argument('--wandb_name', type=str, required=True, help='name of wandb run')

mode = p.parse_known_args()[0].mode

if (mode == 'all') or (mode == 'train'):
    p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the experiment.')

    # load experiment_class choices dynamically from experiments module
    experiment_classes_dict = {name: clss for name, clss in inspect.getmembers(experiments, inspect.isclass) if clss.__bases__[0] == experiments.Experiment}
    p.add_argument('--experiment_class', type=str, default='DeepReach', choices=experiment_classes_dict.keys(), help='Experiment class to use.')
    # load special experiment_class arguments dynamically from chosen experiment class
    experiment_class = experiment_classes_dict[p.parse_known_args()[0].experiment_class]
    experiment_params = {name: param for name, param in inspect.signature(experiment_class.init_special).parameters.items() if name != 'self'}
    for param in experiment_params.keys():
        p.add_argument('--' + param, type=experiment_params[param].annotation, required=True, help='special experiment_class argument')

    # simulation data source options
    p.add_argument('--numpoints', type=int, default=65000, help='Number of points in simulation data source __getitem__.')
    p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')
    p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
    p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
    p.add_argument('--tMax', type=float, default=1.0, required=False, help='End time of the simulation')
    p.add_argument('--counter_start', type=int, default=0, required=False, help='Defines the initial time for the curriculum training')
    p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
    p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples (initial-time samples) at each time step')
    p.add_argument('--num_target_samples', type=int, default=0, required=False, help='Number of samples inside the target set')

    # model options
    p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'], help='Type of model to evaluate, default is sine.')
    p.add_argument('--model_mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'], help='Whether to use uniform velocity parameter')
    p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
    p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')

    # training options
    p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=100, help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size used during training (irrelevant, since len(dataset) == 1).')
    p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
    p.add_argument('--num_epochs', type=int, default=100000, help='Number of epochs to train for.')
    p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
    p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
    p.add_argument('--adj_rel_grads', default=True, type=bool, help='adjust the relative magnitude of the losses')
    p.add_argument('--dirichlet_loss_divisor', default=1.0, required=False, type=float, help='What to divide the dirichlet loss by for loss reweighting')

    # cost-supervised learning (CSL) options
    p.add_argument('--use_CSL', default=False, action='store_true', help='use cost-supervised learning (CSL)')
    p.add_argument('--CSL_lr', type=float, default=2e-5, help='The learning rate used for CSL')
    p.add_argument('--CSL_dt', type=float, default=0.0025, help='The dt used in rolling out trajectories to get cost labels')
    p.add_argument('--epochs_til_CSL', type=int, default=10000, help='Number of epochs between CSL phases')
    p.add_argument('--num_CSL_samples', type=int, default=1000000, help='Number of cost samples in training dataset for CSL phases')
    p.add_argument('--CSL_loss_frac_cutoff', type=float, default=0.1, help='Fraction of initial cost loss on validation dataset to cutoff CSL phases')
    p.add_argument('--max_CSL_epochs', type=int, default=100, help='Max number of CSL epochs per phase')
    p.add_argument('--CSL_loss_weight', type=float, default=1.0, help='weight of cost loss (relative to PDE loss)')
    p.add_argument('--CSL_batch_size', type=int, default=1000, help='Batch size for training in CSL phases')

    # validation (during training) options
    p.add_argument('--val_x_resolution', type=int, default=200, help='x-axis resolution of validation plot during training')
    p.add_argument('--val_y_resolution', type=int, default=200, help='y-axis resolution of validation plot during training')
    p.add_argument('--val_z_resolution', type=int, default=5, help='z-axis resolution of validation plot during training')
    p.add_argument('--val_time_resolution', type=int, default=3, help='time-axis resolution of validation plot during training')

    # loss options
    p.add_argument('--minWith', type=str, required=True, choices=['none', 'zero', 'target'], help='BRS vs BRT computation (typically should be using target for BRT)')

    # load dynamics_class choices dynamically from dynamics module
    dynamics_classes_dict = {name: clss for name, clss in inspect.getmembers(dynamics, inspect.isclass) if clss.__bases__[0] == dynamics.Dynamics}
    p.add_argument('--dynamics_class', type=str, required=True, choices=dynamics_classes_dict.keys(), help='Dynamics class to use.')
    # load special dynamics_class arguments dynamically from chosen dynamics class
    dynamics_class = dynamics_classes_dict[p.parse_known_args()[0].dynamics_class]
    dynamics_params = {name: param for name, param in inspect.signature(dynamics_class).parameters.items() if name != 'self'}
    for param in dynamics_params.keys():
        if dynamics_params[param].annotation is bool:
            p.add_argument('--' + param, type=dynamics_params[param].annotation, default=False, help='special dynamics_class argument')
        else:
            p.add_argument('--' + param, type=dynamics_params[param].annotation, required=True, help='special dynamics_class argument')

if (mode == 'all') or (mode == 'test'):
    p.add_argument('--dt', type=float, default=0.0025, help='The dt used in testing simulations')
    p.add_argument('--checkpoint_toload', type=int, default=None, help="The checkpoint to load for testing (-1 for final training checkpoint, None for cross-checkpoint testing")
    p.add_argument('--num_scenarios', type=int, default=100000, help='The number of scenarios sampled in scenario optimization for testing')
    p.add_argument('--num_violations', type=int, default=1000, help='The number of violations to sample for in scenario optimization for testing')
    p.add_argument('--control_type', type=str, default='value', choices=['value', 'ttr', 'init_ttr'], help='The controller to use in scenario optimization for testing')
    p.add_argument('--data_step', type=str, default='run_basic_recovery', choices=['plot_violations', 'run_basic_recovery', 'plot_basic_recovery', 'collect_samples', 'train_binner', 'run_binned_recovery', 'plot_binned_recovery', 'plot_cost_function'], help='The data processing step to run')

opt = p.parse_args()

# start wandb
wandb.init(
    project = opt.wandb_project,
    entity = "aklin",
    group = opt.wandb_group,
    name = opt.wandb_name,
)
wandb.config.update(opt)

experiment_dir = os.path.join(opt.experiments_dir, opt.experiment_name)
if (mode == 'all') or (mode == 'train'):
    # create experiment dir
    if os.path.exists(experiment_dir):
        overwrite = input("The experiment directory %s already exists. Overwrite? (y/n)"%experiment_dir)
        if not (overwrite == 'y'):
            print('Exiting.')
            quit()
        shutil.rmtree(experiment_dir)     
    os.makedirs(experiment_dir)
elif mode == 'test':
    # confirm that experiment dir already exists
    if not os.path.exists(experiment_dir):
        raise RuntimeError('Cannot run test mode: experiment directory not found!')

current_time = datetime.now()
# log current config
with open(os.path.join(experiment_dir, 'config_%s.txt' % current_time.strftime('%m_%d_%Y_%H_%M')), 'w') as f:
    for arg, val in vars(opt).items():
        f.write(arg + ' = ' + str(val) + '\n')

if (mode == 'all') or (mode == 'train'):
    # set counter_end appropriately if needed
    if opt.counter_end == -1:
        opt.counter_end = opt.num_epochs

    # log original options
    with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'wb') as opt_file:
        pickle.dump(opt, opt_file)

# load original experiment settings
with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'rb') as opt_file:
    orig_opt = pickle.load(opt_file)

# set the experiment seed
torch.manual_seed(orig_opt.seed)
random.seed(orig_opt.seed)
np.random.seed(orig_opt.seed)

dynamics_class = getattr(dynamics, orig_opt.dynamics_class)
dynamics = dynamics_class(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'})

dataset = dataio.ReachabilityDataset(
    dynamics=dynamics, numpoints=orig_opt.numpoints, 
    pretrain=orig_opt.pretrain, pretrain_iters=orig_opt.pretrain_iters, 
    tMin=orig_opt.tMin, tMax=orig_opt.tMax, 
    counter_start=orig_opt.counter_start, counter_end=orig_opt.counter_end, 
    num_src_samples=orig_opt.num_src_samples, num_target_samples=orig_opt.num_target_samples)

model = modules.SingleBVPNet(in_features=dynamics.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                             final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
model.cuda()

experiment_class = getattr(experiments, orig_opt.experiment_class)
experiment = experiment_class(model=model, dataset=dataset, experiment_dir=experiment_dir)
experiment.init_special(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(experiment_class.init_special).parameters.keys() if argname != 'self'})

if (mode == 'all') or (mode == 'train'):
    if dynamics.loss_type == 'brt_hjivi':
        loss_fn = losses.init_brt_hjivi_loss(dynamics, orig_opt.minWith, orig_opt.dirichlet_loss_divisor)
    elif dynamics.loss_type == 'brat_hjivi':
        loss_fn = losses.init_brat_hjivi_loss(dynamics, orig_opt.minWith, orig_opt.dirichlet_loss_divisor)
    else:
        raise NotImplementedError
    experiment.train(
        batch_size=orig_opt.batch_size, epochs=orig_opt.num_epochs, lr=orig_opt.lr, 
        steps_til_summary=orig_opt.steps_til_summary, epochs_til_checkpoint=orig_opt.epochs_til_ckpt, 
        loss_fn=loss_fn, clip_grad=orig_opt.clip_grad, use_lbfgs=orig_opt.use_lbfgs, adjust_relative_grads=orig_opt.adj_rel_grads,
        val_x_resolution=orig_opt.val_x_resolution, val_y_resolution=orig_opt.val_y_resolution, val_z_resolution=orig_opt.val_z_resolution, val_time_resolution=orig_opt.val_time_resolution,
        use_CSL=orig_opt.use_CSL, CSL_lr=orig_opt.CSL_lr, CSL_dt=orig_opt.CSL_dt, epochs_til_CSL=orig_opt.epochs_til_CSL, num_CSL_samples=orig_opt.num_CSL_samples, CSL_loss_frac_cutoff=orig_opt.CSL_loss_frac_cutoff, max_CSL_epochs=orig_opt.max_CSL_epochs, CSL_loss_weight=orig_opt.CSL_loss_weight, CSL_batch_size=orig_opt.CSL_batch_size)

if (mode == 'all') or (mode == 'test'):
    experiment.test(
        current_time=current_time, 
        last_checkpoint=orig_opt.num_epochs, checkpoint_dt=orig_opt.epochs_til_ckpt, 
        checkpoint_toload=opt.checkpoint_toload, dt=opt.dt,
        num_scenarios=opt.num_scenarios, num_violations=opt.num_violations, 
        set_type='BRT' if orig_opt.minWith in ['zero', 'target'] else 'BRS', control_type=opt.control_type, data_step=opt.data_step)