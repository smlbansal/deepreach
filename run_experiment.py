import wandb
import configargparse
import inspect
import os
import torch
import random
import numpy as np
import pickle

from datetime import datetime
from dynamics import dynamics
from experiments import experiments
from utils import modules, dataio, losses

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


p = configargparse.ArgumentParser()
p.add_argument(
    "-c",
    "--config_filepath",
    required=False,
    is_config_file=True,
    help="Path to config file.",
)
p.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["all", "train", "test"],
    help="Experiment mode to run (new experiments must choose 'all' or 'train').",
)

# save/load directory options
p.add_argument(
    "--experiments_dir",
    type=str,
    default="./runs",
    help="Where to save the experiment subdirectory.",
)
p.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of the experient subdirectory.",
)
p.add_argument(
    "--use_wandb", default=False, action="store_true", help="use wandb for logging"
)
use_wandb = p.parse_known_args()[0].use_wandb
if use_wandb:
    p.add_argument(
        "--wandb_project",
        type=str,
        default="deepreach_mpc",
        required=False,
        help="wandb project",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default="YOUR_WANDB_ENTITY",
        required=False,
        help="wandb entity",
    )
    p.add_argument(
        "--wandb_group",
        type=str,
        default="test_group",
        required=False,
        help="wandb group",
    )
    p.add_argument(
        "--wandb_name",
        type=str,
        default="test_run",
        required=False,
        help="name of wandb run",
    )


mode = p.parse_known_args()[0].mode

if (mode == "all") or (mode == "train"):
    p.add_argument(
        "--seed", type=int, default=0, required=False, help="Seed for the experiment."
    )

    # load experiment_class choices dynamically from experiments module
    experiment_classes_dict = {
        name: clss
        for name, clss in inspect.getmembers(experiments, inspect.isclass)
        if clss.__bases__[0] == experiments.Experiment
    }
    p.add_argument(
        "--experiment_class",
        type=str,
        default="DeepReach",
        choices=experiment_classes_dict.keys(),
        help="Experiment class to use.",
    )
    # load special experiment_class arguments dynamically from chosen experiment class
    experiment_class = experiment_classes_dict[p.parse_known_args()[0].experiment_class]
    experiment_params = {
        name: param
        for name, param in inspect.signature(
            experiment_class.init_special
        ).parameters.items()
        if name != "self"
    }
    for param in experiment_params.keys():
        p.add_argument(
            "--" + param,
            type=experiment_params[param].annotation,
            required=True,
            help="special experiment_class argument",
        )

    """------------------ parameters that you may want to change --------------------------------------"""
    # simulation data source options
    p.add_argument(
        "--numpoints",
        type=int,
        default=65000,
        help="Number of points in simulation data source __getitem__.",
    )
    p.add_argument(
        "--pretrain",
        action="store_true",
        default=False,
        required=False,
        help="Pretrain dirichlet conditions",
    )
    p.add_argument(
        "--pretrain_iters",
        type=int,
        default=2000,
        required=False,
        help="Number of pretrain iterations",
    )
    p.add_argument(
        "--tMax",
        type=float,
        default=1.0,
        required=False,
        help="End time of the simulation",
    )
    p.add_argument(
        "--counter_start",
        type=int,
        default=0,
        required=False,
        help="Defines the initial time for the curriculum training",
    )
    p.add_argument(
        "--counter_end",
        type=int,
        default=-1,
        required=False,
        help="Defines the linear step for curriculum training starting from the initial time",
    )

    # model options
    p.add_argument(
        "--deepReach_model",
        type=str,
        default="exact",
        required=False,
        choices=["exact", "vanilla", "diff"],
        help="deepreach model",
    )
    p.add_argument(
        "--pretrained_model",
        type=str,
        default="none",
        required=False,
        help="Whether to use pretrained model",
    )
    p.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        required=False,
        help="fine tune the last layer of pretrained model",
    )
    p.add_argument(
        "--num_hl",
        type=int,
        default=3,
        required=False,
        help="The number of hidden layers",
    )  # Don't recommand tuning this
    p.add_argument(
        "--num_nl",
        type=int,
        default=512,
        required=False,
        help="Number of neurons per hidden layer.",
    )  # Don't recommand tuning this unless you want to use a smaller NN for simple problems

    # training options
    p.add_argument(
        "--epochs_til_ckpt",
        type=int,
        default=1000,
        help="epochs until checkpoint is saved and validate plot is generate.",
    )
    p.add_argument("--lr", type=float, default=2e-5, help="learning rate. default=2e-5")
    p.add_argument(
        "--num_epochs", type=int, default=100000, help="Number of epochs to train for."
    )

    # MPC loss options
    p.add_argument(
        "--not_use_MPC", default=False, action="store_true", help="use MPC loss"
    )
    p.add_argument(
        "--not_refine_dataset",
        default=False,
        action="store_true",
        help="refine MPC dataset",
    )  # whether we refine the MPC dataset every H_R seconds (see paper)
    p.add_argument(
        "--MPC_finetune_lambda",
        type=float,
        default=100.0,
        help="MPC finetuning weight for False Positives",
    )  # working fine but tunable. can be lower if causing instability
    p.add_argument(
        "--num_MPC_data_samples",
        type=int,
        default=5000,
        help="Number of MPC data samples used for training",
    )  # working fine but tunable.
    p.add_argument(
        "--no_time_curr",
        default=False,
        action="store_true",
        help="use MPC loss with time curriculum",
    )  # Currently please always use --time_curr with --use_MPC
    # TODO: do a PDE weight curriculum if time_curr==False
    p.add_argument(
        "--MPC_importance_init",
        type=float,
        default=1.0,
        help="importance of MPC loss at the beginning",
    )
    p.add_argument(
        "--MPC_importance_final",
        type=float,
        default=1.0,
        help="importance of MPC loss at the end",
    )

    # MPC dataset generation
    p.add_argument(
        "--time_till_refinement",
        type=float,
        default=0.2,
        help="H_R in the paper, which is the effective MPC horizon (used when refinement_schedule is not provided)",
    )
    p.add_argument(
        "--refinement_schedule",
        type=str,
        default=None,
        help='Comma-separated list of refinement times (e.g., "0,0.05,0.10,0.15,0.2,0.4,0.6,0.8,1.0"). If provided, overrides time_till_refinement.',
    )
    p.add_argument(
        "--MPC_batch_size",
        type=int,
        default=10000,
        help="generate MPC data with N init states in a parallel manner",
    )  # working fine but tunable when GPU memory is limited
    p.add_argument(
        "--num_MPC_batches",
        type=int,
        default=30,
        help="total number of MPC batches generated. Dataset size=MPC batch size * num_batches before bootstrapping",
    )
    p.add_argument(
        "--num_MPC_perturbation_samples",
        type=int,
        default=100,
        help="Number of MPC samples",
    )  # working fine but tunable
    p.add_argument(
        "--num_iterative_refinement",
        type=int,
        default=20,
        help="Number of MPC iterative refinement steps",
    )  # working fine but tunable
    p.add_argument(
        "--MPC_dt", type=float, default=0.02, help="MPC dt"
    )  # working fine but tunable. Making it too small will increase computation time dramatically
    p.add_argument(
        "--MPC_receding_horizon", type=int, default=-1, help="MPC horizon"
    )  # -1 for direct style MPC and >0 (e.g., 2) for receding horizon MPC
    p.add_argument(
        "--MPC_style",
        type=str,
        default="direct",
        required=False,
        choices=["direct", "receding"],
        help="directly perturbing the whole traj to get the value labels v.s. doing receding horizon rollouts",
    )  # most of the time direct is the better choice
    p.add_argument(
        "--MPC_data_path",
        type=str,
        default="none",
        required=False,
        help="MPC data path, where inputs.pt and value_labels.pt exist. Note that inputs.pt is normalized. Specify when using your own dataset",
    )
    p.add_argument(
        "--no_MPC_disturbance_samples",
        default=False,
        action="store_true",
        help="not include disturbance samples for MPC optimization",
    )
    p.add_argument(
        "--not_pretrain_MPC",
        action="store_true",
        default=False,
        help="Whether to have MPC from time = 0.0, or start with only SSL",
    )

    """parameters that you probably don't need to pay attention"""
    # simulation data source options
    p.add_argument(
        "--tMin",
        type=float,
        default=0.0,
        required=False,
        help="Start time of the simulation",
    )
    p.add_argument(
        "--num_src_samples",
        type=int,
        default=3000,
        required=False,
        help="Number of source samples (initial-time samples) at each time step, only for vanilla DeepReach",
    )
    p.add_argument(
        "--num_target_samples",
        type=int,
        default=0,
        required=False,
        help="Number of samples inside the target set",
    )

    # model options
    p.add_argument(
        "--model",
        type=str,
        default="sine",
        required=False,
        choices=["sine", "tanh", "sigmoid", "relu"],
        help="Type of model to evaluate, default is sine.",
    )
    p.add_argument(
        "--model_mode",
        type=str,
        default="mlp",
        required=False,
        choices=["mlp", "rbf", "pinn"],
        help="Whether to use uniform velocity parameter",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=int,
        default=-1,
        required=False,
        help="The number of hidden layers",
    )

    # training options
    p.add_argument(
        "--steps_til_summary",
        type=int,
        default=100,
        help="epochs until summary is printed.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used during training (irrelevant, since len(dataset) == 1).",
    )
    p.add_argument("--clip_grad", default=0.0, type=float, help="Clip gradient.")
    p.add_argument("--use_lbfgs", default=False, type=bool, help="use L-BFGS.")
    p.add_argument(
        "--adj_rel_grads",
        default=True,
        type=bool,
        help="adjust the relative magnitude of the losses",
    )
    p.add_argument(
        "--dirichlet_loss_divisor",
        default=1.0,
        required=False,
        type=float,
        help="What to divide the dirichlet loss by for loss reweighting",
    )

    # MPC options
    p.add_argument(
        "--MPC_mode",
        type=str,
        default="MPC",
        required=False,
        choices=["MPC", "MPPI"],
        help="MPC or MPPI. Use MPPI only when you want the MPC data to be suboptimal and less noisy.",
    )
    p.add_argument(
        "--MPC_sample_mode",
        type=str,
        default="gaussian",
        required=False,
        choices=["gaussian", "binary"],
        help="MPC additive perturbation distribution",
    )
    p.add_argument(
        "--MPC_lambda_",
        type=float,
        default=0.1,
        help="MPPI lambda, which tunes the weighting strategy",
    )
    p.add_argument(
        "--MPC_integration_method",
        type=str,
        default="euler",
        required=False,
        choices=["euler", "rk4"],
        help="Integration method for MPC state propagation (euler or rk4)",
    )
    p.add_argument(
        "--MPC_nbr_action_repeats",
        type=int,
        default=1,
        required=False,
        help="Number of times to repeat the action",
    )

    p.add_argument(
        "--MPC_loss_type",
        type=str,
        default="l1",
        required=False,
        choices=["l1", "l2"],
        help="using l1 or l2 norm for MPC loss",
    )
    p.add_argument(
        "--aug_with_MPC_data",
        type=int,
        default=0,
        help="How many of pde samples are from the MPC traj",
    )
    p.add_argument(
        "--MPC_decay_scheme",
        type=str,
        default="exponential",
        required=False,
        choices=["exponential", "linear"],
        help="MPC loss decaying scheme",
    )

    # validation (during training) options
    p.add_argument(
        "--val_x_resolution",
        type=int,
        default=200,
        help="x-axis resolution of validation plot during training",
    )
    p.add_argument(
        "--val_y_resolution",
        type=int,
        default=200,
        help="y-axis resolution of validation plot during training",
    )
    p.add_argument(
        "--val_z_resolution",
        type=int,
        default=5,
        help="z-axis resolution of validation plot during training",
    )
    p.add_argument(
        "--val_time_resolution",
        type=int,
        default=6,
        help="time-axis resolution of validation plot during training",
    )

    # loss options
    p.add_argument(
        "--minWith",
        type=str,
        required=False,
        default="target",
        choices=["none", "zero", "target"],
        help="BRS vs BRT computation (typically should be using target for BRT)",
    )
    # min with none will yield BRS, while min with zero/target corresponding to HJB-PDE and HJB-VI for computing BRT.
    # Typically min with target works better than min with zero

    # load dynamics_class choices dynamically from dynamics module
    dynamics_classes_dict = {}
    for name, clss in inspect.getmembers(dynamics, inspect.isclass):
        if (
            issubclass(clss, dynamics.Dynamics)
            and clss != dynamics.Dynamics
            and not getattr(clss, "__abstractmethods__", None)
        ):
            dynamics_classes_dict[name] = clss
    p.add_argument(
        "--dynamics_class",
        type=str,
        required=True,
        choices=dynamics_classes_dict.keys(),
        help="Dynamics class to use.",
    )
    # load special dynamics_class arguments dynamically from chosen dynamics class
    dynamics_class = dynamics_classes_dict[p.parse_known_args()[0].dynamics_class]
    dynamics_params = {
        name: param
        for name, param in inspect.signature(dynamics_class).parameters.items()
        if name != "self"
    }
    for param in dynamics_params.keys():
        param_obj = dynamics_params[param]
        if param_obj.annotation is bool:
            p.add_argument(
                "--" + param,
                type=param_obj.annotation,
                default=False,
                help="special dynamics_class argument",
            )
        else:
            has_default = param_obj.default != inspect.Parameter.empty
            if has_default:
                p.add_argument(
                    "--" + param,
                    type=param_obj.annotation,
                    default=param_obj.default,
                    help="special dynamics_class argument",
                )
            else:
                p.add_argument(
                    "--" + param,
                    type=param_obj.annotation,
                    required=True,
                    help="special dynamics_class argument",
                )

if (mode == "all") or (mode == "test"):
    p.add_argument(
        "--dt", type=float, default=0.0025, help="The dt used in testing simulations"
    )
    p.add_argument(
        "--checkpoint_toload",
        type=int,
        default=None,
        help="The checkpoint to load for testing (-1 for final training checkpoint, None for cross-checkpoint testing",
    )
    p.add_argument(
        "--num_scenarios",
        type=int,
        default=100000,
        help="The number of scenarios sampled in scenario optimization for testing",
    )
    p.add_argument(
        "--num_violations",
        type=int,
        default=1000,
        help="The number of violations to sample for in scenario optimization for testing",
    )
    p.add_argument(
        "--control_type",
        type=str,
        default="value",
        choices=["value", "ttr", "init_ttr"],
        help="The controller to use in scenario optimization for testing",
    )
    p.add_argument(
        "--data_step",
        type=str,
        default="run_basic_recovery",
        choices=[
            "plot_ND",
            "run_basic_recovery",
            "plot_basic_recovery",
            "run_robust_recovery",
            "plot_robust_recovery",
            "eval_w_gt",
        ],
        help="The data processing step to run",
    )
    p.add_argument(
        "--gt_data_path",
        type=str,
        default="none",
        help="Folder for gt data where coords.pt and gt_values.pt exist",
    )

opt = p.parse_args()

# Process refinement schedule
if hasattr(opt, "refinement_schedule") and opt.refinement_schedule is not None:
    try:
        opt.refinement_schedule_list = [
            float(x.strip()) for x in opt.refinement_schedule.split(",")
        ]
        if not all(
            a <= b
            for a, b in zip(
                opt.refinement_schedule_list, opt.refinement_schedule_list[1:]
            )
        ):
            raise ValueError("Refinement schedule must be in ascending order")
        if opt.refinement_schedule_list[0] <= 0:
            raise ValueError("First refinement time must be positive")
    except Exception as e:
        raise ValueError(f"Invalid refinement schedule format: {e}")
elif hasattr(opt, "refinement_schedule"):
    opt.refinement_schedule_list = None

# start wandb
if use_wandb:
    wandb.init(
        project=opt.wandb_project,
        entity=opt.wandb_entity,
        group=opt.wandb_group,
        name=opt.wandb_name,
    )
    wandb.config.update(opt)

experiment_dir = os.path.join(opt.experiments_dir, opt.experiment_name)
if (mode == "train") and (opt.resume_checkpoint > 0):
    experiment_dir = experiment_dir + "_cond"
if (mode == "all") or (mode == "train"):
    # create experiment dir with versioning
    if os.path.exists(experiment_dir):
        version = 1
        versioned_dir = f"{experiment_dir}_v{version}"
        while os.path.exists(versioned_dir):
            version += 1
            versioned_dir = f"{experiment_dir}_v{version}"
        experiment_dir = versioned_dir
    os.makedirs(experiment_dir)
elif mode == "test":
    # confirm that experiment dir already exists
    if not os.path.exists(experiment_dir):
        raise RuntimeError("Cannot run test mode: experiment directory not found!")

current_time = datetime.now()
# log current config
with open(
    os.path.join(
        experiment_dir, "config_%s.txt" % current_time.strftime("%m_%d_%Y_%H_%M")
    ),
    "w",
) as f:
    for arg, val in vars(opt).items():
        f.write(arg + " = " + str(val) + "\n")

if (mode == "all") or (mode == "train"):
    # set counter_end appropriately if needed
    if opt.counter_end == -1:
        opt.counter_end = opt.num_epochs

    # log original options
    with open(os.path.join(experiment_dir, "orig_opt.pickle"), "wb") as opt_file:
        pickle.dump(opt, opt_file)

# load original experiment settings
with open(os.path.join(experiment_dir, "orig_opt.pickle"), "rb") as opt_file:
    orig_opt = pickle.load(opt_file)

# set the experiment seed
# torch.manual_seed(orig_opt.seed)
torch.manual_seed(12)
random.seed(orig_opt.seed)
np.random.seed(orig_opt.seed)

dynamics_class = getattr(dynamics, orig_opt.dynamics_class)
dynamics = dynamics_class(
    **{
        argname: getattr(orig_opt, argname)
        for argname in inspect.signature(dynamics_class).parameters.keys()
        if argname != "self"
    }
)
if (mode == "train") and (opt.resume_checkpoint > 0):
    orig_opt.counter_start = opt.resume_checkpoint
    orig_opt.pretrain = False
    orig_opt.num_epochs -= opt.resume_checkpoint

dynamics.set_model(orig_opt.deepReach_model)
if mode == "test":
    orig_opt.not_use_MPC = True
    orig_opt.no_time_curr = True


# # use single model
model = modules.SingleBVPNet(
    in_features=dynamics.input_dim,
    out_features=1,
    type=orig_opt.model,
    mode=orig_opt.model_mode,
    final_layer_factor=1.0,
    hidden_features=orig_opt.num_nl,
    num_hidden_layers=orig_opt.num_hl,
    periodic_transform_fn=dynamics.periodic_transform_fn,
)
model.to(device)
policy = None
if orig_opt.pretrained_model != "none":
    model.load_state_dict(
        torch.load(
            "./runs/%s/training/checkpoints/model_final.pth" % orig_opt.pretrained_model
        )["model"]
    )

    if orig_opt.finetune:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name == "net.net.4.0.weight" or name == "net.net.4.0.bias":
                param.requires_grad = True
            print(name, param.requires_grad)
    policy = model

elif (mode == "train") and (opt.resume_checkpoint > 0):
    experiment_original_dir = experiment_dir.split("_cond")[0]
    model.load_state_dict(
        torch.load(
            os.path.join(
                "%s/training/checkpoints/model_epoch_%04d.pth"
                % (experiment_original_dir, opt.resume_checkpoint)
            )
        )["model"]
    )
    policy = model

if dynamics.disturbance_dim == 0:
    print("using traditional MPC")
    dataset_type = dataio.ReachabilityDataset
else:
    print("using robust MPC")
    dataset_type = dataio.RobustReachabilityDataset

dataset = dataset_type(
    dynamics=dynamics,
    numpoints=orig_opt.numpoints,
    pretrain=orig_opt.pretrain,
    pretrain_iters=orig_opt.pretrain_iters,
    tMin=orig_opt.tMin,
    tMax=orig_opt.tMax,
    counter_start=orig_opt.counter_start,
    counter_end=orig_opt.counter_end,
    num_src_samples=orig_opt.num_src_samples,
    num_target_samples=orig_opt.num_target_samples,
    use_MPC=(not orig_opt.not_use_MPC),
    time_curr=(not orig_opt.no_time_curr),
    MPC_data_path=orig_opt.MPC_data_path,
    num_MPC_perturbation_samples=orig_opt.num_MPC_perturbation_samples,
    MPC_dt=orig_opt.MPC_dt,
    MPC_mode=orig_opt.MPC_mode,
    MPC_sample_mode=orig_opt.MPC_sample_mode,
    MPC_style=orig_opt.MPC_style,
    MPC_lambda_=orig_opt.MPC_lambda_,
    MPC_batch_size=orig_opt.MPC_batch_size,
    MPC_receding_horizon=orig_opt.MPC_receding_horizon,
    num_MPC_data_samples=orig_opt.num_MPC_data_samples,
    num_iterative_refinement=orig_opt.num_iterative_refinement,
    time_till_refinement=orig_opt.time_till_refinement,
    refinement_schedule=orig_opt.refinement_schedule_list,
    num_MPC_batches=orig_opt.num_MPC_batches,
    aug_with_MPC_data=orig_opt.aug_with_MPC_data,
    policy=policy,
    refine_dataset=(not orig_opt.not_refine_dataset),
    initiate_with_MPC=(not orig_opt.not_pretrain_MPC),
    add_disturbance_samples=(not orig_opt.no_MPC_disturbance_samples),
    MPC_integration_method=orig_opt.MPC_integration_method,
    MPC_nbr_action_repeats=orig_opt.MPC_nbr_action_repeats,
)

experiment_class = getattr(experiments, orig_opt.experiment_class)
experiment = experiment_class(
    model=model, dataset=dataset, experiment_dir=experiment_dir, use_wandb=use_wandb
)
experiment.init_special(
    **{
        argname: getattr(orig_opt, argname)
        for argname in inspect.signature(
            experiment_class.init_special
        ).parameters.keys()
        if argname != "self"
    }
)

if (mode == "all") or (mode == "train"):
    if dynamics.loss_type == "brt_hjivi":
        loss_fn = losses.init_brt_hjivi_loss(
            dynamics,
            orig_opt.minWith,
            orig_opt.dirichlet_loss_divisor,
            orig_opt.MPC_loss_type,
            (not orig_opt.not_use_MPC),
            MPC_finetune_lambda=orig_opt.MPC_finetune_lambda,
        )
    elif dynamics.loss_type == "brat_hjivi":
        loss_fn = losses.init_brat_hjivi_loss(
            dynamics,
            orig_opt.minWith,
            orig_opt.dirichlet_loss_divisor,
            orig_opt.MPC_loss_type,
            (not orig_opt.not_use_MPC),
            MPC_finetune_lambda=orig_opt.MPC_finetune_lambda,
        )
    else:
        raise NotImplementedError
    experiment.train(
        batch_size=orig_opt.batch_size,
        epochs=orig_opt.num_epochs,
        lr=orig_opt.lr,
        steps_til_summary=orig_opt.steps_til_summary,
        epochs_til_checkpoint=orig_opt.epochs_til_ckpt,
        loss_fn=loss_fn,
        clip_grad=orig_opt.clip_grad,
        use_lbfgs=orig_opt.use_lbfgs,
        adjust_relative_grads=orig_opt.adj_rel_grads,
        val_x_resolution=orig_opt.val_x_resolution,
        val_y_resolution=orig_opt.val_y_resolution,
        val_z_resolution=orig_opt.val_z_resolution,
        val_time_resolution=orig_opt.val_time_resolution,
        MPC_importance_init=orig_opt.MPC_importance_init,
        MPC_importance_final=orig_opt.MPC_importance_final,
        MPC_decay_scheme=orig_opt.MPC_decay_scheme,
    )

if (mode == "all") or (mode == "test"):
    experiment.test(
        current_time=current_time,
        last_checkpoint=orig_opt.num_epochs,
        checkpoint_dt=orig_opt.epochs_til_ckpt,
        checkpoint_toload=opt.checkpoint_toload,
        dt=opt.dt,
        num_scenarios=opt.num_scenarios,
        num_violations=opt.num_violations,
        set_type="BRT" if orig_opt.minWith in ["zero", "target"] else "BRS",
        control_type=opt.control_type,
        data_step=opt.data_step,
        gt_data_path=opt.gt_data_path,
    )
