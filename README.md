# DeepReach: A Deep Learning Approach to High-Dimensional Reachability
### [Project Page](http://people.eecs.berkeley.edu/~somil/index.html) | [Paper](https://arxiv.org/pdf/2011.02082.pdf)<br>

Repository Maintainers<br>
[Albert Lin](https://www.linkedin.com/in/albertkuilin/),
[Zeyuan Feng](https://thezeyuanfeng.github.io/),
[Javier Borquez](https://javierborquez.github.io/),
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html)<br>
University of Southern California

Original Authors<br>
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html),
Claire Tomlin<br>
University of California, Berkeley

(Still to come...) The Safe and Intelligent Autonomy (SIA) Lab at the University of Southern California
is still working on an easy-to-use DeepReach Python package which will follow much of the same organizational principles as
the [hj_reachability package in JAX](https://github.com/StanfordASL/hj_reachability) from the Autonomous Systems Lab at Stanford.
The future version will include the newest tips and tricks of DeepReach developed by SIA.

(In the meantime...) This branch provides a moderately refactored version of DeepReach to facilitate easier outside research on DeepReach.

## High-Level Structure
The code is organized as follows:
* `dynamics/dynamics.py` defines the dynamics of the system.
* `experiments/experiments.py` contains generic training routines.
* `utils/modules.py` contains neural network layers and modules.
* `utils/dataio.py` loads training and testing data.
* `utils/diff_operators.py` contains implementations of differential operators.
* `utils/losses.py` contains loss functions for the different reachability cases.
* `run_experiment.py` starts a standard DeepReach experiment run.

## External Tutorial
Follow along these [tutorial slides](https://docs.google.com/presentation/d/19zxhvZAHgVYDCRpCej2svCw21iRvcxQ0/edit?usp=drive_link&ouid=113852163991034806329&rtpof=true&sd=true) to get started, or continue reading below.

## Environment Setup
Create and activate a virtual python environment (env) to manage dependencies:
```
python -m venv env
env\Scripts\activate
```
Install DeepReach dependencies:
```
pip install -r requirements.txt
```
Install the appropriate PyTorch package for your system. For example, for a Windows system with CUDA 12.1:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running a DeepReach Experiment
`run_experiment.py` implements a standard DeepReach experiment. For example, to learn the value function for the avoid Dubins3D system with parameters `goalR=0.25`, `velocity=0.6`, `omega_max=1.1`, run:
```
python run_experiment.py --mode train --experiment_class DeepReach --dynamics_class Dubins3D --experiment_name dubins3d_tutorial_run --minWith target --goalR 0.25 --velocity 0.6 --omega_max 1.1 --angle_alpha_factor 1.2 --set_mode avoid
```
Note that the script provides many common training arguments, like `num_epochs` and the option to `pretrain`. Up-to-date, documentation for these different training schemes is lacking; feel free to reach out to the lab for questions. `use_CSL` is an experimental training option (similar in spirit to actor-critic methods) being developed by SIA for improved value function learning. 

## Monitoring a DeepReach Experiment
Results for the Dubins3D system specified in the above section can be found in this [online WandB project](https://wandb.ai/aklin/DeepReachTutorial).
We highly recommend users use the `--use_wandb` flag to log training progress to the free cloud-based Weights & Biases AI Developer Platform, where it can be easily viewed and shared.

Throughout training, the training loss curves, value function plots, and model checkpoints are saved locally to `runs/experiment_name/training/summaries` and `runs/experiment_name/training/checkpoints` (and to WandB, if specified).

## Defining a Custom System
Systems are defined in `dynamics/dynamics.py` and inherit from the abstract `Dynamics` class. At a minimum, users must define:
* `__init(self, ...)__`, which must call `super().__init__(loss_type, set_mode, state_dim, ...)`
* `state_test_range(self)`, which specifies the state space that will be visualized in training plots
* `dsdt(self, state, control, disturbance)`, which implements the forward dynamics
* `boundary_fn(self, state)`,  which implements the boundary function that implicitly represents the target set
* `hamiltonian(self, state, dvds)`, which implements the system's hamiltonian
* `plot_config(self)`, which specifies the state slices and axes visualized in training plots

## Citation
If you find our work useful in your research, please cite:
```
@software{deepreach2024,
  author = {Lin, Albert and Feng, Zeyuan and Borquez, Javier and Bansal, Somil},
  title = {{DeepReach Repository}},
  url = {https://github.com/smlbansal/deepreach},
  year = {2024}
}
```

```
@inproceedings{bansal2020deepreach,
    author = {Bansal, Somil
              and Tomlin, Claire},
    title = {{DeepReach}: A Deep Learning Approach to High-Dimensional Reachability},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year={2021}
}
```

## Contact
If you have any questions, please feel free to email the authors.