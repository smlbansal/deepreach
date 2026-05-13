# HJI-VI Branch: MADR & DeepReach MPC Integration

**Overview** This branch integrates **MADR** and **DeepReach MPC** to provide robust value function approximations for both single-player optimal control and two-player, zero-sum differential games. The code structure follows DeepReach convention for ease of use. Please refer to the external tutorial slides and papers below for more details.

## Publications & Resources
**MADR: MPC-guided Adversarial DeepReach** [[Project Website]](https://land-dev.github.io/madr/) | [[arXiv]](https://arxiv.org/abs/2510.18845)

Authors: [Ryan Teoh](https://github.com/Land-dev/ryanteoh)\*, [Sander Tonkens](https://stonkens.github.io/)\*, [William Sharpless](https://willsharpless.github.io/), [Aijia Yang](https://annieyaj.github.io/), [Zeyuan Feng](https://thezeyuanfeng.github.io/), [Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html), [Sylvia Herbert](https://sylviaherbert.com/)

\* Authors contributed equally. Affiliations: UCLA, UC San Diego, Stanford University.

**Bridging Model Predictive Control and Deep Learning for Scalable Reachability Analysis** or **DeepReach MPC** [[arXiv]](https://arxiv.org/pdf/2505.03830)


Authors: [Zeyuan Feng](https://thezeyuanfeng.github.io/), Le Qiu, [Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html)

**Acknowledgement**: This repo is built on [DeepReach](https://github.com/smlbansal/deepreach). Thanks all the maintainers for the supports! <br>
[Albert Lin](https://www.linkedin.com/in/albertkuilin/),
[Zeyuan Feng](https://thezeyuanfeng.github.io/),
[Javier Borquez](https://javierborquez.github.io/),
[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html)<br>

## High-Level Structure
The code is organized as follows:
* `dynamics/dynamics.py` defines the example dynamics of the system.
* `experiments/experiments.py` contains generic training routines.
* `utils/MPC.py` contains the MPC class for the different reachability cases.
* `utils/modules.py` contains neural network layers and modules.
* `utils/dataio.py` loads training and testing data.
* `utils/diff_operators.py` contains implementations of differential operators.
* `utils/losses.py` contains loss functions for the different reachability cases.
* `utils/error_evaluators.py` contains the helper functions for formal verification.
* `utils/quaternion.py` contains the helper functions for quaternion computation.
* `run_experiment.py` starts a standard DeepReach experiment run.
* `run_experiment.sh` contains example commands to train and validate DeepReach with MPC.


## Environment Setup
Create and activate a virtual python environment (env) in the DeepReach_MPC folder to manage dependencies:
```
python -m venv env
```
Activate virtual environment
```
source env/bin/activate # Linux user
env\Scripts\activate # Windows user
```

Install DeepReach dependencies:
```
pip install -r requirements.txt
```
Install the appropriate PyTorch package for your system. For example, for a Windows system with CUDA 12.1:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## External Tutorial
Follow along these [tutorial slides](https://docs.google.com/presentation/d/1qLU4i1aBQR58G-FiyGb-l9IycMWoJlgq/edit?usp=sharing&ouid=112832011741826436488&rtpof=true&sd=true) to get started, or continue reading below. Currently the tutorial slides include the instruction for writing your own reachability problems, training the network for BRTs, and verifying the BRTs. More tutorials are coming soon.

## Running a DeepReach Experiment
Here we take the first example from the [tutorial slides](https://docs.google.com/presentation/d/1qLU4i1aBQR58G-FiyGb-l9IycMWoJlgq/edit?usp=sharing&ouid=112832011741826436488&rtpof=true&sd=true), that learns the HJB-VI (optimal control) value function for the avoid Vertical Drone 2D system. Make sure to replace YOUR_WANDB_NAME with your wandb account before running:
```
python run_experiment.py --mode train --experiment_name VD --dynamics_class VertDrone2D --tMax 1.2 --pretrain --pretrain_iters 1000 --num_epochs 10000 --counter_end 6000 --num_nl 128 --lr 3e-5 --num_iterative_refinement 10 --MPC_batch_size 100 --num_MPC_batches 10 --num_MPC_data_samples 100 --numpoints 10000 --time_till_refinement 0.24 --use_wandb --wandb_project MPC --wandb_name VD --wandb_group VertDrone2D --wandb_entity YOUR_WANDB_NAME
```

Alternatively, you can train the vertical drone system with disturbance to learn the HJI-VI solution:
```
run_experiment.py --mode train --experiment_name PVD --dynamics_class VertDrone2DWithDist --tMax 1.2 --pretrain --pretrain_iters 1000 --num_epochs 10000 --counter_end 9000 --num_nl 128 --lr 5e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 10 --num_MPC_data_samples 500 --numpoints 20000 --epochs_til_ckpt 1000 --use_wandb --wandb_project Drone2DDist --wandb_group Drone2D --wandb_entity YOUR_WANDB_NAME --max_disturbance 0.1 --not_pretrain_MPC
```

Note that the script provides many common training arguments, like `num_epochs` and the option to `pretrain`. Please refer to the [tutorial slides](https://docs.google.com/presentation/d/1qLU4i1aBQR58G-FiyGb-l9IycMWoJlgq/edit?usp=sharing&ouid=112832011741826436488&rtpof=true&sd=true) and `run_experiment.py` for more details.

For verifying the learned value function, run:
```
python run_experiment.py --mode test --experiment_name VD --checkpoint_toload -1 --data_step run_basic_recovery
```

For visualizing the verification results, run:
```
python run_experiment.py --mode test --experiment_name VD --checkpoint_toload -1 --data_step plot_basic_recovery
```
The plot will be available at `./runs/VD/basic_BRTs.png`.


For any question, please feel free to raise an issue. 


---

## Citation

If you find our methods or code useful in your research, please consider citing:

```bibtex
@misc{teoh2025madrmpcguidedadversarialdeepreach,
      title={MADR: MPC-guided Adversarial DeepReach}, 
      author={Ryan Teoh and Sander Tonkens and William Sharpless and Aijia Yang and Zeyuan Feng and Somil Bansal and Sylvia Herbert},
      year={2025},
      eprint={2510.18845},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.18845}, 
}

@misc{feng2025bridgingmodelpredictivecontrol,
      title={Bridging Model Predictive Control and Deep Learning for Scalable Reachability Analysis}, 
      author={Zeyuan Feng and Le Qiu and Somil Bansal},
      year={2025},
      eprint={2505.03830},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.03830}, 
}

@misc{bansal2020deepreachdeeplearningapproach,
      title={DeepReach: A Deep Learning Approach to High-Dimensional Reachability}, 
      author={Somil Bansal and Claire Tomlin},
      year={2020},
      eprint={2011.02082},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2011.02082}, 
}
```
If you find the verification tool useful in your research, please consider citing:
```bibtex
@InProceedings{pmlr-v242-lin24a,
  title = 	 {Verification of neural reachable tubes via scenario optimization and conformal prediction},
  author =       {Lin, Albert and Bansal, Somil},
  booktitle = 	 {Proceedings of the 6th Annual Learning for Dynamics &amp; Control Conference},
  pages = 	 {719--731},
  year = 	 {2024},
  editor = 	 {Abate, Alessandro and Cannon, Mark and Margellos, Kostas and Papachristodoulou, Antonis},
  volume = 	 {242},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {15--17 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v242/lin24a/lin24a.pdf},
  url = 	 {https://proceedings.mlr.press/v242/lin24a.html},
}

```

For any questions, please feel free to open an issue.