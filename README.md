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

This is the newest official implementation of the paper "DeepReach: A Deep Learning Approach to High-Dimensional Reachability".

## High-Level Structure
The code is organized as follows:
* dynamics/dynamics.py defines the dynamics of the system.
* experiments/experiments.py contains generic training routines.
* utils/modules.py contains layers and full neural network modules.
* utils/dataio.py loads training and testing data.
* utils/diff_operators.py contains implementations of differential operators.
* utils/losses.py contains loss functions for the different reachability cases.
* run_experiment.py starts a standard DeepReach run.

## Tutorial
Follow along these [tutorial slides](https://docs.google.com/presentation/d/19zxhvZAHgVYDCRpCej2svCw21iRvcxQ0/edit?usp=drive_link&ouid=113852163991034806329&rtpof=true&sd=true) to get started, or continue reading below.

## Getting Started
You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate siren
```

## Running DeepReach
The directory `experiment_scripts` contains one script per experiment in the paper.

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root.

To start training DeepReach for air3D, you can run:
```
CUDA_VISIBLE_DEVICES=0 python experiment_scripts/train_hji_air3D.py --experiment_name experiment_1 --minWith target --tMax 1.1 --velocity 0.75 --omega_max 3.0 --angle_alpha 1.2 --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000
```
This will regularly save checkpoints in the directory specified by the rootpath in the script, in a subdirectory "experiment_1". 

We also provide pre-trained checkpoints that can be used to visualize the results in the paper. The checkpoints can be downloaded from 
```
https://drive.google.com/file/d/18VkOTctkzuYuyK2GRwQ4wmN92WhdXtvS/view?usp=sharing
```
To visualize the trained BRTs, please run:
```
CUDA_VISIBLE_DEVICES=0 python validation_scripts/air3D_valfunc_and_BRT.py 
```

## Defining a Custom System

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