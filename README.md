# DeepReach: A Deep Learning Approach to High-Dimensional Reachability
### [Project Page](http://people.eecs.berkeley.edu/~somil/index.html) | [Paper](https://arxiv.org/pdf/2011.02082.pdf)<br>

[Somil Bansal](http://people.eecs.berkeley.edu/~somil/index.html),
Claire Tomlin<br>
University of California, Berkeley

This is the official implementation of the paper "DeepReach: A Deep Learning Approach to High-Dimensional Reachability".

## Get started
You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate siren
```

## High-Level structure
The code is organized as follows:
* dataio.py loads training and testing data.
* training.py contains a generic training routine.
* modules.py contains layers and full neural network modules.
* utils.py contains utility functions.
* diff_operators.py contains implementations of differential operators.
* loss_functions.py contains loss functions for the different experiments.
* ./experiment_scripts/ contains scripts to reproduce experiments in the paper.
* ./validation_scripts/ contains scripts to reproduce figures in the paper.

## Reproducing experiments
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

## Citation
If you find our work useful in your research, please cite:
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
