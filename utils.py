import numpy as np
import torch
import os


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
