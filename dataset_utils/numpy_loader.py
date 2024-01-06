import torch
import numpy as np


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample
