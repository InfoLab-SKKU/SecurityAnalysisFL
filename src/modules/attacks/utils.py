import os
import random

import numpy as np
import torch

def get_ar_params(num_classes, file_path=None):
    """
    Load AR parameter lists from 'file_path' if it exists,
    otherwise generate them randomly and save.

    Generate 3*3*3 kernels for each class.
    """
    #TODO add seed param for reproducibility
    if file_path is None or not os.path.exists(file_path) :
        b_list = []
        for _ in range(num_classes):
            b = torch.randn((3, 3, 3))
            for c in range(3):
                b[c][2][2] = 0
                b[c] /= torch.sum(b[c])
            b_list.append(b.numpy())
    else:
        data = np.load(file_path, allow_pickle=True)
        b_list = data["b_list"]  # This should be a numpy object array
        print(f"Loaded AR parameters from {file_path}")



    b_list = torch.tensor(b_list).float()

    return b_list
