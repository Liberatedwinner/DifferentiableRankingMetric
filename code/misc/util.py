import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy import sparse
import data
from implicit.als import AlternatingLeastSquares as WMF

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def get_wrapper(uv, iv, ub, ib):
    u = np.hstack([uv, ub[:, np.newaxis], np.ones_like(ub[:, np.newaxis])])
    i = np.hstack([iv, np.ones_like(ib[:, np.newaxis]), ib[:, np.newaxis]])
    w = WMF(use_gpu=False)
    w.user_factors = u
    w.item_factors = i
    return w
