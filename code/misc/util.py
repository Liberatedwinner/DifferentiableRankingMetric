import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy import sparse
import data
from implicit.als import AlternatingLeastSquares as WMF

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    #row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : 1 for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def get_wrapper(uv, iv, ub, ib):
    u = np.hstack([uv, ub[:, np.newaxis], np.ones_like(ub[:, np.newaxis])])
    i = np.hstack([iv, np.ones_like(ib[:, np.newaxis]), ib[:, np.newaxis]])
    w = WMF(use_gpu=False)
    w.user_factors = u
    w.item_factors = i
    return w