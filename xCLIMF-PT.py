import concurrent.futures
import importlib
import itertools
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch
from implicit.als import AlternatingLeastSquares as WMF
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import misc.util as util
import models
import neuralsort.pl as pl
from eval.rec_eval import *
from misc.loader import AEDataset, RecDataset
from misc.util import *
from models.loss import *
from models.loss import neuPrecLoss
from models.xclimf import *

os.environ['MKL_NUM_THREADS'] = '1'



parser = argparse.ArgumentParser(description='Model union')
parser.add_argument('--dataset_name', type=str,
                    help='a dataset which is needed', default='ml-1m-new')
parser.add_argument('--model_name', type=str,
                    help='a model we use to run', default='neumf')
parser.add_argument('--cuda_switch', type=bool,
                    help='turn CUDA on or off', default='False')
argments = parser.parse_args()
dataset_name = argments.dataset_name


with open("data/parsed/%s" % dataset_name, 'rb') as f:
    (tr_users, val_users, te_users, train_data, val_tr, val_te, te_tr, te_te) = pickle.load(f)


trtr = (train_data + val_tr + te_tr).tocoo()

def print_mrr(i, objective, U, V, tr, te, params):
    if (i + 1) % 10:
        return
    x = WMF()
    x.user_factors = U
    x.item_factors = V
    curr_map = ranking_metrics_at_k(x, tr, te, K=10)['map']
    del x
    if curr_map > params['last_best']:
        print("update!")
        last_map = curr_map
        params['last_best'] = curr_map
        note = {
            'U': U,
            'V': V,
            "dim": params['dims'],
            "reg": params['lambda'],
            "lr": params['gamma'],
            "iter": (i + 1),
            'map10': curr_map
        }
        return note
    else:
        return None


dims = [10, 20, 30]
regs = [1e-3, 1e-4]
lrs = [1e-3, 5*1e-4, 1e-4]

ttt_list = shuffle(list(itertools.product(*[dims, regs, lrs])))

last_mp = -1
def tmp(x):
    dim, lr, reg = x
    params = {
          "dims": dim,
          "lambda": reg,
          "gamma": lr,
          "iters": 200,
          "last_best": 0,
          "nicrow": 0}
    (U, V), note = gradient_ascent(trtr.tocsr(), val_te.tocsr(), params, foreach=print_mrr)
    return ((U, V), note)

ret = []
with concurrent.futures.ProcessPoolExecutor(8) as executor:
    for _ in zip(executor.map(tmp, ttt_list)):
        ret.push_back(_)
        torch.save(ret, os.path.join(
            "saved_models", dataset_name, "xclmf"))
