import itertools
import os
os.environ['MKL_NUM_THREADS'] = '1'
import pickle

import torch
from sklearn.utils import shuffle
import argparse
from models.neumf import neuMF
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions
from eval.rec_eval import leave_k_eval

model_name = 'spot-neumf'

n_users, n_items = tr.shape

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--num_threads', type=int, default=8)

args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

tr = tr.tocoo()
data =Interactions(tr.row, tr.col)

dims = [64, 128]
regs = [1e-4, 1e-3, 5 * 1e-4]
lrs = [1e-4, 5 * 1e-4, 1e-3, 5 * 1e-3]
batch_sizes = [1024, 2048]
param_search_list = shuffle(list(itertools.product(*[dims, regs, lrs, batch_sizes])))


best = -10
iter = 5
for cnt, (dim, reg, lr, loss, batch_size) in enumerate(param_search_list):
    noc = 0
    last = -1
    model = ImplicitFactorizationModel(
        n_iter=iter,
        embedding_dim=dim,
        loss=loss,
        l2=reg,
        learning_rate=lr,
        batch_size=batch_size,
        representation=neuMF(n_users, n_items, dim),
        use_cuda=True)

    for i in range(60):
        model.fit(data, verbose=False)
        metric = leave_k_eval(model, tr, val, K=args.kk)[args.eval_metric]
        savedir = os.path.join("saved_models", args.dataset_name)

        if best < metric:
            best = metric
            note = {"params": model._representation,
                    "dim": dim,
                    'metric': metric,
                    "reg": reg,
                    "lr": lr,
                    "loss": loss,
                    "batch_size": batch_size,
                    "iter": (i + 1) * iter}
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            torch.save(note, os.path.join(savedir, model_name))

        if metric >= last:
            noc = 0
            last = metric
        else:
            noc +=1
        if noc >= 5:
            break
