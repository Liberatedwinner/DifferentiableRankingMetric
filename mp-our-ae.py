
import itertools
import os
os.environ['MKL_NUM_THREADS'] = '1'
import pickle
import numpy as np
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k

import models.loss
import models.ae
import torch
from misc.loader import AENeuDataset
from torch.utils.data import DataLoader
from misc.util import naive_sparse2tensor, naive_sparse2tensor
from models.loss import detNeuralSort, neuPrec
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-new')
parser.add_argument('--model_name', type=str, default='cdae')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--para', type=int, default=8)
args = parser.parse_args()
torch.cuda.set_device(args.device_id)

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)


n_users, n_items = tr.shape
batch_size = 500
savedir = os.path.join("saved_models", args.dataset_name)
best_paramset = bp = torch.load(os.path.join(savedir, "vae"), map_location="cpu")
_anneal_cap = bp['anneal_cap']
tobeat = bp['best']
dim, dropout, lamb, lr = bp['dim'], bp['dropout'], bp['lamb'], bp['lr']
batch_size, epoch = bp['batch_size'], bp['epoch']


taus = [0.1, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
alphas = [0.05, 0.1, 0.5]
_ks = [1, 3, 5]
lr = [1e-4 * 5]

if args.dataset_name.startswith("ml-1m"):
    total_anneal_steps = 10000
elif args.dataset_name.startswith("melon"):
    total_anneal_steps = 200000
else:
    total_anneal_steps = 50000

n_epochs = 300
rollable_params_list = shuffle(list(itertools.product(taus, alphas, _ks, lr)))
ae_dataset = AENeuDataset(tr, n_pos=np.max(_ks), n_neg=np.max(_ks) * 20)
best = -1
workers = []
curr_roll_idx = 0
poolsize = args.para
max_k = np.max(_ks)

loader = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
for ep in itertools.count():
    added = False
    while curr_roll_idx < len(rollable_params_list) and len(workers) < poolsize:
        tau, alpha, _k, lr = rollable_params_list[curr_roll_idx]

        if args.model_name == 'vae':
            model = models.ae.MultiVAE(dim + [n_items], dropout=dropout).cuda()
        elif args.model_name == 'dae':
            model = models.ae.MultiDAE(dim + [n_items], n_users=-1, dropout=dropout).cuda()
        elif args.model_name == 'cdae':
            model = models.ae.MultiDAE(dim + [n_items], n_users=n_users, dropout=dropout).cuda()
        model = model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)
        workers.append(
            [model, optimizer, tau, alpha, _k, lr, {
                "update_count": 0,
                "epoch": 0, "ll": 0, "last": -1, "del": False, "noc": 0,
            }]
        )
        curr_roll_idx = curr_roll_idx + 1
        added = True
        print(curr_roll_idx, 'new worker added!')
    if curr_roll_idx == len(rollable_params_list) and len(workers) == 0:
        break

    tot_loss = 0
    model = model.train()
    losses = []
    for pos_, neg_, uid, rowl in (loader):
        row = rowl.float().cuda()
        uid = uid.cuda()
        pos_ = pos_.cuda()
        neg_ = neg_.cuda()
        for worker in workers:
            model, optimizer, tau, alpha, _k, lr, dd = worker
            if args.model_name == 'vae':
                scores, mu, logvar = model.forward(row)
                anneal_cap = min(_anneal_cap, 1. * dd['update_count'] / total_anneal_steps)
                loss = models.ae.loss_function(scores, row, mu, logvar, anneal=anneal_cap)
            elif args.model_name == 'dae':
                scores = model.forward(row, uid)
                loss = models.loss.MultinomialLoss(row, scores)
            elif args.model_name == 'cdae':
                scores = model.forward(row)
                loss = models.loss.MSELoss(row, scores)
            _pos = pos_[:, :_k]
            _neg = neg_[:, max_k:max_k + _k * 20]
            pos = scores.gather(1, _pos)
            neg = scores.gather(1, _neg)
            y_hat = torch.cat([pos, neg], dim=-1)
            p_hat = detNeuralSort(y_hat, tau=tau, k=_k)
            ps = p_hat.sum(1).clamp(0, 1)
            a = ps[:, :_k]
            b = ps[:, _k:]
            loss1 = (((a - 1)**2).sum(-1) + (b ** 2).sum(-1)).mean()
            dd['update_count'] += 1
            model.zero_grad()
            (loss.mean()+ alpha * loss1).backward()
            #loss1.backward()
            optimizer.step()
            losses.append(loss.detach())
    for worker in workers:
        worker[-1]['epoch'] = worker[-1]['epoch'] + 1

    for worker in workers:
        model, optimizer, tau, alpha, _k, lr,  dd = worker

        if dd['epoch'] < 10:
            continue

        model = model.eval()
        wrapper = models.ae.implicitWrapper(model, naive_sparse2tensor, vae=args.model_name != 'cdae')
        metric = ranking_metrics_at_k(wrapper, tr, val, K=args.kk)[args.eval_metric]
        if metric >= best:
            best = metric
            print("tobeat:%0.3f" % tobeat, "[iter %d]\t[%sc %s@%d: %0.4f]" % (dd['epoch'], args.dataset_name, args.eval_metric, args.kk, metric),
                "at dim:", dim, "lr: %0.4f tau: %0.3f alpha: %0.2f, _k: %d" % (lr, tau, alpha, _k))
            savedir = os.path.join("saved_models", args.dataset_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            best_paramset = {"model": model,
                             "eval_metric": args.eval_metric,
                             "epoch": dd['epoch'],
                             "batch_size": batch_size,
                             "lr": lr,
                             "dim": dim,
                             "lamb": lamb,
                             'best': best,
                             "anneal_cap": _anneal_cap,
                             "dropout": dropout,
                             "tau": tau,
                             "lr": lr, 
                             "alpha": alpha,
                             "_k": _k}

            torch.save(best_paramset, os.path.join(savedir, args.model_name))

        if metric >= dd['last']:
            dd['noc'] = 0
            dd['last'] = metric
        else:
            dd['noc'] += 1
        if dd['noc'] >= 5:
            dd['del'] = True
        if dd['epoch'] >= 300:
            dd['del'] = True 

        model = model.train()
    old_workers = [worker for worker in workers if worker[-1]['del'] is True]
    workers = [worker for worker in workers if worker[-1]['del'] is False]

    del old_workers
