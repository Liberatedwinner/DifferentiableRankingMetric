import itertools
import os
os.environ['MKL_NUM_THREADS'] = '1'
import pickle
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k

import models.loss
import models.ae
import torch
from misc.loader import AEDataset
from torch.utils.data import DataLoader
from misc.util import naive_sparse2tensor, naive_sparse2tensor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-new')
parser.add_argument('--model_name', type=str, default='cdae')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--num_threads', type=int, default=1)

args = parser.parse_args()
torch.cuda.set_device(args.device_id)

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)


n_users, n_items = tr.shape
batch_size = 500
dims = [[200], [600]]
if args.model_name == 'cdae':
    dims = [x for x in dims if len(x) == 1]

lrs = [1e-3, 5.0 * 1e-4, 1e-4]
dropouts = [0.2, 0.5, 0.8]
if args.model_name == 'vae':
    lambs = [0]
    _anneal_caps = [0.2, 0.3, 0.4]
else:
    lambs = [0]
    _anneal_caps = [0]

if args.dataset_name.startswith("ml-1m"):
    total_anneal_steps = 10000
elif args.dataset_name.startswith("melon"):
    total_anneal_steps = 200000
else:
    total_anneal_steps = 50000

n_epochs = 300
param_search_list = list(itertools.product(dims, dropouts, lambs, lrs, _anneal_caps))
ae_dataset = AEDataset(tr)

best = -1
for dim, dropout, lamb, lr, _anneal_cap in param_search_list:
    print("dim", dim, 'dropout %0.2f'% dropout, 'anneal_cap %0.2f' % _anneal_cap)
    loader = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if args.model_name == 'vae':
        model = models.ae.MultiVAE(dim + [n_items], dropout=dropout).cuda()
    elif args.model_name == 'dae':
        model = models.ae.MultiDAE(dim + [n_items], n_users=-1, dropout=dropout).cuda()
    elif args.model_name == 'cdae':
        model = models.ae.MultiDAE(dim + [n_items], n_users=n_users, dropout=dropout).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)
    noc = 0
    update_count= 0
    last = -1
    for epoch in range(1, n_epochs + 1):
        tot_loss = 0
        model = model.train()
        losses = []
        for uid, rowl in (loader):
            row = rowl.float().cuda()
            uid = uid.cuda()
            if args.model_name == 'vae':
                scores, mu, logvar = model.forward(row)
                anneal_cap = min(_anneal_cap, 1. * update_count / total_anneal_steps)
                loss = models.ae.loss_function(scores, row, mu, logvar, anneal=anneal_cap)
            elif args.model_name == 'dae':
                scores = model.forward(row, uid)
                loss = models.loss.MultinomialLoss(row, scores)
            elif args.model_name == 'cdae':
                scores = model.forward(row)
                loss = models.loss.MSELoss(row, scores)

            update_count += 1
            model.zero_grad()
            loss.mean().backward()
            optimizer.step()
            losses.append(loss.detach())

        if (epoch % 1 == 0):
            loss_mean = torch.cat(losses).mean()
            model = model.eval()
            if "-l-" in args.dataset_name:
                metric = leave_k_eval(model, tr, val, leavek=1, K=args.kk)[args.eval_metric]
            elif "712" in args.dataset_name:
                wrapper = models.ae.implicitWrapper(model, naive_sparse2tensor, vae=args.model_name != 'cdae')
                metric = ranking_metrics_at_k(wrapper, tr, val, K=args.kk)[args.eval_metric]

            if metric >= best:
                best = metric
                print("[%sc %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric),
                      "at dim:", dim, "drop: %0.2f lamb: %0.4f, lr: %0.4f, anneal %0.4f" % (dropout, lamb, lr, _anneal_cap))
                savedir = os.path.join("saved_models", args.dataset_name)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                best_paramset = {"model": model,
                                 "eval_metric": args.eval_metric,
                                 "epoch": epoch,
                                 "batch_size": batch_size,
                                 "lr": lr,
                                 "dim": dim,
                                 "lamb": lamb,
                                 'best': best,
                                 "anneal_cap": _anneal_cap,
                                 "dropout": dropout}
                """
                for test_k in [5, 10, 20, 30]:
                    if "-l-" in args.dataset_name:
                        best_paramset["test_at_%d" % test_k] = leave_k_eval(model, tr, te, leavek=1, K=args.kk)
                    if "712" in args.dataset_name:
                        best_paramset["test_at_%d" % test_k] = ranking_metrics_at_k(wrapper, tr, te, K=args.kk)
                """
                torch.save(best_paramset, os.path.join(savedir, args.model_name))

            if metric >= last:
                noc = 0
                last = metric
            else:
                noc += 1
            if noc >= 5:
                break
