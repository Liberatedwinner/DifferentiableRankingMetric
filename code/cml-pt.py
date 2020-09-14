import os
os.environ['MKL_NUM_THREADS'] = '1'
import torch
import itertools
import pickle
from torch.utils.data import DataLoader
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k
import models.mf
import misc.loader
import argparse

model_name = "CML"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--num_threads', type=int, default=8)
parser.add_argument('--infer_dot', type=int, default=0)
parser.add_argument('--suffix', type=str, default="")
args = parser.parse_args()
infer_dot = args.infer_dot == 1

model_name = model_name + args.suffix
torch.cuda.set_device(args.device_id)

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

n_users, n_items = tr.shape
dims = [16, 32, 64, 128]
regs = [3.0, 5.0]
if infer_dot is True:
    regs = [0]
lrs = [1e-4, 1e-3, 5 * 1e-3, 0.01, 0.03, 0.05, 0.1]
batch_size = 8192
param_search_list = list(itertools.product(dims, regs, lrs))
n_epochs = 300

best = -1



def warp_loss(pos, neg, n_items, margin=1.0):
    neg_highest, _ = neg.max(-1)
    impostors = torch.log(1.0 + torch.clamp(-pos.unsqueeze(-1) + neg + margin, 0).mean(-1) * n_items).detach()

    loss_per_pair = torch.clamp(-pos + neg_highest + margin, 0)  # batch_size
    return (impostors * loss_per_pair).sum()


for dim, reg, lr in param_search_list:
    dataset = misc.loader.RecDataset(tr, K=21)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_threads)
    model = models.mf.mfrec(n_users, n_items, dim, infer_dot=infer_dot).cuda()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    noc = 0
    update_count= 0
    last = -1
    for epoch in range(1, n_epochs + 1):
        tot_loss = 0
        model = model.train()
        ll = 0
        for u, i, j in loader:
            u = u.cuda()
            i = i.cuda()
            j = j.cuda()
            y = model.forward(u, i, j)
            pos = y[:, 0]
            neg = y[:, 1:]
            loss = warp_loss(pos, neg, n_items).sum()
            model.zero_grad()
            if infer_dot is False:
                loss = loss + reg * model.cov(u, torch.cat([i, j.flatten()])).sum()
            loss.backward()
            optimizer.step()
            model.normalize(u, _target='uid')
            model.normalize(i, _target='iid')
            model.normalize(j, _target='iid')
            ll += loss.detach().cpu().numpy()

        if (epoch % 5 == 0):
            model = model.eval()

            metric = ranking_metrics_at_k(model, tr, val, K=args.kk)[args.eval_metric]
            if metric >= best:
                best = metric
                print("[%sc %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric),
                      "at dim: %d reg: %0.3f, lr: %0.4f" % (dim, reg, lr))
                savedir = os.path.join("saved_models", args.dataset_name)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                best_paramset = {"model": model,
                                 "eval_metric": args.eval_metric,
                                 "epoch": epoch,
                                 "lr": lr,
                                 "dim": dim,
                                 "reg": reg,
                                 'best': best}
                torch.save(best_paramset, os.path.join(savedir, model_name))

            if metric > last:
                noc = 0
                last = metric
            else:
                noc += 1
            if noc >= 3:
                break
