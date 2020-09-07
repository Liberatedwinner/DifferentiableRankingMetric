import os
import itertools

os.environ['MKL_NUM_THREADS'] = '1'
import torch
import pickle
from torch.utils.data import DataLoader
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k
import models.mf
import misc.loader
import argparse
from sklearn.utils import shuffle


def warp_loss(pos, neg, n_items, margin=1.0):
    neg_highest, _ = neg.max(-1)
    impostors = torch.log(1.0 + torch.clamp(-pos.unsqueeze(-1) + neg + margin, 0).mean(-1) * n_items).detach()

    loss_per_pair = torch.clamp(-pos + neg_highest + margin, 0)  # batch_size
    return (impostors * loss_per_pair).sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
    parser.add_argument('--eval_metric', type=str, default='recall')
    parser.add_argument('--kk', type=int, default=50)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--para', type=int, default=8)
    args = parser.parse_args()
    model_name = "torch-bpr"
    infer_dot = True

    torch.cuda.set_device(args.device_id)

    with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
        (tr, val, te) = pickle.load(f)
    n_users, n_items = tr.shape

    dims = [64, 128]
    regs = [1e-3, 1e-3*5 , 1e-4, 1e-2]
    lrs = [0.1, 0.05, 0.01]
    batch_size = 8192
    rollable_params = [
        dims, regs, lrs,
    ]
    rollable_params_list = shuffle(list(itertools.product(*rollable_params)))

    print((rollable_params_list))
    n_epochs = 300
    best = -1
    poolsize = args.para

    best = -1
    curr_roll_idx = 0
    workers = []
    dataset = misc.loader.RecDataset(tr, K=2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_threads, pin_memory=True)
    for ep in itertools.count():
        added = False
        while curr_roll_idx < len(rollable_params_list) and len(workers) < poolsize:
            dim, reg, lr = rollable_params_list[curr_roll_idx]
            model = models.mf.mfrec(n_users, n_items, dim, infer_dot=infer_dot).cuda()
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
            workers.append([model, optimizer, dim, reg, lr, {"epoch": 0, "ll": 0, "last": -1, 'del': False}])
            curr_roll_idx = curr_roll_idx + 1
            added = True
            print(curr_roll_idx, 'new worker added!')
        if curr_roll_idx == len(rollable_params_list) and len(workers) == 0:
            break
        for __ in range(5):
            uu = []
            ii = []
            _i = 0
            for u, i, j in loader:
                _i += 1
                u = u.cuda()
                i = i.cuda()
                j = j.cuda()
                for model, optimizer, dim, reg, lr, dd in workers:
                    y = model.dot_fow(u, i, j)
                    pos = y[:, 0]
                    neg = y[:, 1]
                    loss = -torch.log(torch.sigmoid(pos - neg)).sum()
                    reg_term = reg * ((model.u_emb.weight[u].flatten() ** 2).sum()
                                        +(model.i_emb.weight[i].flatten() ** 2).sum()
                                        +(model.i_emb.weight[j].flatten() ** 2).sum()).sum()
                    model.zero_grad()
                    (loss + reg_term).backward()
                    optimizer.step()

        for model, optimizer, dim, reg, lr, dd in workers:
            dd['epoch'] += 5
        for worker in workers:
            model, optimizer, dim, reg, lr, dd = worker
            if dd['epoch'] <= 50:
                continue
            model.eval()
            if "-l-" in args.dataset_name:
                metric = leave_k_eval(model, tr, val, leavek=1, K=args.kk)[args.eval_metric]
            elif "712" in args.dataset_name:
                metric = ranking_metrics_at_k(model, tr, val, K=args.kk)[args.eval_metric]
            print("[epoch:%d]" % dd['epoch'], "[%sc %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric),
                  "dim: %d reg: %0.3f, lr: %0.4f" % (dim, reg, lr),
                  "best: %0.4f" % best)
            if metric > best:
                best = metric
                savedir = os.path.join("saved_models", args.dataset_name)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                best_paramset = {
                                 "eval_metric": args.eval_metric,
                                 "epoch": dd['epoch'],
                                 "lr": lr,
                                 "dim": dim,
                                 "reg": reg,
                                 'best': best}

                torch.save(best_paramset, os.path.join(savedir, model_name))
            if metric > dd['last']:
                dd['noc'] = 0
                dd['last'] = metric
            else:
                dd['noc'] += 1
            if dd['noc'] >= 3:
                dd['del'] = True
            if dd['epoch'] >= n_epochs:
                dd['del'] = True

        old_workers = [worker for worker in workers if worker[-1]['del'] is True]
        workers = [worker for worker in workers if worker[-1]['del'] is False]
        del old_workers
