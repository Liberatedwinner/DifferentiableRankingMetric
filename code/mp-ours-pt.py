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
from models.loss import detNeuralSort
from sklearn.utils import shuffle

def warp_loss(pos, neg, n_items, margin=1.0):
    neg_highest, _ = neg.max(-1)
    impostors = torch.log(1.0 + torch.clamp(-pos.unsqueeze(2) + neg.unsqueeze(1) + margin, 0).mean(-1) * n_items).detach()

    loss_per_pair = torch.clamp(-pos + neg_highest.unsqueeze(-1) + margin, 0)
    return (impostors * loss_per_pair).sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
    parser.add_argument('--eval_metric', type=str, default='recall')
    parser.add_argument('--kk', type=int, default=50)
    parser.add_argument('--pos_sample', type=int, default=3)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--infer_dot', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--para', type=int, default=8)
    args = parser.parse_args()
    model_name = args.model_name + '-%d' % args.pos_sample
    infer_dot = args.infer_dot == 1

    torch.cuda.set_device(args.device_id)

    with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
        (tr, val, te) = pickle.load(f)
    n_users, n_items = tr.shape
    dims = [16]
    regs = [0.5, 1.0]
    if infer_dot is True:
        regs = [0]
    lrs = [5 * 1e-3, 0.01]
    taus = [0.5, 1.0]
    alphas = [0.1, 0.5]
    batch_size = 8192

    rollable_params = [
        dims, regs, lrs, taus, alphas
    ]

    rollable_params_list = shuffle(list(itertools.product(*rollable_params)))

    n_epochs = 300
    best = -1
    poolsize = args.para
    best = -1
    curr_roll_idx = 0
    workers = []
    _k = args.pos_sample
    negs = 20 * _k
    dataset = misc.loader.DRMDataset(tr, n_pos=_k, n_neg=negs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_threads, pin_memory=True)
    for ep in itertools.count():
        added = False
        while curr_roll_idx < len(rollable_params_list) and len(workers) < poolsize:
            dim, reg, lr, tau, alpha = rollable_params_list[curr_roll_idx]
            model = models.mf.mfrec(n_users, n_items, dim, infer_dot=infer_dot).cuda()
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
            workers.append([model, optimizer, dim, reg, lr, tau, alpha, {"epoch": 0, "ll": 0, "last": -1, 'del': False}])
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
                i = torch.cat([i, j], dim=-1)
                for model, optimizer, dim, reg, lr, tau, alpha, dd in workers:
                    y = model.fow2(u, i)
                    pos = y[:, :_k]
                    neg = y[:, _k:]
                    p_hat = detNeuralSort(y, tau=tau, k=_k)
                    ps = p_hat.sum(1).clamp(0, 1)
                    a = ps[:, :_k]
                    b = ps[:, _k:]
                    loss1 = ((a - 1)**2).sum() + (b ** 2).sum()
                    loss = warp_loss(pos, neg, model.n_items).sum()
                    model.zero_grad()
                    if infer_dot:
                        (loss + alpha * loss1).backward()
                    else:
                        (loss + alpha * loss1 + reg * model.cov(u, i.flatten()).sum()).backward()
                    optimizer.step()
                    model.normalize(u, _target='uid')
                    model.normalize(i, _target='iid')
        for model, optimizer, dim, reg, lr, tau, alpha, dd in workers:
            dd['epoch'] += 5
        """
        __ = PPool.map(mapper, [(worker, tr, val, args.kk, args.dataset_name, args.eval_metric, args.model_name, best) for worker in workers])
        print("!?!")
        bests = [x[1] for x in __]
        metrics = [x[0] for x in __]
        best = np.max(bests)
        for metric, (model, optimizer, dim, reg, lr, tau, alpha, dd) in zip(metrics, workers):
            if metric > dd['last']:
                dd['noc'] = 0
                dd['last'] = metric
            else:
                dd['noc'] += 1
            if dd['noc'] >= 3:
                dd['del'] = True
            if dd['epoch'] >= n_epochs:
                dd['del'] = True
        """
        for worker in workers:
            model, optimizer, dim, reg, lr, tau, alpha, dd = worker
            model.eval()
            metric = ranking_metrics_at_k(model, tr, val, K=args.kk)[args.eval_metric]
            if metric > best:
                best = metric
                print("[epoch:%d]" % dd['epoch'], "[%sc %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric),
                      "alpha: %0.2f, tau: %0.2f reg: %0.3f, lr: %0.4f" % (alpha, tau, reg, lr))
                savedir = os.path.join("saved_models", args.dataset_name)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                best_paramset = {"model": model,
                                 "eval_metric": args.eval_metric,
                                 "epoch": dd['epoch'],
                                 "lr": lr,
                                 "dim": dim,
                                 "alpha": alpha,
                                 "tau": tau,
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
