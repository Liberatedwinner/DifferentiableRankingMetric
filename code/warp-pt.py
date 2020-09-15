import argparse
import os
os.environ['MKL_NUM_THREADS'] = '1'

import pickle
import misc.util
from lightfm import LightFM
import torch
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
parser.add_argument('--model_name', type=str, default='warp')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--num_threads', type=int, default=8)

args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

dims = [16]
regs = [1e-4, 1e-3]
lrs = [1e-3, 1e-4 * 5]

best = -1
for dim in dims:
    for reg in regs:
        for lr in lrs:
            model = LightFM(no_components=dim,
                            learning_rate=lr,
                            item_alpha=reg,
                            user_alpha=reg,
                            max_sampled=20,
                            loss=args.model_name)
            iter = 5
            noc = 0
            epochs = 20
            last = -1
            for i in range(epochs):
                model.fit_partial(tr, epochs=iter, num_threads=args.num_threads)
                ub, uv = model.get_user_representations()
                ib, iv = model.get_item_representations()
                wrapper = misc.util.get_wrapper(uv, iv, ub, ib)
                metric = ranking_metrics_at_k(wrapper, tr, val, K=args.kk, num_threads=args.num_threads)[args.eval_metric]
                print("[%s %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric),
                      "at (dim: %d lr: %0.4f reg: %0.4f)" % (dim, lr, reg))
                if metric > best:
                    best = metric
                    best_paramset ={
                        "metric": args.eval_metric,
                        "dim": dim,
                        "reg": reg,
                        "lr": lr,
                        "iter": (i + 1) * iter,
                        "best": metric,
                        "user_factors": wrapper.user_factors,
                        "item_factors": wrapper.item_factors
                    }
                    savedir = os.path.join("saved_models", args.dataset_name)
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    torch.save(best_paramset, os.path.join(savedir, args.model_name))

                if metric >= last:
                    noc = 0
                    last = metric
                else:
                    noc += 1
                if noc >= 3:
                    break
