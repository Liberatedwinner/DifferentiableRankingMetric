import argparse
import os
import pickle
from implicit.als import AlternatingLeastSquares as WMF
from implicit.bpr import BayesianPersonalizedRanking as BPR
import eval.rec_eval
import torch

os.environ['MKL_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
parser.add_argument('--model_name', type=str, default='wmf')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--num_threads', type=int, default=8)


args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

tr.data[:] = 1.0
dims = [16, 32, 64, 128]
confs = [1, 3, 5, 10, 15, 20]
regs = [1e-4, 1e-3, 3 * 1e-3, 0.01, 0.03]
lrs = [1e-4, 1e-3, 5e-3, 0.01, 0.03, 0.05]

best = -1
for dim in dims:
    for reg in regs:
        # bpr일 경우 lr, wmf면 weight
        if args.model_name == 'wmf':
            toiter = confs
        else:
            toiter = lrs
        for w in toiter:
            iter = 3
            losses = []
            last_map =-1
            if args.model_name == 'wmf':
                model = WMF(factors=dim,
                            regularization=reg,
                            use_cg=False,
                            iterations=iter,
                            use_gpu=False,
                            num_threads=args.num_threads)
            else:
                iter = 5
                model = BPR(factors=dim,
                            learning_rate=w,
                            regularization=reg,
                            iterations=iter,
                            use_gpu=False,
                            num_threads=args.num_threads)
            noc = 0
            epochs = 10 if args.model_name == 'wmf' else 60
            last = -1
            for i in range(epochs):

                model.fit(w * tr.T, show_progress=False)
                if  args.model_name == 'bpr' and epochs < 10:
                    continue
                metric = eval.rec_eval.ranking_metrics_at_k(model, tr, val, K=args.kk)[args.eval_metric]
                print("[%s %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric), 
                      "at (dim: %d w: %0.4f reg: %0.4f)" % (dim, w, reg))
                if metric > best:
                    best = metric
                    best_paramset ={
                        "metric": args.eval_metric,
                        "dim": dim,
                        "reg": reg,
                        "iter": (i + 1) * iter,
                        "best": metric,
                        "user_factors": model.user_factors,
                        "item_factors": model.item_factors
                    }
                    if args.model_name == 'wmf':
                        best_paramset['conf'] = w
                    else:
                        best_paramset['lr'] = w

                    savedir = os.path.join("saved_models", args.dataset_name)
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    torch.save(best_paramset, os.path.join(savedir, args.model_name))

                if metric > last:
                    noc = 0
                    last = metric
                else:
                    noc += 1
                if noc >= 3:
                    break
