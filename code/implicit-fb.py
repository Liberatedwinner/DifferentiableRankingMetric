import argparse
import os
import pickle
from implicit.als import AlternatingLeastSquares as WMF
from implicit.bpr import BayesianPersonalizedRanking as BPR
import eval.rec_eval
import torch
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k

os.environ['MKL_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='sketchfab-parsed')
parser.add_argument('--model_name', type=str, default='wmf')
parser.add_argument('--num_threads', type=int, default=8)


args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

tr = (tr + val).tocsr()
n_users, n_items = tr.shape
tr.data[:] = 1

bp = best_paramset = torch.load(os.path.join("saved_models", args.dataset_name, args.model_name), map_location='cuda:0')
print(bp)

"""
    "validation_measure" : args.eval_metric,
    "dim": dim,
    "reg": reg,
    "epoch": (i + 1) * iter,
    "best": metric
    "conf": -> exists when WMF used
    "lr": -> exists when BPR used
"""
dim, reg, iter = bp['dim'], bp['reg'], bp['epoch']

if args.model_name == 'wmf':
    conf = bp['conf']
    model = WMF(factors=dim,
                regularization=reg,
                use_cg=False,
                iterations=iter,
                use_gpu=False,
                num_threads=args.num_threads)
    model.fit(conf * tr.T, show_progress=True)

elif args.model_name == 'bpr':
    lr = bp['lr']
    model = BPR(factors=dim,
                learning_rate=lr,
                regularization=reg,
                iterations=iter,
                use_gpu=False,
                num_threads=args.num_threads)
    model.fit(tr.T, show_progress=True)

ev_range = [5, 10, 20, 30, 50]

for test_k in ev_range:
    testkey = "test_at_%d" % test_k
    if testkey not in best_paramset:
        best_paramset[testkey] = []
        best_paramset[testkey].append(ranking_metrics_at_k(model, tr, te, K=test_k))
savedir = os.path.join("best_res", args.dataset_name)
if not os.path.exists(savedir):
    os.makedirs(savedir)
torch.save({"model": model, "param": best_paramset}, os.path.join(savedir, args.model_name))
