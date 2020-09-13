import argparse
import os
os.environ['MKL_NUM_THREADS'] = '1'

import pickle
import torch
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k
from scipy.sparse import dok_matrix
from SLIM import SLIM, SLIMatrix
import numpy as np
model_name = 'slim'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--num_threads', type=int, default=8)

args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

n_users, n_items =tr.shape
trainmat = SLIMatrix(tr)

l1rs = [1.0, 3.0, 5.0, 10.0, 20.0]
l2rs = [0.1, 0.5, 1.0]

"""
class wrapper():
    def __init__(self, spmat):
        self.csr = spmat.tocsc()

    def rank_items(self, user, tr, val):
        scores = (tr[user] * self.csr[:, val]).todense().A1
        i = np.argsort(-scores)
        s = scores[i].tolist()
        i = val[i].tolist()
        return list(zip(i, s))

"""

class wrapper():
    def __init__(self, pred):
        self.out, self.outscores = pred

    def rank_items(self, user, tr, val):
        scores = (tr[user] * self.csr[:, val]).todense().A1
        i = np.argsort(-scores)
        s = scores[i].tolist()
        i = val[i].tolist()
        return list(zip(i, s))

    def recommend(self, userid, user_items, N=10):
        i = self.out[userid].tolist()
        s = self.out[userid].tolist()
        return sorted(list(zip(i, s)), key= lambda x: -x[1])


## temp_fname = "tmp" + model_name + ".tmp"
best = -1
tv = (tr + val).tocsr()
for l2r in l2rs:
    for l1r in l1rs:
        params = {'algo': 'cd', 'simtype': 'dotp', 'nthreads': args.num_threads, 'l1r': l1r, 'l2r': l2r}
        model = SLIM()
        model.train(params, trainmat)
        """
        model.save_model(modelfname=temp_fname, mapfname='map.csr')
        spmat = dok_matrix((n_items, n_items), dtype=np.float32)
        with open(temp_fname, 'r') as f:
            l = "!"
            idx = 0
            while l != "":
                l = f.readline()
                line = l.strip().split()
                i = [int(x) for (k, x) in enumerate(line) if k % 2 == 0]
                v = [float(x) for (k, x) in enumerate(line) if k % 2 == 1]
                for key, vv in zip(i, v):
                    spmat[idx, key] = vv
                idx += 1
        w = wrapper(spmat)
        metric = leave_k_eval(w, tr, val, leavek=1, K=args.kk)[args.eval_metric]
        """

        #generate top-10 recommendations
        prediction_res = model.predict(trainmat, nrcmds=args.kk, returnscores=True)
        w = wrapper(prediction_res)
        """

        """
        metric = ranking_metrics_at_k(w, tr, val, K=args.kk)[args.eval_metric]
        if metric > best:
            print("[%s %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric), "at l1r: %0.1f l2r: %0.1f" % (l1r, l2r))
            best = metric
            __best_paramset = {
                "l1-reg": l1r,
                "l2-reg": l2r,
                "metric": args.eval_metric,
                "best": metric,
            }
            """
            for test_k in [5, 10, 20, 30]:
                __best_paramset["test_at_%d" % test_k] = leave_k_eval(w, tr, te, leavek=1, K=args.kk)
            savedir = os.path.join("saved_models", args.dataset_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            torch.save(__best_paramset, os.path.join(savedir, model_name))
            """

            prediction_res = model.predict(trainmat, nrcmds=100, returnscores=True)
            w = wrapper(prediction_res)
            for test_k in [10, 30, 50, 100]:
                __best_paramset["test_at_%d" % test_k] = ranking_metrics_at_k(w, (tr + val), te, K=args.kk)
            savedir = os.path.join("saved_models", args.dataset_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            torch.save(__best_paramset, os.path.join(savedir, model_name))
