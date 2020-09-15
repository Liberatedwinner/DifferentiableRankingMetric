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
parser.add_argument('--dataset_name', type=str, default='sketchfab-parsed')
parser.add_argument('--eval_metric', type=str, default='recall')
parser.add_argument('--kk', type=int, default=50)
parser.add_argument('--num_threads', type=int, default=8)
args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

n_users, n_items =tr.shape
trainmat = SLIMatrix(tr)

l1rs = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0]
l2rs = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0]

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


best = -1
tv = (tr + val).tocsr()
for l2r in l2rs:
    for l1r in l1rs:
        params = {'algo': 'cd', 'simtype': 'dotp', 'nthreads': args.num_threads, 'l1r': l1r, 'l2r': l2r}
        model = SLIM()
        model.train(params, trainmat)


        #generate top-10 recommendations
        prediction_res = model.predict(trainmat, nrcmds=args.kk, returnscores=True)
        w = wrapper(prediction_res)
        metric = ranking_metrics_at_k(w, tr, val, K=args.kk)[args.eval_metric]
        if metric > best:
            print("[%s %s@%d: %0.4f]" % (args.dataset_name, args.eval_metric, args.kk, metric), "at l1r: %0.1f l2r: %0.1f" % (l1r, l2r))
            best = metric
            __best_paramset = {
                "l1-reg": l1r,
                "l2-reg": l2r,
                "validation_measure": args.eval_metric,
                "validation_best": metric,
            }

            prediction_res = model.predict(trainmat, nrcmds=100, returnscores=True)
            w = wrapper(prediction_res)
            for test_k in [10, 30, 50, 100]:
                __best_paramset["test_at_%d" % test_k] = ranking_metrics_at_k(w, (tr + val), te, K=args.kk)
            savedir = os.path.join("saved_models", args.dataset_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            torch.save(__best_paramset, os.path.join(savedir, model_name))
