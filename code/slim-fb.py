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
parser.add_argument('--num_threads', type=int, default=8)

args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

n_users, n_items =tr.shape
tv = (tr + val).tocsr()
trainmat = SLIMatrix(tv)

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
savedir = os.path.join("saved_models", args.dataset_name)
best_paramset = torch.load(os.path.join(savedir, model_name))
ev_range = [5, 10, 20, 30, 50]
for test_k in ev_range:
    testkey = "test_at_%d" % test_k
    if testkey in best_paramset:
        del best_paramset[testkey]

params = {'algo': 'cd', 'simtype': 'dotp', 'nthreads': args.num_threads, 'l1r': best_paramset['l1-reg'], 'l2r': best_paramset['l2-reg']}

cnt = 0
while cnt < 5:
    model = SLIM()
    model.train(params, trainmat)
    for test_k in ev_range:
        fail = 0
        try:
            res = model.predict(trainmat, nrcmds=test_k, returnscores=True)
            w = wrapper(res)
            testkey = "test_at_%d" % test_k
            if 'param' not in best_paramset:
                best_paramset['param'] = dict()
            if testkey not in best_paramset['param']:
                best_paramset['param'][testkey] = []
            best_paramset['param'][testkey].append(ranking_metrics_at_k(w, tv, te, K=test_k))
        except:
            fail += 1
    if fail == 0:
        cnt += 1

    savedir = os.path.join("best_res", args.dataset_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    torch.save(best_paramset, os.path.join(savedir, model_name))
