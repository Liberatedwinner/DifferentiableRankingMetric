import argparse
import os
os.environ['MKL_NUM_THREADS'] = '1'

import pickle
import misc.util
from lightfm import LightFM
import torch
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='sketchfab-parsed')
parser.add_argument('--num_threads', type=int, default=8)

args = parser.parse_args()
model_name = "warp"

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

tr = (tr + val).tocsr()
n_users, n_items = tr.shape

savedir = os.path.join("saved_models", args.dataset_name)
bp = best_paramset = torch.load(os.path.join(savedir, model_name))
ev_range = [5, 10, 20, 30, 50]

for __ in range(1):
    model = LightFM(no_components=bp["dim"],
                    learning_rate=bp["lr"],
                    item_alpha=bp["reg"],
                    user_alpha=bp["reg"],
                    max_sampled=100,
                    loss=model_name)

    model.fit(tr, epochs=bp['iter'], num_threads=args.num_threads, verbose=True)
    ub, uv = model.get_user_representations()
    ib, iv = model.get_item_representations()
    wrapper = misc.util.get_wrapper(uv, iv, ub, ib)
    for test_k in ev_range:
        testkey = "test_at_%d" % test_k
        if testkey not in best_paramset['param']:
            best_paramset['param'][testkey] = []
        best_paramset['param'][testkey].append(ranking_metrics_at_k(wrapper, tr, te, K=test_k))

    savedir = os.path.join("best_res", args.dataset_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    torch.save(best_paramset, os.path.join(savedir, model_name))
