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


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='CML')
parser.add_argument('--dataset_name', type=str, default='sketchfab-parsed')
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--num_threads', type=int, default=4)

args = parser.parse_args()

torch.cuda.set_device(args.device_id)

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)
savedir = os.path.join("saved_models", args.dataset_name)


model = torch.load(os.path.join(savedir, args.model_name), map_location="cuda:%d" % args.device_id)
print(model)
model = model['model']
print(model)
tv = (tr + val).tocsr()
for kk in [10, 30, 50]:
    if "-1-" in args.model_name:
        metric = leave_k_eval(model, tv, te, leavek=1, K=kk)
    else:
        metric = ranking_metrics_at_k(model, tv, te, K=kk)
    print("@%d" % kk, '\t'.join(["%s: %0.4f" %(k, v) for (k, v) in metric.items()]))
