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
import argparse
from misc.loader import AEDataset
from torch.utils.data import DataLoader
from misc.util import naive_sparse2tensor, naive_sparse2tensor


parser = argparse.ArgumentParser()
model_name = 'vae'
parser.add_argument('--dataset_name', type=str, default='ml-1m-new')
parser.add_argument('--device_id', type=int, default=1)

args = parser.parse_args()
torch.cuda.set_device(args.device_id)

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

tr = (tr + val).tocsr()

n_users, n_items = tr.shape

savedir = os.path.join("saved_models", args.dataset_name)
best_paramset = bp = torch.load(os.path.join(savedir, model_name), map_location="cuda:%d" % args.device_id)
del bp['model']
_anneal_cap = bp['anneal_cap']
dim, dropout, lamb, lr = bp['dim'], bp['dropout'], bp['lamb'], bp['lr']
batch_size, epoch = bp['batch_size'], bp['epoch']
ae_dataset = AEDataset(tr)
model = model = models.ae.MultiVAE(dim + [n_items], dropout=dropout).cuda()

if args.dataset_name.startswith("ml-1m"):
    total_anneal_steps = 10000
elif args.dataset_name.startswith("melon"):
    total_anneal_steps = 200000
else:
    total_anneal_steps = 50000


ev_range = [5, 10, 20, 30, 50]
loader = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)
update_count = 0
for __ in range(epoch):
    print(__, epoch)
    for uid, rowl in (loader):
        row = rowl.float().cuda()
        uid = uid.cuda()
        scores, mu, logvar = model.forward(row)
        anneal_cap = min(_anneal_cap, 1. * update_count / total_anneal_steps)
        loss = models.ae.loss_function(scores, row, mu, logvar, anneal=anneal_cap)
        model.zero_grad()
        loss.mean().backward()
        optimizer.step()
        update_count += 1

model = model.eval()
wrapper = models.ae.implicitWrapper(model, naive_sparse2tensor, vae=model_name != 'cdae')

for test_k in ev_range:
    testkey = "test_at_%d" % test_k
    if testkey not in best_paramset:
        best_paramset[testkey] = []
    best_paramset[testkey].append(ranking_metrics_at_k(wrapper, tr, te, K=test_k))
best_paramset['model'] = model
savedir = os.path.join("best_res", args.dataset_name)
if not os.path.exists(savedir):
    os.makedirs(savedir)
torch.save(best_paramset, os.path.join(savedir, model_name))
