import os
import itertools

os.environ['MKL_NUM_THREADS'] = '1'
import torch
import pickle
from torch.utils.data import DataLoader
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k
import models.mf
import models.ae
import misc.loader
import argparse
from models.loss import detNeuralSort
from sklearn.utils import shuffle
import argparse
from misc.loader import AEDataset
from torch.utils.data import DataLoader
from misc.util import naive_sparse2tensor


parser = argparse.ArgumentParser()
model_name = 'cdae'
parser.add_argument('--dataset_name', type=str, default='sketchfab-parsed')
parser.add_argument('--device_id', type=int, default=0)

args = parser.parse_args()
torch.cuda.set_device(args.device_id)

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

tr = (tr + val).tocsr()

n_users, n_items = tr.shape

savedir = os.path.join("saved_models", args.dataset_name)
best_paramset = bp = torch.load(os.path.join(savedir, model_name))

epoch, dim, dropout, reg, lr = bp['epoch'], bp['dim'], bp['dropout'], bp['reg'], bp['lr']
batch_size = 500
ae_dataset = AEDataset(tr)
model = models.ae.CDAE(dim + [n_items], n_users=n_users, dropout=dropout).cuda()

ev_range = [5, 10, 20, 30, 50]
loader = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
for __ in range(epoch):
    print(__, epoch)
    for uid, rowl in (loader):
        row = rowl.float().cuda()
        uid = uid.cuda()
        scores = model.forward(row, uid)
        loss = models.loss.MSELoss(row, scores)
        model.zero_grad()
        loss.mean().backward()
        optimizer.step()

model = model.eval()
wrapper = models.ae.implicitWrapper(model, naive_sparse2tensor, vae=False)

for test_k in ev_range:
    testkey = "test_at_%d" % test_k
    if 'param' not in best_paramset.keys():
        best_paramset['param'] = dict()
    if testkey not in best_paramset['param']:
        best_paramset['param'][testkey] = []
    best_paramset['param'][testkey].append(ranking_metrics_at_k(wrapper, tr, te, K=test_k))
best_paramset['model'] = model
savedir = os.path.join("best_res", args.dataset_name)
if not os.path.exists(savedir):
    os.makedirs(savedir)
torch.save(best_paramset, os.path.join(savedir, model_name))
