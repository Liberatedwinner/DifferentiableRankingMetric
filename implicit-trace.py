import argparse
import os
import pickle
from implicit.als import AlternatingLeastSquares as WMF
from implicit.bpr import BayesianPersonalizedRanking as BPR
import eval.rec_eval
import torch
from eval.rec_eval import leave_k_eval, ranking_metrics_at_k
from implicit import _als
from torch.utils.data import DataLoader
import misc.loader


os.environ['MKL_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
parser.add_argument('--model_name', type=str, default='wmf')
parser.add_argument('--num_threads', type=int, default=8)


args = parser.parse_args()

with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
    (tr, val, te) = pickle.load(f)

tr = (tr + val).tocsr()
n_users, n_items = tr.shape
tr.data[:] = 1

bp = best_paramset = torch.load(os.path.join("saved_models", args.dataset_name, args.model_name), map_location='cpu')
print(bp)

"""
    "metric": args.eval_metric,
    "dim": dim,
    "reg": reg,
    "iter": (i + 1) * iter,
    "best": metric,
    "user_factors": model.user_factors,
    "item_factors": model.item_factors
"""
try:
    dim, reg, iter = bp['dim'], bp['reg'], bp['iter']
except:
    dim, reg, iter = 128, bp['reg'], bp['iter']
ev_range = [10, 50]

ret = {
    "loss": [],
    "reg": [],
}
if args.model_name == 'wmf':
    conf = bp['conf']
    model = WMF(factors=dim,
                regularization=reg,
                use_cg=False,
                iterations=1,
                use_gpu=False,
                num_threads=args.num_threads)

    for i in range(101):
        model.fit(conf * tr.T)
        if ((i % 5) == 0):
            loss = _als.calculate_loss(tr, model.user_factors, model.item_factors, model.regularization, num_threads=model.num_threads)
            ret['loss'].append(loss)
            for test_k in ev_range:
                testkey = "test_at_%d" % test_k
                if testkey not in ret:
                    ret[testkey] = []
                ret[testkey].append(ranking_metrics_at_k(model, tr, te, K=test_k))
            savedir = os.path.join("trace", args.dataset_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            torch.save(ret, os.path.join(savedir, args.model_name))
elif args.model_name == 'bpr':
    lr = bp['lr']
    model = BPR(factors=dim,
                learning_rate=lr,
                regularization=reg,
                iterations=5,
                use_gpu=False,
                num_threads=args.num_threads)
    dataset = misc.loader.RecDataset(tr, K=2)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=args.num_threads)
    for iters in range(1 + (300 // 5)):
        model.fit(tr.T)
        _uf = torch.from_numpy(model.user_factors).cuda()
        _if = torch.from_numpy(model.item_factors).cuda()
        loss = 0
        for u, i, j in loader:
            u = u.flatten().cuda()
            i = i.flatten().cuda()
            j = j.flatten().cuda()
            loss = -torch.log(torch.sigmoid((_uf[u] * (_if[i] - _if[j])).sum(-1))).sum()
        loss = loss / tr.nnz
        _reg = reg * (_uf * _uf).sum() + reg * (_if * _if).sum()
        reg['reg'].append(float(_reg.detach().cpu().numpy()))
        ret['loss'].append(float(loss.detach().cpu().numpy()))
        for test_k in ev_range:
            testkey = "test_at_%d" % test_k
            if testkey not in ret:
                ret[testkey] = []
            ret[testkey].append(ranking_metrics_at_k(model, tr, te, K=test_k))
            print(ret[testkey])
        savedir = os.path.join("trace", args.dataset_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        torch.save(ret, os.path.join(savedir, args.model_name))
