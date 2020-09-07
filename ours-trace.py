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


def warp_loss(pos, neg, n_items, margin=1.0):
    neg_highest, _ = neg.max(-1)
    impostors = torch.log(1.0 + torch.clamp(-pos.unsqueeze(2) + neg.unsqueeze(1) + margin, 0).mean(-1) * n_items).detach()

    loss_per_pair = torch.clamp(-pos + neg_highest.unsqueeze(-1) + margin, 0)
    return (impostors * loss_per_pair).sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
    parser.add_argument('--pos_sample', type=int, default=3)
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--infer_dot', type=int, default=0)
    args = parser.parse_args()
    infer_dot = args.infer_dot == 1

    torch.cuda.set_device(args.device_id)

    with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
        (tr, val, te) = pickle.load(f)
    if infer_dot is False:
        model_name = "k=%d-%d" % (args.pos_sample, args.pos_sample)
    else:
        model_name = "k=%d-dot-%d" % (args.pos_sample, args.pos_sample)

    bp = best_paramset = torch.load(os.path.join("saved_models", args.dataset_name, model_name), map_location='cuda:0')

    n_users, n_items = tr.shape
    n_epochs = 300
    best = -1
    _k = args.pos_sample
    negs = 20 * _k
    batch_size = 8192
    ev_range = [10, 50]

    dataset = misc.loader.MyDataset(tr, n_pos=_k, n_neg=negs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dim, reg, lr, tau, alpha = bp['dim'], bp['reg'], bp['lr'], bp['tau'], bp['alpha']
    model = models.mf.mfrec(n_users, n_items, dim, infer_dot=infer_dot).cuda()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    ret = {"loss": []}
    for __ in range(500):
        uu = []
        ii = []
        _i = 0
        print(__)
        losses = 0 
        for u, i, j in loader:
            _i += 1
            u = u.cuda()
            i = i.cuda()
            j = j.cuda()
            i = torch.cat([i, j], dim=-1)
            y = model.fow2(u, i)
            pos = y[:, :_k]
            neg = y[:, _k:]
            p_hat = detNeuralSort(y, tau=tau, k=_k)
            ps = p_hat.sum(1).clamp(0, 1)
            a = ps[:, :_k]
            b = ps[:, _k:]
            loss1 = ((a - 1)**2).sum() + (b ** 2).sum()
            loss = warp_loss(pos, neg, model.n_items).sum()
            model.zero_grad()
            if infer_dot:
                (loss + alpha * loss1).backward()
            else:
                _loss = (loss + alpha * loss1 + reg * model.cov(u, i.flatten()).sum())
                _loss.backward()
            optimizer.step()
            model.normalize(u, _target='uid')
            model.normalize(i, _target='iid')

        if ((__ + 1) % 5) == 0: 
            ret['loss'].append(float(_loss.detach().cpu().numpy()))
            for test_k in ev_range:
                testkey = "test_at_%d" % test_k
                if testkey not in ret:
                    ret[testkey] = []
                ret[testkey].append(ranking_metrics_at_k(model, tr, te, K=test_k))
            savedir = os.path.join("trace", args.dataset_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            torch.save(ret, os.path.join(savedir, model_name))
