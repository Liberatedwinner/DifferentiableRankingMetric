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
    impostors = torch.log(1.0 + torch.clamp(-pos.unsqueeze(-1) + neg + margin, 0).mean(-1) * n_items).detach()

    loss_per_pair = torch.clamp(-pos + neg_highest + margin, 0)  # batch_size
    return (impostors * loss_per_pair).sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ml-1m-l-1-100')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--infer_dot', type=int, default=0)
    args = parser.parse_args()
    infer_dot = args.infer_dot == 1
    model_name = "CML"
    if infer_dot is True:
        model_name = "torch-warp"
    torch.cuda.set_device(args.device_id)

    with open("data/parsed/%s" % args.dataset_name, 'rb') as f:
        (tr, val, te) = pickle.load(f)
    tr = (tr + val).tocsr()
    bp = best_paramset = torch.load(os.path.join("saved_models",args.dataset_name, model_name), map_location='cuda:%d' % args.device_id)

    n_users, n_items = tr.shape
    n_epochs = 300
    batch_size = 8192
    ev_range = [5, 10, 20, 30, 50]
    for num_run in range(1):
        dataset = misc.loader.RecDataset(tr, K=21)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        dim, reg, lr = bp['dim'], bp['reg'], bp['lr']
        print(dim, reg, lr)
        model = models.mf.mfrec(n_users, n_items, dim, infer_dot=infer_dot).cuda()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

        for __ in range(bp['epoch']):
            uu = []
            ii = []
            _i = 0
            print(__, bp['epoch'])
            for u, i, j in loader:
                _i += 1
                u = u.cuda()
                i = i.cuda()
                j = j.cuda()
                if infer_dot is True:
                    y = model.dot_fow(u, i, j)
                else:
                    y = model.forward(u, i, j)
                pos = y[:, 0]
                neg = y[:, 1:]
                loss = warp_loss(pos, neg, model.n_items).sum()
                model.zero_grad()
                if infer_dot:
                    reg_term = 0
                else:
                    reg_term = reg * model.cov(u, torch.cat([i, j.flatten(), -1]))
                (loss + reg_term).backward()
                optimizer.step()
                model.normalize(u, _target='uid')
                model.normalize(i, _target='iid')
                model.normalize(j, _target='iid')
        best_paramset['model'] = model
        for test_k in ev_range:
            testkey = "test_at_%d" % test_k
            if testkey not in best_paramset:
                best_paramset[testkey] = []
            best_paramset[testkey].append(ranking_metrics_at_k(model, tr, te, K=test_k))

        savedir = os.path.join("best_res", args.dataset_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        torch.save({'model': model,'bp': best_paramset}, os.path.join(savedir, model_name))
