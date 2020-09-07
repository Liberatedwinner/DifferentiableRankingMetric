import numpy as np
import torch
from torch import nn
import itertools
from .neuralsort import NeuralSort
from .loss import neuPrecLoss




def norm(x):
    div = torch.clamp((x * x).sum(-1), min=1)
    return (x.T / div).T

def loss_mse(u, i, j, reduction=True):
    ui = (u * i).sum(-1)
    uj = (u.unsqueeze(1) * j).sum(-1)
    ret = ((ui - 1) **2 + (uj ** 2).sum(-1))
    if reduction is True:
        ret = ret / (j.shape[1] + 1)
    return ret.mean()


def loss_ce(u, i, j, reduction=True):
    ui = (u * i).sum(-1)
    uj = (u.unsqueeze(1) * j).sum(-1)
    ui = torch.sigmoid(ui)
    uj = torch.sigmoid(uj)

    ret = -(torch.log(ui) + torch.log(1 - uj).sum(-1))
    if reduction is True:
        ret = ret / (j.shape[1] + 1)
    return ret.mean()



def loss_bpr(u, i, j):
    ui = (u * i).sum(-1)
    uj = (u.unsqueeze(1) * j).sum(-1)
    return -torch.log(torch.sigmoid((ui.unsqueeze(-1) - uj)).mean(-1)).mean()


def loss_warp(u, i, j, n_items, margin=1.0):
    ui = (u * i).sum(-1) # batch_size
    uj = (u.unsqueeze(1) * j).sum(-1) #batch_size x num_neg_samples
    uc, _ = uj.max(-1)
    impostors = torch.log(1.0 + torch.clamp(-ui.unsqueeze(-1) + uj + margin, 0).mean(-1) * n_items)
    loss_per_pair = torch.clamp(-ui + uc + margin, 0) #batch_size
    return (impostors * loss_per_pair).sum()

def loss_cml(u, i, j, n_items, margin=1.0):
    ui = ((u - i) ** 2).sum(-1)
    uj = ((u.unsqueeze(1) - j) ** 2).sum(-1)

    #DIST Loss:
    uc, _ = uj.min(-1)
    loss_per_pair = torch.clamp(ui - uc + margin, 0, 10)
    rank_approx = ((ui.unsqueeze(-1) - uj + margin) > 0).float().sum(1)
    w = torch.log(1.0 + rank_approx * n_items)
    rank_loss = (w * loss_per_pair)
    return rank_loss.mean()

def loss_reg(u, i, j):
    return (u ** 2).sum() + (i ** 2).sum() + (j ** 2).sum()


class mfrec(nn.Module):

    def __init__(self,
                 n_users,
                 n_items,
                 emb_size=32,
                 infer_dot=False,
                 rec_batch_size=64):
        super(mfrec, self).__init__()

        self.save = None
        self.savekey = -1
        self.rec_batch_size = rec_batch_size
        self.emb_size = emb_size
        self.n_users, self.n_items = n_users, n_items
        sq = 1 / np.sqrt(self.emb_size)
        self.u_emb = nn.Embedding(self.n_users, self.emb_size)
        self.i_emb = nn.Embedding(self.n_items, self.emb_size)
        nn.init.uniform_(self.u_emb.weight, -sq, sq)
        nn.init.uniform_(self.i_emb.weight, -sq, sq)
        self.l2 = not infer_dot

    def predict(self, userid, itemids):
        u = self.u_emb(torch.LongTensor([userid]).cuda())
        i = self.i_emb(torch.LongTensor(itemids).cuda())
        return -((u - i) ** 2).sum(-1).detach().cpu().numpy()

    def rank_items(self, userid, tr, itemids):
        scores = self.predict(userid, itemids)
        order = (-scores).argsort()
        s = scores[order].tolist()
        i = itemids[order].tolist()
        return list(zip(i, s))

    def forward(self, u, i, j):
        """
            u = batch_size
            i = batch_Size
            j = batch_size x neg_samples
        """
        uemb = self.u_emb(u)
        iemb = self.i_emb(i)
        jemb = self.i_emb(j)
        pos = ((uemb - iemb) ** 2).sum(-1)
        neg = ((uemb.unsqueeze(1) - jemb) ** 2).sum(-1) # (batch size x neg_samples)
        return -torch.cat([pos.unsqueeze(-1), neg], -1)
        
    def dot_fow(self, u, i, j):
        """
            u = batch_size
            i = batch_Size
            j = batch_size x neg_samples
        """
        uemb = self.u_emb(u)
        iemb = self.i_emb(i)
        jemb = self.i_emb(j)
        pos = (uemb * iemb).sum(-1)
        neg = (uemb.unsqueeze(1) * jemb).sum(-1) # (batch size x neg_samples)
        return torch.cat([pos.unsqueeze(-1), neg], -1)
 
    def fow2(self, u, i):
        uemb = self.u_emb(u)
        iemb = self.i_emb(i)
        if self.l2 is True:
            v = -((uemb.unsqueeze(1) - iemb) ** 2).sum(-1) # (batch size x neg_samples)
        else:
            v = (uemb.unsqueeze(1) * iemb).sum(-1) # (batch size x neg_samples)
        return v

    def recommend(self, userid, user_items, N=10):
        rec_batch_size = self.rec_batch_size
        start = (userid // rec_batch_size) * rec_batch_size
        end = min(start + rec_batch_size, self.n_users)
        savekey = userid // rec_batch_size
        if self.savekey != savekey:
            user = torch.arange(start, end, dtype=torch.int64).to(self.i_emb.weight.device)
            u_emb = self.u_emb(user.unsqueeze(-1)).to(self.i_emb.weight.device)
            if self.l2 is False:
                v = (u_emb * self.i_emb.weight.unsqueeze(0)).sum(-1).detach().cpu().numpy()
            else:
                v = -((u_emb - self.i_emb.weight.unsqueeze(0)) ** 2).sum(-1).detach().cpu().numpy()
            del u_emb
            self.save = v
            self.savekey = savekey
        liked = set()
        liked.update(user_items[userid].indices)
        scores = self.save[userid % rec_batch_size].ravel()
        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    def cov(self, u, i):
        mat = torch.cat([self.u_emb.weight[u], self.i_emb.weight[i]], 0)
        cov = mat - mat.mean(0)
        cov_mat = torch.matmul(cov.T, cov) / (mat.shape[0])
        cov_mat.fill_diagonal_(0)
        return cov_mat

    def normalize(self, ids, _target='uid'):
        ids = ids.flatten()
        if _target == 'uid':
            target = self.u_emb
        elif _target == 'iid':
            target = self.i_emb
        with torch.no_grad():
            target.weight[ids] = norm(target.weight[ids])
