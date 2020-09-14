'''
"Differentiable k-nearest neighbors" layer.

Given a set of M queries and a set of N neighbors,
returns an M x N matrix whose rows sum to k, indicating to what degree
a certain neighbor is one of the k nearest neighbors to the query.
At the limit of tau = 0, each entry is a binary value representing
whether each neighbor is actually one of the k closest to each query.
'''
import numpy as np
from scipy.sparse import *
import numpy.random
import torch
from .pl import PL
from .neuralsort import NeuralSort
from torch.utils.data import Dataset



class DKNN (torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False, method='deterministic', num_samples=-1):
        super(DKNN, self).__init__()
        self.k = k
        self.soft_sort = NeuralSort(tau=tau, hard=hard)
        self.method = method
        self.num_samples = num_samples

    # query: M x p
    # neighbors: N x p
    #
    # returns:
    def forward(self, query, neighbors, tau=1.0):
        diffs = (query.unsqueeze(1) - neighbors.unsqueeze(0))
        squared_diffs = diffs ** 2
        l2_norms = squared_diffs.sum(2)
        norms = l2_norms
        scores = -norms
        print(scores.shape)
        if self.method == 'deterministic':
            P_hat = self.soft_sort(scores)
            top_k = P_hat[:, :self.k, :].sum(1)
            return top_k
        if self.method == 'stochastic':
            pl_s = PL(scores, tau, hard=False)
            P_hat = pl_s.sample((self.num_samples,))
            top_k = P_hat[:, :, :self.k, :].sum(2)
            return top_k

class SC(torch.nn.Module):

    def __init__(self, hard=False, method='deterministic', num_samples=-1):
        super(SC, self).__init__()
        self.soft_sort = NeuralSort(hard=hard)
        self.method = method
        self.num_samples = num_samples

    # query: M x p
    # neighbors: N x p
    #
    # returns:
    def forward(self, scores, k=3, tau=1.0):
        if self.method == 'deterministic':
            P_hat = self.soft_sort(scores, tau=tau)
            top_k = P_hat[:, :k, :]
            return top_k
        if self.method == 'stochastic':
            pl_s = PL(scores.unsqueeze(-1), tau, hard=False)
            P_hat = pl_s.sample((self.num_samples,))
            top_k = P_hat[:, :, :k, :]
            return top_k


class RecDataset(Dataset):
    def __init__(self, tr_csr, seq_len=100, K=5):
        super(RecDataset, self).__init__()
        self.n_users, self.n_items = tr_csr.shape
        self.indices = tr_csr.indices.astype(np.int64)
        self.indptr = tr_csr.indptr.astype(np.int64)
        self.data = tr_csr.data
        self.user_seen_cnts = np.ediff1d(self.indptr)
        self.seq_len = seq_len
        self.K = K

    def __getitem__(self, i):
        user_seen = self.user_seen_cnts[i]

        while user_seen < self.K:
            i = np.random.randint(0, self.n_users)
            user_seen = self.user_seen_cnts[i]

        pos = self.indices[self.indptr[i]:self.indptr[i+1]].copy()
        if user_seen > self.K:
            pos = pos[np.random.choice(user_seen, self.K, replace=False)]
            user_seen = self.seq_len

        neg = np.random.choice(user_seen, self.seq_len + 3 * self.K, replace=True).tolist()
        neg = [x for x in neg if x not in set(pos)]
        while len(neg) < self.seq_len - self.K:
            neg = np.random.choice(user_seen, self.seq_len + 3 * self.K, replace=True).tolist()
            neg = [x for x in neg if x not in set(pos)]

        return i, pos, np.asarray(neg)[:self.seq_len - self.K]

    def __len__(self,):
        return self.n_users