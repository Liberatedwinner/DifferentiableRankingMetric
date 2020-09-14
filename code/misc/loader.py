from collections import Counter
import itertools
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from .util import sparse2torch_sparse
from .util import naive_sparse2tensor
import fastrand

class AEDataset(Dataset):
    def __init__(self, tr_csr, max_seq_len=10, K=5):
        super(AEDataset, self).__init__()
        self.n_users, self.n_items = tr_csr.shape
        self.tr_csr = tr_csr
        self.user_seen_cnts = np.ediff1d(self.tr_csr.indptr)
    def __getitem__(self, i):
        return i, naive_sparse2tensor(self.tr_csr[i]).squeeze(0)


    def __len__(self,):
        return self.n_users



class AENeuDataset(Dataset):
    def __init__(self, tr_csr, n_pos=3, n_neg=63):
        super(AENeuDataset, self).__init__()
        self.n_users, self.n_items = tr_csr.shape
        self.tr_csr = tr_csr
        self.n_pos, self.n_neg = n_pos, n_neg

        self.user_seen_items = [
            set(self.tr_csr.indices[self.tr_csr.indptr[i]:self.tr_csr.indptr[i + 1]].tolist()) for i in range(self.n_users)]
        self.user_seen_cnts = np.ediff1d(self.tr_csr.indptr)

    def __getitem__(self, i):
        user_seen = self.user_seen_cnts[i]
        while user_seen == 0:
            i = fastrand.pcg32bounded(self.n_users)
            user_seen = self.user_seen_cnts[i]

        uid = i
        row = naive_sparse2tensor(self.tr_csr[i]).squeeze(0)
        s_pos = self.user_seen_items[i]

        neg = []
        pos = []
        while len(neg) < self.n_neg:
            _i = fastrand.pcg32bounded(self.n_items)
            if _i in s_pos:
                pos.append(_i)
            else:
                neg.append(_i)
        upos = self.tr_csr.indices[self.tr_csr.indptr[i]:self.tr_csr.indptr[i + 1]]
        while len(pos) < self.n_pos:
            _i = upos[fastrand.pcg32bounded(user_seen)]
            pos.append(_i)
        return np.asarray(pos[:self.n_pos], dtype=np.int64), np.asarray(neg[:self.n_neg], dtype=np.int64), i, naive_sparse2tensor(self.tr_csr[i]).squeeze(0)


    def __len__(self,):
        return self.n_users


class RecDataset(Dataset):
    def __init__(self, tr_csr, K=10, filter_neg=True):
        super(RecDataset, self).__init__()
        self.n_users, self.n_items = tr_csr.shape
        coo = tr_csr.tocoo()
        self.row = coo.row.astype(np.int64)
        self.col = coo.col.astype(np.int64)
        self.nnz = tr_csr.nnz
        self.indices = tr_csr.indices.astype(np.int64)
        self.indptr = tr_csr.indptr.astype(np.int64)
        self.data = tr_csr.data
        self.user_seen_cnts = np.ediff1d(self.indptr)
        self.K = K
        self.K_1 = K - 1
        self.filter_neg = filter_neg
        self.user_seen_items = [
            set(self.indices[self.indptr[i]:self.indptr[i + 1]].tolist()) for i in range(self.n_users)]
        self.ones = np.ones(self.n_items, dtype=np.int64)
        self.zeros = np.zeros(self.n_items, dtype=np.int64)

    def __getitem__(self, idx):
        i = self.row[idx]
        pos = self.col[idx]
        # CHECK if this is not good;
        user_seen = self.user_seen_cnts[i]
        while user_seen == 0:
            i = self.row[fastrand.pcg32bounded(self.nnz)]
            user_seen = self.user_seen_cnts[i]
        s_pos = self.user_seen_items[i]
        neg = []#np.random.choice(self.n_items, size=(2 * self.K_1), replace=False).tolist()
        #neg = np.random.choice(self.n_items, size=(2 * self.K_1), replace=False).tolist()
        if self.filter_neg:
            neg = [x for x in neg if x not in s_pos]
            while len(neg) < self.K_1:
                _i = fastrand.pcg32bounded(self.n_items)
                if (_i not in neg) and (_i not in s_pos):
                    neg.append(_i)
        else:
            neg = np.random.choice(self.n_items, size=(2 * self.K_1), replace=False).tolist()


        return i, pos, np.asarray(neg[:self.K_1])

    def __len__(self,):
        return self.nnz



class MyDataset(Dataset):
    def __init__(self, tr_csr, n_pos=2, n_neg=18, filter_neg=True):
        super(MyDataset, self).__init__()
        self.n_users, self.n_items = tr_csr.shape
        coo = tr_csr.tocoo()
        self.row = coo.row.astype(np.int64)
        self.col = coo.col.astype(np.int64)
        self.nnz = tr_csr.nnz
        self.indices = tr_csr.indices.astype(np.int64)
        self.indptr = tr_csr.indptr.astype(np.int64)
        self.data = tr_csr.data
        self.user_seen_cnts = np.ediff1d(self.indptr)
        self.K_1 = n_neg
        self.n_pos = n_pos
        self.filter_neg = filter_neg
        self.user_seen_items = [
            set(self.indices[self.indptr[i]:self.indptr[i + 1]].tolist()) for i in range(self.n_users)]
        self.ones = np.ones(self.n_items, dtype=np.int64)
        self.zeros = np.zeros(self.n_items, dtype=np.int64)

    def __getitem__(self, idx):
        i = self.row[idx]
        pos = self.col[idx]
        # CHECK if this is not good;
        user_seen = self.user_seen_cnts[i]
        while user_seen == 0:
            i = self.row[np.random.randint(self.nnz)]
            user_seen = self.user_seen_cnts[i]

        s_pos = self.user_seen_items[i]
        #neg = np.random.choice(self.n_items, size=self.K_1, replace=False).tolist()
        neg = []# np.random.choice(self.n_items, size=self.K_1, replace=False).tolist()
        pos = []
        while len(neg) < self.K_1:
            _i = fastrand.pcg32bounded(self.n_items)
            if _i in s_pos:
                pos.append(_i)
            else:
                neg.append(_i)
        upos = self.indices[self.indptr[i]:self.indptr[i + 1]]
        while len(pos) < self.n_pos:
            _i = upos[fastrand.pcg32bounded(user_seen)]
            pos.append(_i)
        return i, np.asarray(pos[:self.n_pos]), np.asarray(neg[:self.K_1])

    def __len__(self,):
        return self.nnz