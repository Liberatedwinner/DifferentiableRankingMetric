import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools

class implicitWrapper():
    def __init__(self, AEModel, tensor_wrapper, vae=False):
        self.model = AEModel
        self.vae = vae
        self.tensor_wrapper = tensor_wrapper
        self.save = None
        self.savekey = -1
    def recommend(self, userid, user_items, N=10):
        rec_batch_size = 64

        savekey = userid // rec_batch_size
        start = savekey * rec_batch_size
        end = min(start + rec_batch_size, user_items.shape[0])
        # calculate the top N items, removing the users own liked items from the results
        if self.savekey != savekey:
            i = self.tensor_wrapper(user_items[start:end]).cuda()
            if self.vae:
                _ = self.model(i)
                try:
                    scores, _, _ = _
                except:
                    scores = _
                self.save = scores.cpu().detach().numpy()
            else:
                self.save = self.model(i, use_dropout=False).cpu().detach().numpy()
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


class CDAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, n_users=-1, dropout=0.5):
        super(CDAE, self).__init__()
        self.p_dims = p_dims
        self.n_users = n_users
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        if n_users > -1:
            self.use_user = True
            self.V_u = nn.Embedding(self.n_users, p_dims[-2])
        else:
            self.use_user = False

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dims[:-1], self.dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.dd = nn.Dropout(0.2)
        self.init_weights()

    def forward(self, input, user=None, use_dropout=False):
        h = F.normalize(input)
        h = self.drop(h)

        for i, l in enumerate(self.layers):
            h = l(h)
            if (i == 0) and (self.use_user) and user is not None:
                h = h + self.V_u(user)
            if i != len(self.layers) - 1:
                h = self.dd(h)
                h = torch.tanh(h)
        return h

    def rank_items(self, userid, tr, itemids):
        self.eval()
        if isinstance(itemids, list):
            itemids = np.array(itemids)
        x = torch.from_numpy(np.asarray(tr[userid].todense()).astype(np.float32)).cuda()
        h = F.normalize(x)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                h = layer(h)
                if (i == 0) and (self.use_user):
                    h = h + self.V_u.weight[userid]
                h = torch.tanh(h)
            else:
                w = torch.matmul(layer.weight[itemids], h.T).squeeze(-1) + layer.bias[itemids]
        w = w.detach().cpu().numpy()
        idx = (-w).argsort()
        return list(zip(itemids[idx], w[idx]))

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.normal_(0.0, 0.001)
            torch.clamp(layer.bias.data, min=-0.001, max=0.001)
        if self.use_user:
            self.V_u.weight.data.normal_(0, 0.001)
