import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools

dg = torch.digamma
dgd = lambda x: torch.polygamma(1, x)
dgdsqrt = lambda x: torch.sqrt(dgd(x))

def g_eps(a, e):
    return torch.exp(e * dgdsqrt(a) + dg(a)).clamp(0, 10000)

def r_beta(alpha, beta):
    ns1 = torch.normal(mean=0, std=1, size=alpha.shape).clamp(-3, 3).cuda()
    ns2 = torch.normal(mean=0, std=1, size=alpha.shape).clamp(-3, 3).cuda()
    a_samples = g_eps(torch.exp(0.5 * alpha), ns1)
    b_samples = g_eps(torch.exp(0.5 * beta), ns2)
    print(a_samples.min(), b_samples.min())
    return a_samples / (a_samples + b_samples+ 1e-10)

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


class MGR(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MGR, self).__init__()
        self.p_dims = p_dims
        self.n_items = p_dims[-1]
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.p_dims[-1] = self.n_items * 2
        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.dd = nn.Dropout(0.1)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        alpha, beta = self.decode(z)
        y = r_beta(alpha, beta)
        return y, mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        if self.training:
            h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = self.dd(h)
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            eps = eps.clamp(-3, 3)
            return mu + eps * std
        else:
            return mu
        #return mu
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = self.dd(h)
                h = torch.tanh(h)
        alpha, beta = h[:, :self.n_items], h[:, self.n_items:]

        return alpha, beta

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            #layer.weight.data.normal_(0.0, std)
            torch.nn.init.xavier_uniform_(layer.weight)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            torch.clamp(layer.bias.data, min=-0.001, max=0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            #std = np.sqrt(2.0 / (fan_in + fan_out))
            #layer.weight.data.normal_(0.0, std)
            torch.nn.init.xavier_uniform_(layer.weight)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)



class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.dd = nn.Dropout(0.1)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        if self.training:
            h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = self.dd(h)
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            eps = eps.clamp(-3, 3)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = self.dd(h)
                h = torch.tanh(h)
        return h

    def rank_items(self, userid, tr, itemids):
        self.eval()
        if isinstance(itemids, list):
            itemids = np.array(itemids)
        x = torch.from_numpy(np.asarray(tr[userid].todense()).astype(np.float32)).cuda()
        h = F.normalize(x)
        h, logvar = self.encode(h)
        for i, layer in enumerate(self.p_layers):
            if i < len(self.p_layers) - 1:
                h = layer(h)
                h = torch.tanh(h)
            else:
                w = torch.matmul(layer.weight[itemids], h.T).squeeze(-1) + layer.bias[itemids]
        w = w.detach().cpu().numpy()
        idx = (-w).argsort()
        return list(zip(itemids[idx], w[idx]))

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            #layer.weight.data.normal_(0.0, std)
            torch.nn.init.xavier_uniform_(layer.weight)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            torch.clamp(layer.bias.data, min=-0.001, max=0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            #std = np.sqrt(2.0 / (fan_in + fan_out))
            #layer.weight.data.normal_(0.0, std)
            torch.nn.init.xavier_uniform_(layer.weight)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiDAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, n_users=-1, dropout=0.5):
        super(MultiDAE, self).__init__()
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

        # Last dimension of q- network is for mean and variance
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




class CMAE(nn.Module):

    def __init__(self, p_dims, q_dims=None, n_users=-1, dropout=0.5):
        super(CMAE, self).__init__()
        self.p_dims = p_dims
        self.n_users = n_users
        self.n_items = p_dims[-1]
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

        self.item_emb = nn.Embedding(self.n_items, p_dims[-2])
        # Last dimension of q- network is for mean and variance
        self.dims = self.q_dims + self.p_dims[1:]
        self.dims = self.dims[:-1]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dims[:-1], self.dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.dd = nn.Dropout(0.2)

        self.init_weights()

    def forward(self, input, user=None, use_dropout=False):
        h = F.normalize(input)
        h = self.drop(h)

        for i, l in enumerate(self.layers):
            h = l(h)
            #if (i == 0) and (self.use_user) and user is not None:
            #    h = h + self.V_u(user)
            h = self.dd(h)
            h = torch.tanh(h)

        # h <- user latent embedding #batch size x 1 x emb_dim
        # w = self.item_emb.weight = #1 x n_items x emb_dim
        #torch.matmul(h, self.item_emb.weight.T) # batch size x n_items
        dist = ((h.unsqueeze(1) - self.item_emb.weight.unsqueeze(0)) ** 2).sum(-1)
        return dist


    def rank_items(self, userid, tr, itemids):
        self.eval()
        if isinstance(itemids, list):
            itemids = np.array(itemids)
        x = torch.from_numpy(np.asarray(tr[userid].todense()).astype(np.float32)).cuda()
        h = F.normalize(x)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            #if (i == 0) and (self.use_user):
            #    h = h + self.V_u.weight[userid]
            h = torch.tanh(h)

        dist = ((h.unsqueeze(1) - self.item_emb.weight[itemids].unsqueeze(0)) ** 2).sum(-1)
        dist = dist.detach().cpu().numpy()
        idx = dist.argsort()
        return list(zip(itemids[idx], -dist[idx]))

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.normal_(0.0, 0.001)
            torch.clamp(layer.bias.data, min=-0.001, max=0.001)
        if self.use_user:
            self.V_u.weight.data.normal_(0, 0.001)

        torch.nn.init.xavier_uniform_(self.item_emb.weight)



def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = 0.5 * (torch.sum(-logvar + torch.exp(logvar) + mu ** 2 - 1, dim=1))
    return BCE + anneal * KLD
