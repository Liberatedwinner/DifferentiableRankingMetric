
import torch
import torch.nn as nn
from spotlight.layers import ScaledEmbedding, ZeroEmbedding


class neuMF(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, sparse=False):

        super(neuMF, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embed = ScaledEmbedding(num_users, embedding_dim * 2, sparse=sparse)
        self.item_embed = ScaledEmbedding(num_items, embedding_dim * 2, sparse=sparse)
        self.user_embed.weight.data.normal_(0.0, 0.01)
        self.item_embed.weight.data.normal_(0.0, 0.01)

        self.l1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.l2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.last = nn.Linear(embedding_dim // 2 + embedding_dim, 1)

    def forward(self, user_ids, item_ids):
        ue = self.user_embed(user_ids)
        ie = self.item_embed(item_ids)
        gmf_u, mlp_u = ue[:, :self.embedding_dim], ue[:, self.embedding_dim:]
        gmf_i, mlp_i = ie[:, :self.embedding_dim], ie[:, self.embedding_dim:]

        left = (gmf_u * gmf_i)

        right = torch.cat([mlp_u, mlp_i], dim=-1)
        right = torch.relu(self.l1(right))
        right = torch.relu(self.l2(right))

        o = self.last(torch.cat([left, right], -1)).squeeze(-1)
        return o
