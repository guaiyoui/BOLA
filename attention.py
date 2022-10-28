from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Simple Attention layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, k_neighbors):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(2*in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        self.k_neighbors = k_neighbors

    def forward(self, X, Candidate, neigh_dist, neigh_ind, data_m_train, data_m_batch, test):
        no, dim = X.shape
        data_m_candidate = data_m_train[neigh_ind]
        X = torch.cat([X, data_m_batch], axis = 1)
        Candidate_m = torch.cat([Candidate, data_m_candidate], axis = 2)
        WX = torch.matmul(X, self.W)
        WC = torch.matmul(Candidate_m, self.W)


        e = self._prepare_attentional_mechanism_input(WX, WC)
        e = e.reshape((e.shape[0], e.shape[1]))
        _,index = e.topk(k=self.k_neighbors, dim=1)
        indicator = torch.zeros((e.shape[0], e.shape[1]))
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                indicator[i, index[i,j]] = 1
        indicator1 = 1-indicator
        attention = indicator.bool()
        attention1 = indicator1.bool()
        
        Candidate = Candidate[attention].reshape((no, self.k_neighbors, dim))
        data_m_candidate0 = data_m_candidate[attention].reshape((no, self.k_neighbors, dim))
        data_m_candidate1 = data_m_candidate[attention1].reshape((no, dim))
        
        a = torch.sum(data_m_candidate0)/(no*self.k_neighbors)
        b = torch.sum(data_m_candidate1)/no
        neigh_dist = neigh_dist[attention].reshape((no, self.k_neighbors))

        neigh_ind = neigh_ind[attention].reshape((no, self.k_neighbors))

        return Candidate, neigh_dist, neigh_ind, a, b
        

    def _prepare_attentional_mechanism_input(self, WX, WC):
        WX = WX.repeat(1, WC.shape[1]).reshape((WX.shape[0], WC.shape[1], WX.shape[1]))
        input = torch.cat([WX, WC], axis = 2)
        e = torch.matmul(input, self.a[:2*self.out_features, :])
        return e
