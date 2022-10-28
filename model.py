import torch.nn as nn
import torch.nn.functional as F
import torch
from attention import AttentionLayer
from torch.autograd import Variable
from utils import dist2sim
from torch.nn import Conv1d

def interaction(X, Neighbors):
    return X-Neighbors

class Imputation(nn.Module):
    def __init__(self, input_dim, output_dim, k_neighbors):
        super(Imputation, self).__init__()
        self.k_neighbors = k_neighbors
        self.attentions = AttentionLayer(input_dim, output_dim, k_neighbors)
        self.w_1 = nn.Linear(input_dim*4+1, output_dim*2)
        self.w_2 = nn.Linear(output_dim*2, output_dim*1)
        self.w_3 = nn.Linear(output_dim*1, 1)
        self.relu = nn.ReLU()
        self.dropout = 0.6
        self.conv1d = Conv1d(
                    in_channels=1, 
                    out_channels=1,
                    kernel_size=3,
                    padding=1,
                )
        self.conv1d_nei = Conv1d(
                    in_channels=k_neighbors, 
                    out_channels=k_neighbors,
                    kernel_size=5,
                    padding=2,
                )

    def forward(self, X, X_train, neigh_ind, y_train, neigh_dist, data_m_train, data_m_batch, test=False):
        
        neigh_dist = Variable(neigh_dist, requires_grad=True)
        y_train = Variable(y_train, requires_grad=True)
        Candidate = X_train[neigh_ind]
        Candidate, neigh_dist, neigh_ind, a, b = self.attentions(X, Candidate, neigh_dist, neigh_ind, data_m_train, data_m_batch, test)
        
        weights = torch.Tensor(dist2sim(neigh_dist)).to('cuda:0')
        
        pred = torch.sum(y_train[neigh_ind] * weights, axis=1)
    
        
        weights = weights.reshape((weights.shape[0], weights.shape[1],1))
        weights = weights.repeat(1,1,X.shape[1])
        Neighbors_o = torch.mul(Candidate, weights)
        Neighbors = self.conv1d_nei(Neighbors_o)
        Neighbors, index = torch.max(Neighbors, axis=1)
        no, dim = X.shape
        X1 = self.conv1d(X.reshape((no, 1, dim)))
        X1 = X1.reshape((no, dim))
        X2 = X.repeat(1,1,self.k_neighbors).reshape([no,self.k_neighbors,dim])

        Inter = torch.cat([X2, Neighbors_o], axis = 2)
        Inter = self.conv1d_nei(Inter)
        Inter, index = torch.max(Inter, axis=1)


        inputs = torch.cat([pred.reshape((-1,1)), (X1), (Inter), (Neighbors)], axis = 1)
        x1 = self.w_1(inputs)
        x1 = F.dropout(x1, 0.60, training=self.training)
        x2 = self.w_2(x1)
        x2 = self.relu(x2)
        x2 = F.dropout(x2, 0.60, training=self.training)
        x3 = self.w_3(x2).reshape((-1))
        x3 = self.relu(x3)
        
        return pred+x3, a, b
    
    def loss(self, y, y_pred):
        return torch.mean((y - y_pred)**2)

