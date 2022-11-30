import argparse
import time
import random
from matplotlib.font_manager import weight_dict
import torch
import numpy as np
from data_loader import data_loader
from utils import normalization, renormalization, rounding, MAE, RMSE, sample_batch_index, dist2sim
from sklearn.neighbors import NearestNeighbors
from model import Imputation
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Reliable Iterative Imputation Network for Missing Data')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--k_candidate', help='how many candidates to save', type=int, default=6)
    parser.add_argument('--k_neighbors', help='how many neighbors to save', type=int, default=4)
    parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=32, type=int)
    parser.add_argument('--miss_rate', help='missing data probability', default=0.2, type=float)
    parser.add_argument('--data_name', choices=['wine', 'letter','spam', 'heart', 'breast', 'phishing', 'wireless', 'turkiye', 'credit', 'connect', 'car', 'chess', 'news', 'shuttle'], default='wine', type=str)
    parser.add_argument('--seed', type=int, default=17, help='Random seed.')
    parser.add_argument('--max_iter', type=int, default=3, help='The maximum number of iterations.')
    
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0100, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=2e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--metric', choices=['cityblock', 'cosine','euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean'], default='euclidean', type=str)

    return parser.parse_args()

def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ori_data_x, data_x, data_m = data_loader(args.data_name, args.miss_rate)
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    nos, dims = norm_data_x.shape
    imputed_X = norm_data_x.copy()
    neigh = NearestNeighbors(n_neighbors=args.k_candidate, metric=args.metric)
    

    imputators = [Imputation(input_dim=dims-1, output_dim=int((dims-1)), k_neighbors=args.k_neighbors).to(device)
                        for _ in range(dims)]
    optimizers = [torch.optim.Adam(imputators[i].parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        for i in range(dims)]
    start = time.time()
    for iteration in range(args.max_iter):
        print("In the", iteration+1, "'s iteration")
        for dim in tqdm(range(dims)):
            X_1 = np.delete(imputed_X, dim, 1)
            data_m_dim = np.delete(data_m, dim, 1)

            i_not_nan_index = data_m[:, dim].astype(bool)
            X_train = X_1[i_not_nan_index]
            y_train = imputed_X[i_not_nan_index, dim]
            X_test = X_1[~i_not_nan_index]
            neigh.fit(X_train)
            data_m_train = data_m_dim[i_not_nan_index]
            data_m_test = data_m_dim[~i_not_nan_index]
            if X_test.shape[0] == 0:
                continue
            for epoch in (range(args.epochs)):
                imputators[dim].train()
                optimizers[dim].zero_grad()
                batch_size = args.batch_size
                batch_idx = sample_batch_index(X_train.shape[0], batch_size)
                X_batch = X_train[batch_idx,:]
                Y_batch = y_train[batch_idx]
                data_m_batch = data_m_train[batch_idx, :]

                neigh_dist, neigh_ind = neigh.kneighbors(X_batch, args.k_candidate, return_distance=True)
                neigh_dist = neigh_dist[:,1:]
                neigh_ind = neigh_ind[:,1:]

                y_pred, a_, b_ = imputators[dim].forward(torch.Tensor(X_batch).to(device), 
                                                 torch.Tensor(X_train).to(device), 
                                                 torch.Tensor(neigh_ind).type(torch.LongTensor).to(device),
                                                 torch.Tensor(y_train).to(device),
                                                 torch.Tensor(neigh_dist).to(device),
                                                 torch.Tensor(data_m_train).to(device),
                                                 torch.Tensor(data_m_batch).to(device),
                                                 test = False)
                loss = imputators[dim].loss(torch.Tensor(Y_batch).to(device), torch.Tensor(y_pred).to(device))
                loss.backward()
                optimizers[dim].step()

            imputators[dim].eval()

            
            # Retrieve
            neigh_dist, neigh_ind = neigh.kneighbors(X_test, args.k_candidate-1, return_distance=True)
            
            y_pred, a_, b_ = imputators[dim].forward(torch.Tensor(X_test).to(device), 
                                                 torch.Tensor(X_train).to(device), 
                                                 torch.Tensor(neigh_ind).type(torch.LongTensor).to(device),
                                                 torch.Tensor(y_train).to(device),
                                                 torch.Tensor(neigh_dist).to(device),
                                                 torch.Tensor(data_m_train).to(device),
                                                 torch.Tensor(data_m_test).to(device),
                                                 test = True)
            y_pred = y_pred.cpu().detach().numpy().reshape((-1))
            imputed_X[~i_not_nan_index, dim] = y_pred
    end = time.time()
    print("\n time used: ", end-start)
    imputed_data = renormalization(imputed_X, norm_parameters)  
    imputed_data = rounding(imputed_data, data_x) 
    print('\n##### RMSE Performance: ' + str(np.round(RMSE(imputed_data, ori_data_x, data_m), 4)))
    print('\n##### MAE Performance: ' + str(np.round(MAE(imputed_data, ori_data_x, data_m), 4)))



    

if __name__ == "__main__":
    args = parse_args()
    main(args)
  
