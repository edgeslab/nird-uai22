import networkx as nx
import pdb
import numpy as np


def compute_rw_matrix(adj_mat, alpha, p):
    ## Computes the random walk kernel matrix
    ## Details in:
    ## Kernels and Regularization on Graphs
    ## Smola and Kondor
    # this is D^{-1/2}
    D = np.diagflat(1 / np.sqrt(adj_mat.sum(0)))
    I = np.eye(D.shape[0])
    L_sym = I - (D @ adj_mat @ D)
    RW = np.linalg.matrix_power(alpha * I - L_sym, p)
    return RW

def gen_correlated(num_observations, adj_mat, alpha=0.5, p=2):
    ## Generated correlated samples by taking samples off of 
    ## a multivariate normal with covariance defined by a random
    ## walk kernel
    RW = compute_rw_matrix(adj_mat, alpha, p)
    
    min_eig = np.min(np.real(np.linalg.eigvals(RW)))
    if min_eig < 0:
        RW -= 10*min_eig * np.eye(*RW.shape)

    return np.random.multivariate_normal(
      np.zeros(adj_mat.shape[0]), RW, num_observations
    )