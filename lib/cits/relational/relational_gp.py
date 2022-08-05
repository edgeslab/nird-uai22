import pdb
import math

import numpy as np
import networkx as nx
import torch
import gpytorch
import ghalton 

from sklearn.metrics.pairwise import euclidean_distances

from .relational_rff import RelationalRFFKernel


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, 
        train_x, 
        train_y,
        train_graph,
        likelihood,
        is_rel=False,
        num_samples=20,
        num_outputs=40
    ):
        train_x_indices = torch.arange(train_x.shape[-2]).reshape(-1, 1)
        super().__init__(train_x_indices, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_outputs]))

        if not is_rel:
            self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_outputs]))
        else:
            self.covar_module = RelationalRFFKernel(
                train_x,
                train_graph,
                num_samples,
                batch_shape=torch.Size([num_outputs]),
            )
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn( #TODO: turn on batch mode later
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
        

def generate_rff(z, lengthscale, normalize=False, num_features=20, center=True, qmc=True):
    if qmc:
        # space filling design on [0, 1]^D
        sequencer = ghalton.GeneralizedHalton(num_features)
        # actual random feature values come from the inverse cdf
        randn_weights = torch.distributions.Normal(0., 1.).icdf(
            torch.tensor(sequencer.get(z.shape[-1])).type(torch.float)
        )
    else:
        randn_weights =  torch.randn(z.shape[-1], num_features)
    x = z.matmul(randn_weights / lengthscale.transpose(-1, -2))
    x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    if normalize:
        x= x / math.sqrt(num_features)
    if center:
        return x - x.mean(0)
    else:
        return x

def infer_bandwidth(Y):
    if np.max(Y) <= 1e-5:
        return 0.0
    dist_matrix = euclidean_distances(Y, Y, squared=True)
    dist_vector = dist_matrix[np.nonzero(np.tril(dist_matrix))]
    dist_median = np.median(dist_vector)
    return dist_median

def train_relational_gp(X, y, A, is_rel=False, num_samples=20, y_bandwidth=torch.tensor([[1]]), training_iterations=100):
    # @TODO: this should be replaced with something  that uses the kernel empirical bayes    # to infer the bandwidth for y
    y_bw = infer_bandwidth(y.numpy())
    rff_features = generate_rff(y, lengthscale=torch.tensor([[y_bw]]),  normalize=True,  num_features=num_samples, center=False, qmc=False)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2 * num_samples)
    model = BatchIndependentMultitaskGPModel(X, rff_features, A, likelihood, is_rel, num_outputs=2 * num_samples)
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    x_indices = torch.arange(X.shape[-2])
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(x_indices)
        loss = -mll(output, rff_features)
        loss.backward()
        if i % 10 == 0:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()
    residuals = y - likelihood(model(x_indices)).mean
    return model, likelihood, residuals


def p_value_of(val, data):
    """The percentile of a value given a data"""

    data = np.sort(data)
    return float(1 - np.searchsorted(data, val, side='right') / len(data))


def H(N):
    """Centering matrix for N samples"""
    return np.eye(N) - 1 / N * np.ones((N, N))


def CI_test(x, y, Z, A, is_x_rel=False, is_y_rel=False, is_z_rel=False, num_samples=200, x_bandwidth=1, y_bandwidth=1, training_iterations=100, num_boot=5000):
    _, _, res_x = train_relational_gp(X=Z, y=x, A=A, is_rel=is_z_rel)
    _, _, res_y = train_relational_gp(X=Z, y=y, A=A, is_rel=is_z_rel)

    D = torch.diagflat(1 / A.sum(1))

    if is_x_rel:
        res_x = D @ A @ res_x

    if is_y_rel:
        res_y = D @ A @ res_y

    N = D.shape[0]
    res_x = res_x.detach().numpy()
    res_y = res_y.detach().numpy()

    def stat(x, y):
        # return float(((x.T @ y) ** 2).sum())
        return float((((1 / N) * x.T @ H(N) @ y) ** 2).sum())

    test_statistics = stat(res_x, res_y)

    def permute(D):
        perm_i = np.random.permutation(D.shape[0])
        # perm_j = np.random.permutation(D.shape[1])
        return D[np.ix_(perm_i)]

    null_distribution = [stat(res_x, permute(res_y)) for _ in range(num_boot)]

    return p_value_of(test_statistics, null_distribution)
