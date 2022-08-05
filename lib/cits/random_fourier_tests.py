import numpy as np


def degree_matrix(adj_mat):
    """Given an adjacency matrix compute the diagonal with D_{i,i} = 1/degree(i)"""
    D = np.diag(1 / np.asarray(adj_mat.sum(1)).reshape(-1))
    return D


def RFF(X, num_features, mean=0, sd=1.0):
    """Compute the random fourier features"""
    projection_mat = np.random.randn(X.shape[1], num_features // 2) * sd + mean
    return np.sqrt(2 / num_features) * np.hstack(
        (np.cos(X @ projection_mat), np.sin(X @ projection_mat))
    )


def H(N):
    """Centering matrix for N samples"""
    return np.eye(N) - 1 / N * np.ones((N, N))


def fourier_hsic(x, y, num_features, compute_pval=True, permutations=100):
    """Compute the random fourier features approximation to HSIC"""
    N = x.shape[0]
    proj_x = RFF(x, num_features)
    proj_y = RFF(y, num_features)
    stat = lambda y_projs: (((1 / N) * proj_x.T @ H(N) @ y_projs) ** 2).sum()
    orig_stat = stat(proj_y)
    if compute_pval:
        null = [
            stat(proj_y[np.random.permutation(np.arange(N)), :])
            for _ in range(permutations)
        ]
    else:
        null = None
        return orig_stat, null


def relational_fourier_hsic(
    x, y, adj_mat, num_features, samples=[], compute_pval=True, permutations=100
):
    """Compute the random fourier features approximation to HSIC"""
    N = x.shape[0] if not samples else len(samples)
    D = degree_matrix(adj_mat)
    relational_proj_x = D @ adj_mat @ RFF(x, num_features)
    proj_y = RFF(y, num_features)

    if samples:
        relational_proj_x = relational_proj_x[samples]
        proj_y = proj_y[samples]

    stat = lambda y_projs: (((1 / N) * relational_proj_x.T @ H(N) @ y_projs) ** 2).sum()
    orig_stat = stat(proj_y)
    if compute_pval:
        null = [
            stat(proj_y[np.random.permutation(np.arange(N)), :])
            for _ in range(permutations)
        ]
    else:
        null = None
    return orig_stat, null
