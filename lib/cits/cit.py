from abc import ABC, abstractmethod
import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.preprocessing import scale


from sdcit.utils import centering, p_value_of, pdinv, residual_kernel, truncated_eigen, eigdec
from sdcit.kcit import kcit_null, python_kcit_K


class ConditionalTest(ABC):

    def __init__(self, seed):
        self.seed = seed

    @abstractmethod
    def get_data(g, cond=False, rmask='000', sample_size=-1):
        pass


    @abstractmethod
    def run_test(g, cond=False, rmask='000', sample_size=-1):
        pass


    def sample_nodes(self, g, size):
        np.random.seed(self.seed)
        sampled_nodes = np.random.choice(g.nodes(), size, replace=False)
        return sampled_nodes


    def infer_gamma(self, A):

        if np.max(A) <= 1e-5:
            return 0.0

        A = scale(A)
        dist_matrix = euclidean_distances(A, A, None, squared=True)
        dist_vector = dist_matrix[np.nonzero(np.tril(dist_matrix))]
        dist_median = np.median(dist_vector)
        return dist_median


    def base_kernel(self, A, s=1.):
        """ Compute radial basis function kernel.

        Parameters:
            A -- Feature matrix.
            s -- Scale parameter (positive float, 1.0 by default).
            
        Return:
            K -- Radial basis function kernel matrix.

        Source: https://github.com/gzampieri/Scuba/blob/master/compute_kernel.py
        """

        gamma = self.infer_gamma(A)
        K = rbf_kernel(A, None, gamma*s)
        return K


    def calculate_rmse(self, model, X_test, Y_test):
        mu, var = model.predict_y(X_test)
        rmse = np.sqrt(((mu - Y_test)**2).mean())
        return rmse


    def residual_kernel(self, K_Y: np.ndarray, K_X: np.ndarray, use_expectation=True, with_gp=True, sigma_squared=1e-3, return_learned_K_X=False):
        """Kernel matrix of residual of Y given X based on their kernel matrices, Y=f(X)"""
        import gpflow
        from gpflow.kernels import White, Linear
        from gpflow.models import GPR

        K_Y, K_X = centering(K_Y), centering(K_X)
        T = len(K_Y)

        if with_gp:
            eig_Ky, eiy = truncated_eigen(*eigdec(K_Y, min(100, T // 4)))
            eig_Kx, eix = truncated_eigen(*eigdec(K_X, min(100, T // 4)))

            X = eix @ np.diag(np.sqrt(eig_Kx))  # X @ X.T is close to K_X
            Y = eiy @ np.diag(np.sqrt(eig_Ky))
            n_feats = X.shape[1]

            linear = Linear(n_feats, ARD=True)
            white = White(n_feats)
            gp_model = GPR(X, Y, linear + white)
            gpflow.train.ScipyOptimizer().minimize(gp_model)

            print('RMSE: %f' % self.calculate_rmse(gp_model, X, Y))
            # pdb.set_trace()

            K_X = linear.compute_K_symm(X)
            sigma_squared = white.variance.value

        P = pdinv(np.eye(T) + K_X / sigma_squared)  # == I-K @ inv(K+Sigma) in Zhang et al. 2011
        if use_expectation:  # Flaxman et al. 2016 Gaussian Processes for Independence Tests with Non-iid Data in Causal Inference.
            RK = (K_X + P @ K_Y) @ P
        else:  # Zhang et al. 2011. Kernel-based Conditional Independence Test and Application in Causal Discovery.
            RK = P @ K_Y @ P

        if return_learned_K_X:
            return RK, K_X
        else:
            return RK