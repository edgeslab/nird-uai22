import pdb

import numpy as np
import networkx as nx
from sdcit.hsic import HSIC_boot
# # from sdcit.sdcit import SDCIT
# from sdcit.utils import centering, p_value_of, pdinv, residual_kernel, truncated_eigen, eigdec
# from sdcit.kcit import kcit_null, python_kcit_K

from .random_fourier_tests import *
from ..cits.cit import ConditionalTest


class MCITest(ConditionalTest):

    def get_data(self, g, cond=False, rmask='000'):
        X = np.array([v['trt'] for k,v in dict(g.nodes(data=True)).items() if 'trt' in v]).reshape(-1, 1)
        Y = np.array([v['out'] for k,v in dict(g.nodes(data=True)).items() if 'out' in v]).reshape(-1, 1)

        Z = None
        if cond:
            Z = np.array([v['att_0'] for k,v in dict(g.nodes(data=True)).items() if 'att_0' in v]).reshape(-1, 1)

        return X, Y, Z


    def relational_kernel(self, V, A, samples=[]):
        K = self.base_kernel(V)
        D = np.diag(1 / np.array(A).sum(1))
        if samples:
            Ds = D[samples, :][:, samples]
            return Ds @ A[samples, :] @ K @ A[:, samples] @ Ds
        return D @ A @ K @ A @ D


    def python_kcit_X(self, Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray, A: np.ndarray, is_rx=False, is_ry=False, is_rz=False, alpha=0.05, with_gp=True, sigma_squared=1e-3, num_bootstrap_for_null=5000, seed=None):
        """ A test for X _||_ Y | Z using KCIT with Gram matrices for X, Y, and Z
            see `kcit_null` for the output
        """
        if seed is not None:
            np.random.seed(seed)

        T = len(Kx)

        Kx, Ky, Kz = centering(Kx * Kz), centering(Ky), centering(Kz)
        D = np.diag(1 / np.array(A).sum(1))

        if with_gp:
            Kxz = self.residual_kernel(Kx, Kz, use_expectation=False, with_gp=with_gp)
            if is_rz:
                Kz = D @ A @ Kz
            Kyz = self.residual_kernel(Ky, Kz, use_expectation=False, with_gp=with_gp)
        else:
            P = eye(T) - Kz @ pdinv(Kz + sigma_squared * eye(T))
            Kxz = P @ Kx @ P.T
            Kyz = P @ Ky @ P.T

        if is_rx:
            Kxz = D @ A @ Kxz

        test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

        return kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic)


    def print_stat(self, X, Y, Z, A, trial):
        D = np.diag(1 / np.asarray(A.sum(1)).reshape(-1))
        import matplotlib.pyplot as plt

        hypo = 'alt'

        # plt.scatter(D @ A @ Z, X)
        # plt.xlabel("mean(Z)", fontsize=18)
        # plt.ylabel("X", fontsize=18)
        # plt.savefig('case_1_%s_x_mz.png' % hypo, format='png')
        # plt.clf()

        # plt.scatter(D @ A @ Z, Y)
        # plt.xlabel("mean(Z)", fontsize=18)
        # plt.ylabel("Y", fontsize=18)
        # plt.savefig('case_1_%s_y_mz.png' % hypo, format='png')
        # plt.clf()

        # plt.scatter(X, Y)
        # plt.xlabel("X", fontsize=18)
        # plt.ylabel("Y", fontsize=18)
        # plt.savefig('case_1_%s_y_x.png' % hypo, format='png')
        # plt.clf()

        plt.scatter(D @ A @ X, Y)
        plt.xlabel("mean(X)", fontsize=18)
        plt.ylabel("Y", fontsize=18)
        plt.savefig('case_1b_%s_y_mx_01.png' % hypo, format='png')
        plt.clf()


        # np.savetxt('data/exp_2d_two/%s/X/trial_%d.csv' % (hypo, trial), X, delimiter=',')
        # np.savetxt('data/exp_2d_two/%s/Y/trial_%d.csv' % (hypo, trial), Y, delimiter=',')
        # np.savetxt('data/exp_2d_two/%s/Z/trial_%d.csv' % (hypo, trial), Z, delimiter=',')
        # np.savetxt('data/exp_2d_two/%s/A/trial_%d.csv' % (hypo, trial), A, delimiter=',')

        pdb.set_trace()

        


    def run_test(self, g, cond=False, rmask='000', samples=[], approx=False, trial=0):
        # build data
        X, Y, Z = self.get_data(g, cond, rmask)
        is_rx = rmask[0]=='1'
        is_ry = rmask[1]=='1'
        is_rz = rmask[2]=='1'

        # create kernels
        N = len(g)
        A = np.array(nx.adjacency_matrix(g).todense())

        # A = A[:X.shape[0], :][:, X.shape[0]:]
        # self.print_stat(X, Y, Z, A, trial)
        # return 0

        if approx:
            from sdcit.utils import p_value_of
            test, null = relational_fourier_hsic(X, Y, A, num_features=50, samples=samples)
            return p_value_of(test, null)

        def get_kernel(V, A, is_rel=False):
            if samples:
                return self.relational_kernel(V, A, samples) if is_rel else self.base_kernel(V[samples])
            else:
                return self.relational_kernel(V, A) if is_rel else self.base_kernel(V)


        # run indep test
        if cond:
            import torch
            from ..cits.relational.relational_gp import CI_test

            tX = torch.Tensor(X)
            tY = torch.Tensor(Y)
            tZ = torch.Tensor(Z)
            # pdb.set_trace()
            p_value = CI_test(tX, tY, tZ, torch.Tensor(A), is_rx, is_ry, is_rz)
            # print(p_value)

            # Kx = get_kernel(X, A)
            # Ky = get_kernel(Y, A, is_ry)
            # Kz = get_kernel(Z, A)
            
            # p_value = self.python_kcit_X(Kx, Ky, Kz, A, is_rx, is_ry, is_rz)[2]

            # XZ = np.concatenate((X, Z), axis=1)
            # YZ = np.concatenate((Y, Z), axis=1)

            # Kxz = get_kernel(XZ, A, rmask[0] == '1')
            # Kyz = get_kernel(YZ, A, rmask[1] == '1')
            # Kz = get_kernel(Z, A, rmask[2] == '1')

            # p_value = python_kcit_K(Kxz, Kyz, Kz)[2]

        else:
            Kx = get_kernel(X, A, is_rx)
            Ky = get_kernel(Y, A, is_ry)

            p_value = HSIC_boot(Kx, Ky, num_boot=N*2)

        return p_value