import numpy as np
import networkx as nx

from pygk.utils import KGraph
# from sdcit.sdcit import SDCIT
from sdcit.hsic import HSIC_boot
from sdcit.utils import centering, p_value_of, pdinv, residual_kernel, truncated_eigen, eigdec
from sdcit.kcit import kcit_null, python_kcit_K
from pygk.labeled import labeled_shortest_path_kernel
from uai2017experiments.utils import normalize_by_diag, c_kernel_matrix

from ..cits.cit import ConditionalTest


class KRCITest(ConditionalTest):

    def get_data(self, g, cond=False, rmask='000'):
        #TODO: add hops parameter

        _x = []
        _y = []
        _z = [] #TODO; handle it

        for i in g.nodes():
            trt = g.nodes[i]['trt']
            out = g.nodes[i]['out']
            z = g.nodes[i]['att_0'] if 'att_0' in g.nodes[i] else None

            neighbors = list(g.neighbors(n=i))
            if rmask == '100':
                for n in neighbors:
                    _x.append(((n, g.nodes[n]['trt']),))
                    _y.append(((i, out),))
                    if z:
                        _z.append(((i, z),))
            elif rmask == '001':
                for n in neighbors:
                    _x.append(((i, trt),))
                    _y.append(((i, out),))
                    _z.append(((n, g.nodes[n]['att_0']),))
            elif rmask == '101':
                for i1 in range(len(neighbors)):
                    for i2 in range(len(neighbors)):
                        n1, n2 = neighbors[i1], neighbors[i2]
                        _x.append(((n1, g.nodes[n1]['trt']),))
                        _y.append(((i, out),))
                        _z.append(((n2, g.nodes[n2]['att_0']),))
            else:
                _x.append(((i, trt),))
                _y.append(((i, out),))
                if z:
                    _z.append(((i, z),))

        X = np.empty((len(_x),),dtype=object)
        X[:] = _x

        Y = np.empty((len(_y),),dtype=object)
        Y[:] = _y

        #TODO: handle Z
        if len(_z) > 0:
            Z = np.empty((len(_z),),dtype=object)
            Z[:] = _z
        else:
            Z = None

        return X, Y, Z

    
    def indexize(self, V):
        _v = []
        for i in range(len(V)):
            _v.append(((i, V[i]),))

        V = np.empty((len(_v),),dtype=object)
        V[:] = _v
        return V


    def python_kcit_X(self, Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray, alpha=0.05, with_gp=True, sigma_squared=1e-3, num_bootstrap_for_null=5000, seed=None):
        """ A test for X _||_ Y | Z using KCIT with Gram matrices for X, Y, and Z
            see `kcit_null` for the output
        """
        if seed is not None:
            np.random.seed(seed)

        T = len(Kx)

        Kx, Ky, Kz = centering(Kx * Kz), centering(Ky), centering(Kz)

        if with_gp:
            Kxz = self.residual_kernel(Kx, Kz, use_expectation=False, with_gp=with_gp)
            Kyz = self.residual_kernel(Ky, Kz, use_expectation=False, with_gp=with_gp)
        else:
            P = eye(T) - Kz @ pdinv(Kz + sigma_squared * eye(T))
            Kxz = P @ Kx @ P.T
            Kyz = P @ Ky @ P.T

        test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

        return kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic)


    def shortest_path_kernel_matrix(self, entities_ug, vertex_kernel_hop):
        """a precomputed kernel matrix-based shortest path kernel function given a k-hop neighborhood subgraph"""

        entities_ordered = list(entities_ug.nodes)
        kgraphs = [None] * len(entities_ordered)
        for i, entity in enumerate(entities_ordered):
            reachable = list(nx.single_source_shortest_path_length(entities_ug, entity, vertex_kernel_hop).keys())
            kgraphs[i] = KGraph(entities_ug.subgraph(reachable), 'item_class', {entity: 'special_label_yo'})

        index_of = {item: index for index, item in enumerate(entities_ordered)}
        VK, _ = labeled_shortest_path_kernel(kgraphs)

        return VK, index_of


    def run_test(self, g, cond=False, rmask='000', samples=[], approx=False):
        # build data
        X, Y, Z = self.get_data(g, cond=False, rmask=rmask)
        is_rx = rmask[0]=='1'
        is_ry = rmask[1]=='1'
        is_rz = rmask[2]=='1'

        # create kernels
        VK, index_of = self.shortest_path_kernel_matrix(g, vertex_kernel_hop=1)
        VK = normalize_by_diag(VK)

        def get_kernel(V, is_rel=False):
            if is_rel:
                gamma_v = self.infer_gamma(np.asarray([v[0][1] for v in V]).reshape(-1, 1))
                return normalize_by_diag(c_kernel_matrix(V, index_of, VK, 1, gamma_v))
            else:
                return self.base_kernel(np.asarray([v[0][1] for v in V]).reshape(-1, 1))

        K_X = get_kernel(X, is_rx)
        K_Y = get_kernel(Y, is_ry)
        K_Z = get_kernel(Z, is_rz) if Z is not None else np.ones(K_Y.shape)
        # K_X = normalize_by_diag(c_kernel_matrix(X, index_of, VK, 1, gamma))
        # K_Y = normalize_by_diag(c_kernel_matrix(Y, index_of, VK, 1, gamma))
        # K_Z = normalize_by_diag(c_kernel_matrix(Z, index_of, VK, 1, gamma)) if Z is not None else np.ones(K_Y.shape)
        K_X2 = normalize_by_diag(c_kernel_matrix(X, index_of, VK, 1, 1.0, ignore_values=True))
        K_Y2 = normalize_by_diag(c_kernel_matrix(Y, index_of, VK, 1, 1.0, ignore_values=True))
        K_Z2 = normalize_by_diag(c_kernel_matrix(Z, index_of, VK, 1, 1.0, ignore_values=True)) if Z is not None else 1
        K_ZG = K_Z * K_X2 * K_Y2 * K_Z2

        # run indep test
        if cond:
            p_value = self.python_kcit_X(K_X, K_Y, K_ZG)[2]
            # p_value = python_kcit_K(K_X, K_Y, K_ZG)[2]
        else:
            p_value = HSIC_boot(K_X, K_Y, num_boot=len(X)*2)

        return p_value
        