import numpy as np
import networkx as nx

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from ..cits.cit import ConditionalTest


class NaiveCITest(ConditionalTest):

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
                    _x.append(g.nodes[n]['trt'])
                    _y.append(out)
                    if z:
                        _z.append(z)
            elif rmask == '001':
                for n in neighbors:
                    _x.append(trt)
                    _y.append(out)
                    _z.append(g.nodes[n]['att_0'])
            elif rmask == '101':
                for i1 in range(len(neighbors)):
                    for i2 in range(len(neighbors)):
                        n1, n2 = neighbors[i1], neighbors[i2]
                        _x.append(g.nodes[n1]['trt'])
                        _y.append(out)
                        _z.append(g.nodes[n2]['att_0'])
            else:
                _x.append(trt)
                _y.append(out)
                if z:
                    _z.append(z)

        X = np.array(_x)
        Y = np.array(_y)

        Z = None
        if len(_z) > 0:
            Z = np.array(_z)

        return X, Y, Z


    def run_test(self, g, cond=False, rmask='000', samples=[], approx=False):
        # build data
        X, Y, Z = self.get_data(g, cond=False, rmask=rmask)

        # run indep test
        if cond:
            Z = np.array(Z).reshape(-1, 1)

            Rzx = LinearRegression().fit(Z, X)
            Ezx = X - Rzx.predict(Z)

            Rzy = LinearRegression().fit(Z, Y)
            Ezy = Y - Rzx.predict(Z)

            _, p_value = pearsonr(Ezx, Ezy)
        else:
            _, p_value = pearsonr(X, Y)

        return p_value
        