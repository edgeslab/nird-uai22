import math
import pdb

import numpy as np
import networkx as nx
from scipy import stats
from collections import Counter
from sklearn.preprocessing import MinMaxScaler



def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def entropy(labels):
    return stats.entropy(np.bincount(labels), base=2)

def complex_agg(x):
    return np.mean(x) + np.var(x)

class CausalAttributeGenerator():

    AGG_MAP = {
        'mean'		:	np.mean,
        'sum'		:	np.sum,
        'entropy'	: 	entropy,
        'complex'	:	complex_agg
    }

    def __init__(self, params):
        self.num_covar = params['num_covar']
        self.num_hops = params['num_hops']
        self.test_type = params['test_type']
        self.hypothesis = params['hypothesis']
        self.network_type = params['network_type']
        self.dependence = params['dependence'] if 'dependence' in params else None
        self.aggregate = params['aggregate'] if 'aggregate' in params else None
        self.dep_type = params['dep_type'] if 'dep_type' in params else None
        self.alpha = params['alpha'] if 'alpha' in params else None
        self.d_steps = params['d_steps'] if 'd_steps' in params else None
        self.conf_coeff = params['conf_coeff'] if 'conf_coeff' in params else None
        self.distribution = params['distribution'] if 'distribution' in params else 'uniform'
        self.noise_scale = params['noise_scale'] if 'noise_scale' in params else 1.0
        
        np.random.seed(params['seed'])


    def plot_histogram(self, data):
        import matplotlib.pyplot as plt
        # plt.hist(data, normed=True, stacked=True, bins=10)
        plt.hist(data, normed=False, bins=10)
        plt.show()


    def get_node_neighbors(self, graph):
        lengths = [i for i in nx.all_pairs_shortest_path_length(graph, cutoff=self.num_hops)]
        node_neighbors = []
        for i in range(len(graph)):
            neighbors = list(lengths[i][1].keys())[1:]
            node_neighbors.append(neighbors)
        return node_neighbors, lengths


    def augment_bipartite(self, graph, case):

        a_samples = [k for k,v in dict(graph.nodes(data=True)).items() if v['item_class'] == 'A']
        b_samples = [k for k,v in dict(graph.nodes(data=True)).items() if v['item_class'] == 'B']

        A = np.array(nx.adjacency_matrix(graph).todense())
        A = A[a_samples, :][:, b_samples]
        D = np.diag(1 / np.asarray(A.sum(axis=1)).reshape(-1))

        # Generate att, Z ~ Unif
        Z = np.random.rand(len(b_samples))

        # Generate treatments, X = f(rel(Z))
        noise = np.random.normal(scale=self.noise_scale, size=len(a_samples))
        X = (D @ A @ Z) * self.conf_coeff + noise
        # X = self.sample_binary_random_probs(X)

        # Generate outcomes, Y = f(Z, rel(X))
        noise = np.random.normal(scale=self.noise_scale, size=len(a_samples))
        Y = (D @ A @ Z) * self.conf_coeff + X * self.dependence + noise

        nx.set_node_attributes(graph, values=dict(zip(a_samples, X)), name='trt')
        nx.set_node_attributes(graph, values=dict(zip(a_samples, Y)), name='out')
        nx.set_node_attributes(graph, values=dict(zip(b_samples, Z)), name='att_0')

        return graph


    def augment(self, graph, case):
        self.node_neighbors, _ = self.get_node_neighbors(graph)

        if self.network_type == 'bipartite':
            if self.test_type == "conditional":
                return self.augment_bipartite(graph, case)
            else:
                raise "Bipartite for Marginal not implemented :("

        if self.test_type == "marginal":
            if self.aggregate == "ltm":
                return self.augment_marginal_ltm(graph, case)
            else:
                return self.augment_marginal(graph, case)
        elif self.test_type == "conditional":
            method = getattr(self, 'augment_conditional_%s_%s' % (case, self.hypothesis))
            return method(graph, case)


    def get_agg_values(self, values, index, neighbor_map, is_rel=False):
        if is_rel:
            agg_f = CausalAttributeGenerator.AGG_MAP[self.aggregate]
            return agg_f(values[neighbor_map[index]])
        else:
            return values[index]


    def equation(self, x1, a1, x2=None, a2=None):
        assert((x2 is None) == (a2 is None))
        
        noise = np.random.normal(scale=self.noise_scale, size=1)

        y = 0
        if self.dep_type == 'linear':
            y = x1 * a1 + noise + (x2 * a2 if x2 is not None else 0)
        elif self.dep_type == 'polynomial':
            y = a1 * x1**3 
            y += (a2 * x2**2) if x2 is not None else 0
            y += noise 
        else:
            raise 'Undefined dep_type!'

        return y[0]

    
    def sample_from_dist(self, size):
        if self.distribution == 'uniform':
            return np.random.rand(size)
        elif self.distribution == 'normal':
            return np.random.normal(0, 0.1, size)


    def sample_binary_random(self, num):
        probs = np.array([0.5] * num)
        random = self.sample_from_dist(len(probs))
        #TODO: fix for normal
        return np.less_equal(random, probs).astype(int)

    
    def sample_binary_random_probs(self, probs):
        if np.min(probs) < 0:
            probs = probs - np.min(probs)
        probs = probs / (np.max(probs) - np.min(probs))
        threshold = np.array([0.5] * len(probs))
        return np.less_equal(threshold, probs).astype(int)

    
    def get_rel_dependence(self, N, R1, R2=None, I=None, a1=None, a2=None):

        vals = []
        for i in range(N):
            _x1 = self.get_agg_values(R1, i, self.node_neighbors, is_rel=True) if R1 is not None else I[i]

            _x2 = None
            if R2 is not None:
                _x2 = self.get_agg_values(R2, i, self.node_neighbors, is_rel=True)
            elif (R1 is not None) and (I is not None):
                _x2 = I[i]

            _val = self.equation(x1=_x1, x2=_x2, a1=a1, a2=a2)
            vals.append(_val)
        return np.array(vals)


    def augment_marginal(self, graph, case):
        '''
            Null:	rel(X) _|_ Y
            Alt:	rel(X) -> Y
        '''

        # Generate treatments, X ~ Unif
        treatments = self.sample_binary_random(len(graph))

        # Generate outcomes, y = f(rel(X))
        node_neighbors, _ = self.get_node_neighbors(graph)

        #TODO: generalize based on case
        outcomes = []
        for i in range(len(treatments)):
            _treatment = self.get_agg_values(treatments, i, node_neighbors, case[1] == 'b')	# only works for 1a,1b,2a,2b
            # _outcome = (self.dependence * _treatment + np.random.normal(scale=0.5, size=1))[0]
            _outcome = self.equation(x1=_treatment, a1=self.dependence)
            outcomes.append(_outcome)

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph


    def augment_marginal_ltm(self, graph, case):
        '''
            Null:	rel(X) _|_ Y
            Alt:	rel(X) -> Y
        '''

        # Generate treatments, X ~ Unif
        treatments = self.sample_binary_random(len(graph))

        if self.hypothesis == "null":
            probs = np.array([0.5] * len(graph))
            random = self.sample_from_dist(len(probs))
            outcomes = np.less_equal(random, probs).astype(int)
        else:
            probs = np.array([0.001] * len(graph))
            random = self.sample_from_dist(len(probs))
            treatments = np.less_equal(random, probs).astype(int)

            node_neighbors, _ = self.get_node_neighbors(graph)
            thresholds = np.random.beta(a=self.alpha, b=self.alpha, size=len(treatments))
            # thresholds = np.random.normal(loc=0.5, scale = self.alpha, size=len(treatments))
            # thresholds = np.random.uniform(low=self.alpha-0.1, high=self.alpha+0.1, size=len(treatments))

            if self.d_steps:
                for j in range(self.d_steps):
                    probs = [np.mean(treatments[nbr]) for nbr in node_neighbors]
                    treatments = np.less_equal(thresholds, probs).astype(int)
            
            outcomes = []
            for i in range(len(treatments)):
                _treatment = np.mean(treatments[node_neighbors[i]])
                _outcome = int(_treatment > thresholds[i])
                outcomes.append(_outcome)

            # _t = np.array([np.mean(treatments[node_neighbors[i]]) for i in range(len(treatments))])
            # print('_t mean: %f' % np.mean(_t))
            # print('_t var: %f' % np.var(_t))
            # print('Y1 mean: %f' % np.mean(_t[np.nonzero(outcomes)[0]]))
            # _not_out = np.logical_xor(outcomes, np.ones(len(outcomes)))
            # print('Y0 mean: %f' % np.mean(_t[np.nonzero(_not_out)[0]]))
            # print('Threshold mean: %f' % np.mean(thresholds))
            # print('Threshold var: %f' % np.std(thresholds))
            # self.plot_histogram(_t)
            # pdb.set_trace()

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph
    

    def augment_conditional_2b_alt(self, graph, case):
        '''V structure: rel(X) <- Z -> Y <- rel(X)'''

        # Generate att, Z ~ Unif
        att_keys = ["att_{0}".format(i) for i in range(self.num_covar)]
        for j in range(self.num_covar):
            vals = self.sample_from_dist(len(graph))
            nx.set_node_attributes(graph, values=dict(enumerate(vals)), name=att_keys[j])

        # Generate treatments, X = f(rel(Z))
        _z = np.array([graph.nodes[i]['att_0'] for i in range(len(graph))])
        treatments = self.get_rel_dependence(N=len(graph), R1=_z, a1=self.conf_coeff)
        treatments = self.sample_binary_random_probs(treatments)

        # Generate outcomes, Y = f(Z, rel(X))
        outcomes = self.get_rel_dependence(N=len(graph), R1=treatments, I=_z, a1=self.dependence, a2=self.conf_coeff)

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph


    def augment_conditional_2d_alt(self, graph, case):
        '''V structure: X <- rel(Z) -> Y <- X'''

        # Generate att, Z ~ Unif
        att_keys = ["att_{0}".format(i) for i in range(self.num_covar)]
        for j in range(self.num_covar):
            vals = self.sample_from_dist(len(graph))
            nx.set_node_attributes(graph, values=dict(enumerate(vals)), name=att_keys[j])
        
        # Generate treatments, X = f(rel(Z))
        _z = np.array([graph.nodes[i]['att_0'] for i in range(len(graph))])
        treatments = self.get_rel_dependence(N=len(graph), R1=_z, a1=self.conf_coeff)
        treatments = self.sample_binary_random_probs(treatments)

        # Generate outcomes, Y = f(rel(Z), X)
        outcomes = self.get_rel_dependence(N=len(graph), R1=_z, I=treatments, a1=self.conf_coeff, a2=self.dependence)

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph


    def augment_conditional_2e_alt(self, graph, case):
        '''V structure: rel(X) <- rel(Z) -> Y <- rel(X)'''

        # Generate att, Z ~ Unif
        att_keys = ["att_{0}".format(i) for i in range(self.num_covar)]
        for j in range(self.num_covar):
            vals = self.sample_from_dist(len(graph))
            nx.set_node_attributes(graph, values=dict(enumerate(vals)), name=att_keys[j])

        # Generate treatments, X = f(Z)
        _z = np.array([graph.nodes[i]['att_0'] for i in range(len(graph))])
        treatments = self.get_rel_dependence(N=len(graph), R1=None, I=_z, a1=self.conf_coeff)
        treatments = self.sample_binary_random_probs(treatments)

        # Generate outcomes, Y = f(rel(Z), rel(X))
        outcomes = self.get_rel_dependence(N=len(graph), R1=_z, R2=treatments, a1=self.conf_coeff, a2=self.dependence)

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph

    
    def augment_conditional_2b_null(self, graph, case):
        '''V structure: rel(X) <- Z -> Y'''

        # Generate att, Z = f(rel(X))
        att_keys = ["att_{0}".format(i) for i in range(self.num_covar)]
        for j in range(self.num_covar):
            vals = self.sample_from_dist(len(graph))
            nx.set_node_attributes(graph, values=dict(enumerate(vals)), name=att_keys[j])

        # Generate outcomes, X = f(rel(Z))
        _z = np.array([graph.nodes[i]['att_0'] for i in range(len(graph))])
        treatments = self.get_rel_dependence(N=len(graph), R1=_z, a1=self.conf_coeff)
        treatments = self.sample_binary_random_probs(treatments)

        # Generate outcomes, Y = f(Z)
        outcomes = self.get_rel_dependence(N=len(graph), R1=None, I=_z, a1=self.conf_coeff)

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph


    def augment_conditional_2d_null(self, graph, case):
        '''V structure: X <- rel(Z) -> Y'''

        # Generate att, Z = f(rel(X))
        att_keys = ["att_{0}".format(i) for i in range(self.num_covar)]
        for j in range(self.num_covar):
            vals = self.sample_from_dist(len(graph))
            nx.set_node_attributes(graph, values=dict(enumerate(vals)), name=att_keys[j])

        # Generate outcomes, X = f(rel(Z))
        _z = np.array([graph.nodes[i]['att_0'] for i in range(len(graph))])
        treatments = self.get_rel_dependence(N=len(graph), R1=_z, a1=self.conf_coeff)
        treatments = self.sample_binary_random_probs(treatments)

        # Generate outcomes, Y = f(rel(Z))
        outcomes = self.get_rel_dependence(N=len(treatments), R1=_z, a1=self.conf_coeff)

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph


    def augment_conditional_2e_null(self, graph, case):
        '''V structure: rel(X) <- rel(Z) -> Y'''

        # Generate att, Z = f(rel(X))
        att_keys = ["att_{0}".format(i) for i in range(self.num_covar)]
        for j in range(self.num_covar):
            vals = self.sample_from_dist(len(graph))
            nx.set_node_attributes(graph, values=dict(enumerate(vals)), name=att_keys[j])

        # Generate outcomes, X = f(rel(Z))
        _z = np.array([graph.nodes[i]['att_0'] for i in range(len(graph))])
        treatments = self.get_rel_dependence(N=len(graph), R1=None, I=_z, a1=self.conf_coeff)
        treatments = self.sample_binary_random_probs(treatments)

        # Generate outcomes, Y = f(rel(Z))
        outcomes = self.get_rel_dependence(N=len(treatments), R1=_z, a1=self.conf_coeff)

        nx.set_node_attributes(graph, values=dict(enumerate(treatments)), name='trt')
        nx.set_node_attributes(graph, values=dict(enumerate(outcomes)), name='out')

        return graph


