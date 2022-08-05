import os
import pdb
import numpy as np
import pandas as pd
import networkx as nx


def make_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


def print_graph_stat(g):
    print(nx.info(g))
    print('#components: %d' % nx.number_connected_components(g))
    
    strt = [v for k,v in nx.get_node_attributes(g, name='trt').items()]
    sout = [v for k,v in nx.get_node_attributes(g, name='out').items()]
    
    print('treatment: ', np.bincount(strt))
    print('outcome: ', np.bincount(sout))
    print('')


def sample_random_nodes(g, sample_size):
    sample_g = g.copy()
    A = np.array(nx.adjacency_matrix(sample_g).todense())
    non_fringe_nodes = np.nonzero(A.sum(1) > 1)[0]
    # non_fringe_non_hateful_nodes = list(filter(lambda x: sample_g.nodes()[x]['trt'] == 0, non_fringe_nodes))

    sample_nodes = np.random.choice(non_fringe_nodes, sample_size, replace=False)
    sample_edges = sample_g.edges(sample_nodes)
    sample_g = nx.edge_subgraph(sample_g, sample_edges).copy()

    # sample_nodes = g.nodes()
    # became_hateful = list(filter(lambda x: sample_g.nodes()[x]['out'] == 1, sample_nodes))
    # non_hatefuls = list(set(sample_nodes) - set(became_hateful))
    # hp = []
    # hp0 = 0
    # for bh in non_hatefuls:
    #     hc = 0
    #     nbrs = list(sample_g.neighbors(bh))
    #     for n in nbrs:
    #         if sample_g.nodes()[n]['trt'] == 1:
    #             hc += 1
    #     hp.append(hc / len(nbrs))
    #     if hc <= 1e-5:
    #         hp0 += 1

    # print('number of samples: %d' % len(sample_nodes))
    # print('zero hateful friends: %d' % hp0)
    # print('avg proportion of hateful friends: %0.04f' % np.mean(hp))

    return sample_g, set(sample_nodes)


def sample_random_edges(g, sample_size):
    num_edges = int(sample_size * 3/4)

    sample_edge_indices = np.random.choice(range(len(g.edges())), num_edges, replace=False)
    edges = np.array(g.edges())
    sample_edges = edges[sample_edge_indices]
    
    sample_nodes = set()
    for edge in sample_edges:
        # sample_nodes.add(edge[0])
        # sample_nodes.add(edge[1])
        if g.nodes()[edge[0]]['trt'] == 0:
            sample_nodes.add(edge[0])
        if g.nodes()[edge[1]]['trt'] == 0:
            sample_nodes.add(edge[1])
    sample_nodes = list(sample_nodes)

    sample_nodes = [k for k,v in dict(filter(lambda e: e[1] > 1, dict(g.degree(sample_nodes)).items())).items()]
    sample_nodes = sample_nodes[:sample_size]

    sample_edges = g.edges(sample_nodes)
    sample_g = nx.edge_subgraph(g, sample_edges).copy()
    sample_g.remove_nodes_from(list(nx.isolates(sample_g)))

    return sample_g, sample_nodes