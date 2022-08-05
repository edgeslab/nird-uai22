import os
import pdb
import copy
import json
import argparse
import pickle as pkl
from tqdm import tqdm

import numpy as np
import pandas as pd
import networkx as nx
import random

from plotter import plot_degree_histogram
from lib.synthgen.net_gen import SyntheticRandomNetwork
from lib.synthgen.att_gen import CausalAttributeGenerator


def make_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


def sample_facebook(graph, num_nodes, avg_degree):
    node_dg = list(graph.degree())
    node_dg_sorted = sorted(node_dg, key=lambda x: x[1], reverse=True)
    sorted_nodes = [k for k,v in node_dg_sorted]
    sampled_nodes = np.random.choice(sorted_nodes[:num_nodes*2], num_nodes, replace=False)
    sampled_graph = graph.subgraph(sampled_nodes)
    # print(nx.info(graph))

    def ad(graph):
        return 0 if len(gg) == 0 else np.mean(list(dict(graph.degree()).values())) 

    gg = nx.Graph()
    while(ad(gg) < avg_degree):
        random_nodes = np.random.choice(sampled_nodes, int(avg_degree), replace=False)
        for v in random_nodes:
            edges = np.array(list(sampled_graph.edges(v)))
            if len(edges) == 0:
                continue
            idx = np.random.choice(range(len(edges)), 1)
            gg.add_edges_from(edges[idx])


    nodes_needed = num_nodes - len(gg)
    if nodes_needed > 0:
        all_edges = list(graph.edges(gg.nodes()))
        gg_edges = list(gg.edges())
        other_edges = set(all_edges) - set(gg_edges)
        outer_edges = np.array(list(filter(lambda e: (e[0] not in gg) or (e[1] not in gg), list(other_edges))))
        idx = np.random.choice(range(len(outer_edges)), nodes_needed)
        gg.add_edges_from(outer_edges[idx])

    return gg
    # return nx.Graph(graph)


def build_bipartite(num_nodes, num_matching=1, seed=12345):
    graph = nx.Graph()
    for i in range(num_nodes):
        graph.add_node(i)
    nx.set_node_attributes(graph, 'A', 'item_class')

    counter = num_nodes
    for i in range(num_nodes):
        for j in range(num_matching):
            graph.add_edge(i, counter)
            counter += 1
    matched_nodes = {i:'B' for i in range(num_nodes, counter, 1)}
    nx.set_node_attributes(graph, matched_nodes, 'item_class')

    return graph


def build_graph(config, case, seed, plot_only=False):
    params = config['params']
    np.random.seed(seed)

    if params['network_type'] == 'Facebook':
        graph = nx.read_edgelist('data/graphs/facebook_combined.txt',nodetype=int)
        # graph = sample_facebook(graph, params['num_nodes'], params['avg_degree'])
    elif params['network_type'] == 'Twitter':
        graph = nx.read_graphml('data/graphs/twitter_11k.graphml')
        # from utils import sample_random_nodes
        # sample_random_nodes(graph, 1213)
    elif params['network_type'] == 'Enron':
        # df = pd.read_csv('data/graphs/enron_email.txt', delimiter='\t')
        # graph = nx.from_pandas_edgelist(df, '0', '1')
        # graph = sample_facebook(graph, 20000, 5)
        graph = nx.read_graphml('data/graphs/enron_20k_ad5.graphml')
    elif params['network_type'] == 'Hateful':
        # rt_graph = "data/graphs/hateful/users_clean.graphml"
        # graph = nx.read_graphml(rt_graph)
        graph = nx.read_graphml(path="data/graphs/hateful/hateful_rt_10k.graphml")
        graph = nx.Graph(graph)
    elif params['network_type'] == 'Barabasi_Albert':
        graph = nx.barabasi_albert_graph(params['num_nodes'], params['num_edges'], seed=seed)
    elif params['network_type'] == 'Erdos_Renyi':
        graph = nx.erdos_renyi_graph(params['num_nodes'], params['edge_prob'], seed=seed)
    elif params['network_type'] == 'Watts_Strogatz':
        graph = nx.watts_strogatz_graph(params['num_nodes'], params['knn'], params['rewire_prob'], seed=seed)
    elif params['network_type'] == 'bipartite':
        graph = build_bipartite(params['num_nodes'], params['num_matching'], seed=seed)

    # print(len(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    
    # re-adjust node-ids if isoltes are removed
    mapping = dict(zip(sorted(graph), range(len(graph))))
    graph = nx.relabel_nodes(graph, mapping)

    # print('average degree: %.2lf' % (sum([d for n,d in list(graph.degree())]) / 100))
    # pdb.set_trace()

    # logging some stat
    if plot_only:
        print(nx.info(graph))
        net_param = params['num_edges'] if params['network_type'] == 'Barabasi_Albert' else params['edge_prob']
        avg_degree = np.mean([n[1] for n in list(graph.degree())])
        hist_filename = 'plots/hists/%s_%0.2f_%0.2f.eps' % (params['network_type'], net_param, avg_degree)
        plot_degree_histogram(graph, filename=hist_filename)
    else:
        #TODO: confirm whether this is correct for Lee's method
        if params['network_type'] != 'bipartite':
            nx.set_node_attributes(graph, values='A', name='item_class')

        # Add attributes
        params['seed'] = seed
        att_gen = CausalAttributeGenerator(params)
        g = att_gen.augment(graph, case)

        return g


# def build_graphs(config, case, seed):
#     params = config['params'][0]

#     graphs = []
#     for i in range(config['num_trials']):
#         # Generate network
#         if params['network_type'] == 'Facebook':
#             graph=nx.read_edgelist('data/graphs/facebook_combined.txt',nodetype=int)
#         if (params['network_type']=='Barabasi_Albert'):
#             graph = nx.barabasi_albert_graph(params['num_nodes'], params['num_edges'], seed=seed+i)
#             #print(len(graph.edges()))
#         elif (params['network_type']=='Erdos_Renyi'):
#             graph = nx.erdos_renyi_graph(params['num_nodes'], params['edge_prob'], seed=seed+i)
#             #print(len(graph.edges()))
#         elif (params['network_type']=='Watts_Strogatz'):
#             #print(params['edge_prob'])
#             graph = nx.watts_strogatz_graph(params['num_nodes'], params['knn'], params['edge_prob'],seed=seed+i)
#             #print(len(graph.edges()))
#         #TODO: confirm whether this is correct for Lee's method
#         nx.set_node_attributes(graph, values='A', name='item_class')
        
#         # Add attributes
#         params['seed'] = seed
#         att_gen = CausalAttributeGenerator(params)
#         g = att_gen.augment(graph, case)
#         graphs.append(g)

#     return graphs
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", required=True)
    # parser.add_argument("--case", help="exp case", required=True)
    # parser.add_argument("--seed", type=int, default=12345, help="random seed", required=False)
    args = parser.parse_args()

    exp_name = args.config.split('/')[-1].split('.')[0]
    config = json.load(open(args.config))
    target = config["target"]

    def get_target_parent(g_config, target):
        return g_config["alt_params"] if target in g_config["alt_params"] else g_config["params"]

    def prep_data(g):
        X = np.array([v['trt'] for k,v in dict(g.nodes(data=True)).items() if 'trt' in v])
        Y = np.array([v['out'] for k,v in dict(g.nodes(data=True)).items() if 'out' in v])
        Z = np.array([v['att_0'] for k,v in dict(g.nodes(data=True)).items() if 'att_0' in v])
        A = np.array(nx.adjacency_matrix(g).todense())

        df = pd.DataFrame()
        df['X'] = X
        df['Y'] = Y
        df['Z'] = Z

        return df, A

    is_conditional = config["case"][0] == '2'
    test_type = 'conditional' if is_conditional else 'marginal'
    
    target_parent = get_target_parent(config["graph"], target)
    target_vals = copy.deepcopy(target_parent[target])

    for i in range(len(target_vals)):
        g_config = copy.deepcopy(config["graph"])
        target_parent = get_target_parent(g_config, target)
        g_config["params"]["test_type"] = test_type
        target_value = target_vals[i]   # i'th target value
        target_parent[config["target"]] = target_value

        def dump_data(null=True):
            gc = copy.deepcopy(g_config)
            hypo = "null" if null else "alt"
            gc["params"].update(gc["%s_params" % hypo])
            g = build_graph(gc, config["case"], config["seed"] + trial)
            data, A = prep_data(g)

            data_dir = "data/%s/%s/%0.1f/XYZ/" % (exp_name, hypo, target_value)
            adj_dir = "data/%s/%s/%0.1f/A/" % (exp_name, hypo, target_value)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            if not os.path.exists(adj_dir):
                os.makedirs(adj_dir)
            data_path = os.path.join(data_dir, "trial_%d.csv" % trial)
            adj_path = os.path.join(adj_dir, "trial_%d.csv" % trial)

            data.to_csv(data_path, index=None)
            np.savetxt(adj_path, A.astype(int), delimiter=',')
            print(data_path)

            

        for trial in range(config["num_trials"]):
            dump_data(null=True)
            dump_data(null=False)

            

    # build_graph(config['graph'], config['case'], config['seed'], plot_only=True)



if __name__ == '__main__':
    exit(main())