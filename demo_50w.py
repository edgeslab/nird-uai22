import os
import pdb
import copy
import json
import random
import logging
import argparse
import itertools
import pickle as pkl
from datetime import datetime

import numpy as np
import pandas as pd

import networkx as nx
import multiprocessing as mp
from joblib import Parallel, delayed

from lib.cits.citf import CITFactory
from utils import make_dir, print_graph_stat
from utils import sample_random_nodes, sample_random_edges

logger = logging.getLogger(__file__)


rmasks = {
    '1a'    :   '000',
    '1b'    :   '100',
    '1c'    :   '110',
    '2a'    :   '000',
    '2b'    :   '100',
    '2c'    :   '110',
    '2d'    :   '001',
    '2e'    :   '101',
    '2f'    :   '111',
}


sampling_map = {
    "random_edges"  :   sample_random_edges,
    "random_nodes"  :   sample_random_nodes
}

def sample_subgraph(g, sample_size, sampling_method):
    # TODO: Needs to do a proper graph sampling here (i.e. snowball sampling )

    sample_g, seed_samples = sampling_map[sampling_method](g, sample_size)

    remove_samples = []
    for n in seed_samples:
        if sample_g.nodes()[n]['trt'] == 1:
            remove_samples.append(n)
    for n in remove_samples:
        seed_samples.remove(n)
    
    sample_g.remove_nodes_from(remove_samples)
    sample_g.remove_nodes_from(list(nx.isolates(sample_g)))

    mapping = dict(zip(sample_g, range(len(sample_g))))
    seed_samples = [mapping[n] for n in seed_samples if n in sample_g]
    sample_g = nx.relabel_nodes(sample_g, mapping)

    print('sampled non-hateful nodes: %d' % len(seed_samples))
    print_graph_stat(sample_g)

    outs = 0
    for n in seed_samples:
        assert(sample_g.nodes()[n]['trt'] == 0)
        if sample_g.nodes()[n]['out'] == 1:
            outs += 1

    print('sampled hateful outcomes: %d' % outs)
    # pdb.set_trace()

    return sample_g, seed_samples



def build_graph(graph_path, att_path):
    with open(graph_path, 'rb') as edgefile:
        graph=nx.read_edgelist(edgefile,nodetype=int)

    with open(att_path, 'rb') as attfile:
        attributes = pkl.load(attfile)

    graph.remove_nodes_from(list(nx.isolates(graph)))

    treatment={}
    outcome={} 
    for i in attributes:
        treatment[i]=attributes[i][0]
        outcome[i]=attributes[i][1]

    nx.set_node_attributes(graph, values='A', name='item_class')
    nx.set_node_attributes(graph, values=treatment, name='trt')
    nx.set_node_attributes(graph, values=outcome, name='out')

    mapping = dict(zip(graph, range(len(graph))))
    graph = nx.relabel_nodes(graph, mapping)

    # print_graph_stat(graph)

    return graph


def run_target_task(i, config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    # parse config
    g_config = copy.deepcopy(config["graph"])

    # init algo objs
    algos = []
    for name in config["algos"]:
        algos.append((name, CITFactory.get_cit(name, config["seed"])))

    g = build_graph(g_config["graph_path"], g_config["att_path"])
    t0 = [k for k,v in nx.get_node_attributes(g, name='trt').items() if v == 0]

    trial_p_vals = {}
    for name in config["algos"]:
        trial_p_vals['%s_all' % name] = -1
        trial_p_vals['%s_t0' % name] = -1
    trial_p_vals['t0'] = len(t0)

    for algo in algos:
        name, cit = algo
        
        p_val_all = cit.run_test(g=g, cond=False, rmask=rmasks[config["case"]], samples=[])
        if t0:
            p_val_ns = cit.run_test(g=g, cond=False, rmask=rmasks[config["case"]], samples=t0)
        else:
            p_val_ns = -1
        
        trial_p_vals['%s_all' % name] = p_val_all
        trial_p_vals['%s_t0' % name] = p_val_ns

    return (i, trial_p_vals)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default='conf/exp_1a_dep.json', help="config", required=True)
    parser.add_argument("-a", type=float, default=0.05, help="significance level: alpha", required=False)
    parser.add_argument("-d", type=int, default=0, help="debug mode (0/1)", required=False)
    parser.add_argument("-l", type=str, default='out/50women.tex', help="tex table output path", required=False)
    parser.add_argument("--cache", action='store_true', help="load p-vals from cache?", required=False)
    parser.add_argument("--nop", action='store_false', help="don't run parallel?", required=False)
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.d else logging.INFO)

    config = json.loads(open(args.config, 'r').read())

    periods = config["graph"]["periods"]
    atts = config["graph"]["attributes"]
    att_types = config["graph"]["att_types"]

    all_configs = []
    for p, a, t in itertools.product(periods, atts, att_types):
        conf = copy.deepcopy(config)
        conf["graph"]["period"] = p
        conf["graph"]["attribute"] = a
        conf["graph"]["att_type"] = t
        conf["graph"]["graph_path"] = "data/graphs/50women/network{period}.data".format(period=p)
        conf["graph"]["att_path"] = "data/graphs/50women/{att}_{att_type}_period{period}.data".format(att=a, att_type=t, period=p)
        all_configs.append(conf)

    if args.nop:
        num_jobs = min(mp.cpu_count(), len(all_configs))
        results = Parallel(n_jobs=num_jobs)(delayed(run_target_task)(i, all_configs[i]) for i in range(len(all_configs)))
    else:
        results = [run_target_task(i, all_configs[i]) for i in range(len(all_configs))]

    results = sorted(results, key=lambda x: x[0])

    data = {"period": [], "attribute": [], "att_type": [], "t0": []}
    for algo in config["algos"]:
        data['%s_all' % algo] = []
        data['%s_t0' % algo] = []

    for i, res in results:
        data["period"].append(all_configs[i]["graph"]["period"])
        data["attribute"].append(all_configs[i]["graph"]["attribute"])
        data["att_type"].append(all_configs[i]["graph"]["att_type"])
        for algo in config["algos"]:
            data['%s_all' % algo].append(res['%s_all' % algo])
            data['%s_t0' % algo].append(res['%s_t0' % algo])
        data['t0'].append(res['t0'])

    rdf = pd.DataFrame(data)
    rdf = rdf.sort_values(by=['period', 'attribute', 'att_type'])
    rdf.loc[rdf['t0'] == 0, 'mcit_t0'] = 'N/A'
    rdf.loc[rdf['t0'] == 0, 't0'] = 'N/A'
    print(rdf)

    make_dir('out/demo/')
    out_file = 'out/demo/' + args.config.split('/')[1].split('.')[0] + '_p_val.csv'
    rdf.to_csv(out_file, index=False)

    if args.l:
        rdf.to_latex(args.l, index=False)


if __name__ == '__main__':
    main()