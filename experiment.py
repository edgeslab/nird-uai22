import os
import pdb
import copy
import json
import logging
import argparse
import pickle as pkl
from datetime import datetime

import numpy as np
import pandas as pd

import multiprocessing as mp
from joblib import Parallel, delayed

from utils import sample_random_nodes
from synthgen import build_graph
from lib.cits.citf import CITFactory
from lib.cits.ci_eval import EvalCI

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


def init_results(target, algo_names):
    results = {target: []}
    for name in algo_names:
        results[name + '_type_i'] = []
        results[name + '_type_ii'] = []
        results[name + '_aupc'] = []
        results[name + '_times'] = []
    return results


def fill_results(results, algo_names, target, target_vals, rdfs, ):
    for i in range(len(target_vals)):
        target_value = target_vals[i]
        results[target].append(target_value)
        for name in algo_names:
            results[name + '_type_i'].append(rdfs[i].loc[name]['type_i'])
            results[name + '_type_ii'].append(rdfs[i].loc[name]['type_ii'])
            results[name + '_aupc'].append(rdfs[i].loc[name]['aupc'])
            avg_time = (rdfs[i].loc[name]['times_null'] + rdfs[i].loc[name]['times_alt']) / 2
            avg_time = avg_time / 60
            results[name + '_times'].append(avg_time)
    return results


def dump_results(results, out_file):

    def dump_metric(suffix):
        columns = list(results.columns)
        suffix_tail = suffix.split('_')[-1]
        t_cols = list(filter(lambda x: x.split('_')[-1]==suffix_tail in x, columns[1:]))
        t_df = results[[columns[0]] + t_cols]
        t_out_file = out_file.split('.')[0] + '_%s.csv' % suffix
        t_df.to_csv(t_out_file, index=False)

    dump_metric('exec_times')
    dump_metric('type_i')
    dump_metric('type_ii')
    dump_metric('aupc')


def get_target_parent(g_config, target):
    return g_config["alt_params"] if target in g_config["alt_params"] else g_config["params"]


def run_target_task(i, config, alpha, log_file, load_cache=False):
    '''
        Run experiment for i'th target value in config.
        Return evaluation result as a dataframe
    '''
    # parse config
    g_config = copy.deepcopy(config["graph"])

    is_conditional = config["case"][0] == '2'
    test_type = 'conditional' if is_conditional else 'marginal'
    g_config["params"]["test_type"] = test_type

    # init algo objs
    algos = []
    for name in config["algos"]:
        algos.append((name, CITFactory.get_cit(name, config["seed"])))

    # set target value for this taks
    target_parent = get_target_parent(g_config, config["target"])
    target_vals = copy.deepcopy(target_parent[config["target"]])
    target_value = target_vals[i]   # i'th target value
    target_parent[config["target"]] = target_value

    # init evaluator
    evaluator = EvalCI(a=alpha, names=config["algos"])
    evaluator.initialize()

    approximate = config["approx"] if "approx" in config else False

    def itest(null=False):
        gc = copy.deepcopy(g_config)
        params = gc["null_params"] if null else gc["alt_params"]
        gc["params"].update(params)
        g = build_graph(gc, config["case"], config["seed"] + trial)
        #print("graph built")

        samples = []
        if config["target"] == 'sample_size':
            _, samples = sample_random_nodes(g, target_value)

        for algo in algos:
            name, cit = algo

            start = datetime.now()
            p_val = cit.run_test(g=g, cond=is_conditional, rmask=rmasks[config["case"]], samples=list(samples), approx=approximate)
            elapsed = (datetime.now() - start).seconds

            evaluator.add_result(name, p_val, null=null, time=elapsed)

    log_file = '{prefix}_{index}.csv'.format(prefix=log_file.split('.')[0], index=i)

    if load_cache:
        evaluator.load_p_vals(log_file)
    else:
        # run trials
        for trial in range(config["num_trials"]):
            itest(null=True)
            itest(null=False)

            print('%s:: trial %d task %d done' % (datetime.now(), trial, i))
            evaluator.log_p_vals(log_file)

    rdf = evaluator.gen_result()
    # print(rdf)
    return (i, rdf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default='conf/exp_1a_dep.json', help="config", required=True)
    parser.add_argument("-s", type=int, default=-1, help="seed", required=False)
    parser.add_argument("-a", type=float, default=0.05, help="significance level: alpha", required=False)
    parser.add_argument("-d", type=int, default=0, help="debug mode (0/1)", required=False)
    parser.add_argument("-nn", type=int, default=-1, help="number of nodes", required=False)
    parser.add_argument("-nt", type=int, default=-1, help="number of trials", required=False)
    parser.add_argument("-jobs", type=int, default=-1, help="number of parallel jobs", required=False)
    parser.add_argument("--cache", action='store_true', help="load p-vals from cache?", required=False)
    parser.add_argument("--nop", action='store_true', help="don't run parallel?", required=False)
    args = parser.parse_args()
    
    # logging.basicConfig(filename='logs/experiment.log', mode='a', format='%(asctime)-15s :: %(message)s')
    logger.propagate = False
    logger.setLevel(logging.DEBUG if args.d else logging.INFO)
    fh = logging.FileHandler('logs/experiment.log', mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # parse config, override if necessary
    config = json.loads(open(args.config, 'r').read())
    config["seed"] = args.s if args.s != -1 else config["seed"]
    config["num_trials"] = args.nt if args.nt != -1 else config["num_trials"]
    if "num_nodes" in config["graph"]:
        config["graph"]["num_nodes"] = args.nn if args.nn != -1 else config["graph"]["num_nodes"]
    target = config["target"]

    exp_name = args.config.split('/')[-1].split('.')[0]
    config['exp_name'] = exp_name

    # init results dict
    results = init_results(target, config["algos"])

    # get target values
    target_parent = get_target_parent(config["graph"], target)
    target_vals = copy.deepcopy(target_parent[target])

    # run target task
    log_file = 'logs/{prefix}.csv'.format(prefix=exp_name)

    logger.info('{name} started'.format(name=exp_name))
    if args.nop:
        rdfs = [run_target_task(i, config, args.a, log_file, args.cache) for i in range(len(target_vals))]
    else:
        num_jobs = args.jobs if args.jobs != -1 else min(mp.cpu_count(), len(target_vals))
        rdfs = Parallel(n_jobs=num_jobs)(delayed(run_target_task)(i, config, args.a, log_file, args.cache) for i in range(len(target_vals)))

    # prepare results
    rdfs = [res[1] for res in sorted(rdfs, key=lambda x: x[0])]
    results = fill_results(results, config["algos"], target, target_vals, rdfs)
    results = pd.DataFrame(data=results)
    print(results)

    # store results
    out_file = 'out/' + args.config.split('/')[1].split('.')[0] + '.csv'
    dump_results(results, out_file)
    logger.info('{name} finished'.format(name=exp_name))
    



if __name__ == '__main__':
    main()