import os
import sys
import pdb
import argparse
import collections

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


xmap = {
    'dependence'    :   'Dependence coefficient',
    'num_nodes'     :   'Number of nodes',
    'num_edges'     :   'Edge connectivity',
    'edge_prob'     :   'Edge probability',
    'rewire_prob'   :   'Rewiring probability',
    'knn'           :   'Neareset neighbors (k)',
    'sample_size'   :   'Sample size',
    'step_size'     :   'Step size',
    'alpha'         :   'Alpha (= Beta)',
    'avg_degree'    :   'Avg Degree',
    'noise_scale'   :   'Noise Scale',
    'dummy'         :   'dummy'
}

ymap = {
    'type_i'    :   'Type-I Error',
    'type_ii'   :   'Type-II Error',
    'aupc'      :   'AUPC',
    'p_val'     :   'Mean P-Value',
    'exec_times':   'Execution time (minutes)',
    'dummy_dummy':  'dummy'
}

lmap = {
    'mcit'      :   'NIRD',
    'mcit-0'    :   'NIRD-Case 0',
    'mcit-2'    :   'NIRD-Case 2',
    'mcit-a'    :   'NIRD-A',
    'krcit'     :   'KRCIT',
    'krcit-0'   :   'KRCIT-Case 0',
    'krcit-2'   :   'KRCIT-Case 2',
    'mcit-d0'   :   'Step 0',
    'mcit-d1'   :   'Step 1',
    'mcit-d3'   :   'Step 3',
    'mcit-d5'   :   'Step 5',
    'mcit-d10'  :   'Step 10',
    'mcit-d15'  :   'Step 15',
    'mcit-d20'  :   'Step 20',
    'mcit-e3'   :   'AP 1E-3',
    'mcit-e4'   :   'AP 1E-4',
    'mcit-5e4'  :   'AP 5E-4',
    'mcit-3e4'  :   'AP 3E-4',
    'mean'      :   'Mean Aggregate',
    'sum'       :   'Sum Aggregate',
    'linear'    :   'Linear dependency',
    'poly'      :   'Polynomial dependency',
    'naive'     :   'Naive',
    'sic'       :   'SIC'
}


def plot_init(fsize, xlabel, ylabel):
    fig = plt.figure(figsize=(16,10))
    plt.rc('legend', fontsize=fsize)
    plt.rc('xtick',labelsize=fsize)
    plt.rc('ytick',labelsize=fsize)
    plt.rcParams["font.family"] = "Times New Roman"

    plt.xlabel(xlabel, fontsize=fsize+5)
    plt.ylabel(ylabel, fontsize=fsize+5)

    return fig
    

def plot_degree_histogram(G, filename):
    plt.figure(figsize=(16,12))
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram", fontsize=16)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Degree", fontsize=12)
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg, rotation=-45)

    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(G)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)

    # plt.show()
    plt.savefig(filename, format=filename.split('.')[-1])


def draw_multi_y_column(df, num_plots, labels, xlabel, ylabel, filename, fmt='eps', fontsize=64, shadow_df=None):
    columns = list(df.columns)
    
    xcol = columns[0]
    ycols = columns[1:]

    fig = plot_init(fsize=fontsize, xlabel=xlabel, ylabel=ylabel)
    
    legend_handles = []
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ["o", "^", "s", "P", "D", ">"]
    # colors = ['blue', 'green', 'gold', 'red', 'purple', 'magenta']
    colors = ['blue', 'green', 'purple', 'red', 'gold', 'magenta']
    ls = 0
    for i in range(num_plots):
        # df[xcols[i]] = df[xcols[i]] * 60
        line, = plt.plot(xcol, ycols[i], data=df, linewidth=3, linestyle=linestyles[ls], color=colors[ls], marker=markers[ls], markersize=16)
        legend_handles.append(line)

        if shadow_df is not None:
            line, = plt.plot(xcol, ycols[i].replace('ii', 'i'), data=shadow_df, linewidth=3, linestyle='dashed', color=colors[ls], marker=markers[ls], markersize=16)
            legend_handles.append(line)

        ls += 1

    axes = plt.gca()
    legend_loc = 'upper right'
    if 'Type-I' in ylabel:
        axes.set_ylim([-0.05, 1.05])
    elif 'Type-II' in ylabel:
        axes.set_ylim([-0.05, 1.05])
    elif 'AUPC' in ylabel:
        axes.set_ylim([-0.05, 1.05])
        legend_loc = 'lower right'
    else:
        legend_loc = 'upper left'
        axes.set_ylim([-5, 385])

    if 'ltm' in filename:
        axes.set_ylim([0.00, 1.00])

    axes.set_xticks(df[xcol])
    # axes.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # # axes.set_yticklabels([])
    # axes.tick_params(which='major', length=14, width=4, direction='inout')

    # plt.legend(handles=legend_handles, labels=labels, loc=legend_loc, prop={'size': 32}, ncol=2)
    if 'time' in filename:
        plt.legend(handles=legend_handles, labels=labels, prop={'size': fontsize-10}, ncol=1, loc='upper left', fancybox=True, framealpha=0.5)
    elif 'ltm' in filename:
        # pltlegend = plt.legend(handles=legend_handles, bbox_to_anchor=(0.50, 1.11), labels=labels, prop={'size': fontsize-5}, ncol=4, loc='upper center')
        pltlegend = plt.legend(handles=legend_handles, labels=labels, prop={'size': fontsize-10}, ncol=2, loc='upper right')

    if fmt == 'eps':
        plt.savefig(filename, format='eps', dpi=2000, bbox_inches='tight')
    else:
        plt.savefig(filename, format=fmt, bbox_inches='tight')

    if ('time' not in filename) and ('ltm' not in filename):
        figlegend = plt.figure(figsize=(36, 2.6))
        figlegend.legend(handles=legend_handles, labels=labels, bbox_to_anchor=(0.85, 1.0), prop={'size': 58}, ncol=3, loc='upper right')
        figlegend.savefig('plots/fig_1_2_legend.eps', dpi=2000, format='eps')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', default='15_pass_eta1_combinedw_arrr.csv', help='combined result file.')
    parser.add_argument('-sres', default='', help='shadow result file.')
    parser.add_argument('-fmt', default='eps', help='image format.')
    parser.add_argument('--t1', action='store_true', help='put dashed type-i error line.')
    # parser.add_argument('-xlabel', default='Dependence coefficient', help='x-label for plot.')
    # parser.add_argument('-out', default='', help='output image filename.')
    parser.add_argument('--all', action='store_true', help='generate plots for all out files.')
    args = parser.parse_args()

    def draw_plot(result_file, shadow_file=''):
        results = pd.read_csv(result_file)
        columns = list(results.columns)

        xlabel = xmap[columns[0]]
        for k in ymap:
            if result_file.split('.')[0].endswith(k):
                error_type = k
                break
            # error_type = '_'.join(result_file.split('.')[0].split('_')[-2:])
        ylabel = ymap[error_type]

        out_file = 'plots/' + result_file.split('/')[1].split('.')[0] + '.' + args.fmt
        labels = list(map(lambda x: lmap[x.split('_')[0]], columns[1:]))

        shadow_result = None
        if shadow_file != '':
            shadow_result = pd.read_csv(shadow_file)
            if result_file.split('.')[0].split('_')[-1] == 'times':
                labels = []
                for c in columns[1:]:
                    labels.append(lmap[c.split('_')[0]] + '-Case 2')
                    labels.append(lmap[c.split('_')[0]] + '-Case 0')

            if result_file.split('.')[0].split('_')[-1] == 'ii':
                # shadow_result = pd.read_csv(result_file.replace('_ii.', '_i.'))
                labels = []
                for c in columns[1:]:
                    labels.append(lmap[c.split('_')[0]] + '-Type II')
                    labels.append(lmap[c.split('_')[0]] + '-Type I')
                ylabel = 'Type-I/II Error'


        draw_multi_y_column(results, results.shape[1]-1, labels, xlabel, ylabel, out_file, fmt=args.fmt, shadow_df=shadow_result)


    if args.all:
        if not os.path.isdir(args.res):
            print("ERROR: -res is not a directory!")
            sys.exit(1)
        for path, _, files in os.walk(args.res):
            for file in files:
                result_file = os.path.join(path, file)
                if result_file.split('.')[-1] != 'csv':
                    continue
                # if 'type' not in result_file:   # TODO: handle times plot
                #     continue
                draw_plot(result_file)
            break
    else:
        if not os.path.isfile(args.res):
            print("ERROR: -res is not a file!")
            sys.exit(1)
        draw_plot(args.res, args.sres)


if __name__ == "__main__":
    main()