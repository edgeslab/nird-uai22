#!/bin/bash

python plotter.py -res out/exp_1b_dep_ba_mean_poly_type_ii.csv -fmt eps -sres out/exp_1b_dep_ba_mean_poly_type_i.csv
python plotter.py -res out/exp_1b_dep_er_mean_poly_type_ii.csv -fmt eps -sres out/exp_1b_dep_er_mean_poly_type_i.csv
python plotter.py -res out/exp_2b_dep_ba_mean_type_ii.csv -fmt eps -sres out/exp_2b_dep_ba_mean_type_i.csv
python plotter.py -res out/exp_2b_dep_er_mean_type_ii.csv -fmt eps -sres out/exp_2b_dep_er_mean_type_i.csv
python plotter.py -res out/exp_2d_dep_ba_mean_type_ii.csv -fmt eps -sres out/exp_2d_dep_ba_mean_type_i.csv
python plotter.py -res out/exp_2d_dep_er_mean_type_ii.csv -fmt eps -sres out/exp_2d_dep_er_mean_type_i.csv


python plotter.py -res out/exp_1b_ba_mean_type_ii.csv -fmt eps -sres out/exp_1b_ba_mean_type_i.csv
python plotter.py -res out/exp_1b_er_mean_type_ii.csv -fmt eps -sres out/exp_1b_er_mean_type_i.csv
python plotter.py -res out/exp_2b_ba_mean_type_ii.csv -fmt eps -sres out/exp_2b_ba_mean_type_i.csv
python plotter.py -res out/exp_2b_er_mean_type_ii.csv -fmt eps -sres out/exp_2b_er_mean_type_i.csv
python plotter.py -res out/exp_2d_ba_mean_type_ii.csv -fmt eps -sres out/exp_2d_ba_mean_type_i.csv
python plotter.py -res out/exp_2d_er_mean_type_ii.csv -fmt eps -sres out/exp_2d_er_mean_type_i.csv


python plotter.py -res out/exp_1b_ltm_fb_samples_type_ii.csv -fmt eps
python plotter.py -res out/exp_2d_nodes_er_mean_exec_times.csv -fmt eps -sres out/exp_1b_nodes_er_mean_exec_times.csv