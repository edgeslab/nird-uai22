#!/bin/bash

# arg 1: number of trials, default: 100
# arg 2: alpha, default 0.05


seed=12345
trials=100
alpha=0.05


if [[ $# -gt 0 ]] ; then
    trials=$1
fi

if [[ $# -gt 1 ]] ; then
    alpha=$2
fi

mkdir -p out/
mkdir -p out/demo
mkdir -p logs/
mkdir -p plots/

if [ -f "logs/experiment.log" ]; then
   mv "logs/experiment.log" "logs/experiment.log_back"
fi


echo "NIRD experiments starting for [trials: "$trials", alpha: "$alpha"]"

STARTTIME=$(date +%s)
MINUTES=60


timed_exp () {
    STARTTIME=$(date +%s)
    for t in $(seq 0 $(( $trials-1 )))
    do
        t_seed=$(( $seed + t ))
        python experiment.py -s $t_seed -nt 1 -a $alpha -config conf/$1.json -jobs $2
    done
    python experiment.py -nt $trials -config conf/$1.json --nop --cache

    ENDTIME=$(date +%s)
    elapsed=`echo "("$ENDTIME - $STARTTIME")/"$MINUTES | bc -l | xargs printf %.2f`
    eval 'push "$1 experiment finished in ['$elapsed'] minutes" > /dev/null'
}



# -------------------------------------- Batch 1 ------------------------------------------

# "dependence coefficient tests: synthetic"

timed_exp "exp_1b_dep_ba_mean_poly" 1
timed_exp "exp_1b_dep_er_mean_poly" 1
timed_exp "exp_1b_ba_mean" 1
timed_exp "exp_1b_er_mean" 1

# # -------------------------------------- Batch 2 ------------------------------------------

timed_exp "exp_2b_dep_ba_mean" 5
timed_exp "exp_2b_dep_er_mean" 5
timed_exp "exp_2d_dep_ba_mean" 5
timed_exp "exp_2d_dep_er_mean" 5

# # -------------------------------------- Batch 3 ------------------------------------------

timed_exp "exp_2b_ba_mean" 3
timed_exp "exp_2b_er_mean" 5
timed_exp "exp_2d_ba_mean" 3
timed_exp "exp_2b_er_mean" 5

# -----------------------------------------------------------------------------------------



# -------------------------------------- Plots ---------------------------------------------
# echo "plot all the results"
# echo "===================="

# python plotter.py -res out -fmt eps --all



# echo "zip all results"
# echo "==============="

# zip -r results@words.zip out logs plots

# -------------------------------------------------------------------------------