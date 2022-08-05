#!/bin/bash

mkdir -p out/
mkdir -p logs/
mkdir -p plots/

python experiment.py -config conf/sample.json                   # default: parallel run, alpha 0.05, logging p-values
python experiment.py -config conf/sample.json -nt 2             # override number of trials
python experiment.py -config conf/sample.json -jobs 2           # specifiy job for parallel
python experiment.py -config conf/sample.json --nop             # no parallel run
python experiment.py -config conf/sample.json -a 0.01 --cache   # specify alpha, gen results from logged p-values, no exp done

zip -r sample.zip out logs plots