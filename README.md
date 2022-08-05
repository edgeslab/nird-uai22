# Non-Parametric Inference of Relational Dependence (NIRD)

----------

### Requirements

Prepare Ubuntu environment:

```
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install zip
```

This project requires Python 3. Follow the instructions to install all the dependencies.

Install [Anaconda](https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh).

a. Create a conda venv:

```
conda create --name rci python=3.6 --yes
```

b. Activate the venv:

```
conda activate rci
```

If it doesn't work, try:

```
source activate rci
```

c. Make sure you are using the python inside the venv by:

```
which python
```

d. **Note for MacOS**: Make sure _**wget**_ is installed in the system. You can install it by [brew](https://brew.sh/).

e. Run the following command which automates the download and installation of dependencies.

```
	sh ./install.sh
```

f. Install _torch_ and [gpytorch](https://gpytorch.ai/) (only needed for NIRD conditional test):

```
	pip install torch
	pip install --upgrade git+https://github.com/cornellius-gp/gpytorch.git
```

	

### Sample Usage

- Run the sample script just to check everything is working fine:

```
	./sample.sh
```

- It'll take less than a minute to finish. It'll create a zip called sample.zip containing the outputs, logs and plots created for the sample run.


### Reproduce reported results

- Run the following script to reproduce the reported results:

	```
	sh ./neurips21.sh 100
	```


### Usage for custom experiments

- Enter your preferred configuration for experiment in a config json file inside _conf_ directory

    - _conf/exp_1a_dep.json_ is a config file for case 1a. It includes both synthetic graph parameters as well as experiment parameters. Refer to _experiments.md_ for detailed explanation of experiment configs.

- Then run the experiment using the following command:
	```
	python experiment.py -config conf/exp_1a_dep.json -o out/exp_1a_dep.csv 
	```
	**-config**: Input config 
	
	**-o**: Output csv path. By default it will be stored in _out_ directory.
	
	Following files will be created:
	
	- out/exp_1a_dep_type_i.csv
	
	- out/exp_1a_dep_type_ii.csv
	
- There are several other arguments you can use:

	```
	python experiment.py -config conf/sample.json -nt 2             # override number of trials
	python experiment.py -config conf/sample.json -jobs 2           # specifiy job for parallel
	python experiment.py -config conf/sample.json --nop             # no parallel run
	python experiment.py -config conf/sample.json -a 0.01 --cache   # specify alpha, get results from logged p-values, no exp done
	```
    
- To plot results run the following:

	```
	python plotter.py -fmt png -res out/exp_1a_dep_type_ii.csv -sres out/exp_1a_dep_type_i.csv
	```
	
	**-res**: Input results csv file (type ii)

	**-sres**: Input results csv file (type i)
	
	**-fmt**: Output image format (png / eps)
