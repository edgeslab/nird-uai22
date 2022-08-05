### Mapping terminologies

First look at a mapping between terminologies presented in the paper and this repo:

| Paper  | Repo    |
|--------|---------|
| NIRD   | MCIT    |
| Case 0 | Case 1b |
| Case 1 | Case 2b |
| Case 2 | Case 2d |

---

### Experiment config
Explanation for different experiment parameters:

```
{
    "case"      : "1b",                 # which case for the test (1a, 1b, 1c, 2a, 2b, 2c ...)
    "seed"      : 12345,                # random seed for synthetic generator
    "algos"     : ["mcit", "krcit"],    # competing algorithms, mcit = NIRD, krcit = Lee's method
    "approx"    : True                  # whether to run the approximte version of mcit or not. Only valid for marginal
    "target"    : "dependence",         # target parameter to vary
    "num_trials": 1,                    # number of trials for each value of target parameter
    "graph" : {                         # graph config
        "params" : {                    # common parameters for both null and alt data 
            "test_type": "marginal",    # marginal or conditional
            "num_nodes": 100,           # number of nodes in the graph
            "num_edges": 2,             # barabasi albert parameter: number of nodes a new node can connect to
            "num_hops": 1,              # number of hops to consider for neighborhood
            "num_covar": 0              # number of variables per node apart from treatment and control
        },
        "null_params" : {               # specific parameters for null data
            "hypothesis": "null",       
            "dependence": 0.0           # this is for marginal case, so no dependence between treatment and outcome
        },
        "alt_params" : {
            "hypothesis": "alt",
            "dependence": [0.1, 0.9]    # In alternate hypothesis, outcome is dependent on treatment
        }
    }
}
```
---

### Log files
- Each experiment produces corresponding log files where results are cached incrementally for each of the target values of the target variable. Considering the name of the above sample config file as _sample.json_ the following log files will be generated:
    ```
    sample_0.csv      # corresponds to the the dependence value 0.1
    sample_1.csv      # corresponds to the the dependence value 0.9
    ```

    _*Note: Remove or backup previous log files before running a fresh experiment. Logs are appended._

---

### Issue with torch and scipy.linalg.eigh

We have experienced an unusual issue where importing torch breaks the library function [scipy.linalg.eigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html). We are not entirely sure whether it is version/platform specific. If you face the same issue, our advice would be to create two virtual environments (venvs) to run the two algos (NIRD, KRCIT) seprately from two different root folders. The venv corresponding to KRCIT should not contain torch and gpytorch installed. Then modify the config files to keep only one algo:

    ```
    "algos"     : ["mcit"]          # for the venv where you run only NIRD 
    "algos"     : ["krcit"]         # for the venv where you run only KRCIT
    ```

A consequence of this would be seperate log files. We have provided a merge script _merge.py_ to merge the log files for two different algos into one. Say, you have two folders like this:

    ```
    sample_mcit
    │   sample_0.csv        #only results for NIRD
    │   sample_1.csv
    sample_krcit
    │   sample_0.csv        #only results for KRCIT
    │   sample_1.csv
    ```

Now run this:

    ```
    python merge.py -r1 sample_mcit -r2 sample_krcit -out sample
    ```

It will produce a new folder with merged logs:

    ```
    sample
    │   │   sample_0.csv    #esults for both NIRD and KRCIT
    │   │   sample_1.csv
    ```

Now move the files inside "logs" directory. Then run the experiment in cached mode:

    ```
    python experiment.py -config conf/sample.json --nop --cached
    ```

It'll load the log files directly and show results, it won't run the experiments again.