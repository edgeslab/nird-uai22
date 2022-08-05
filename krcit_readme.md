# Install KRCIT Dependencies

- Clone the [KRCIT repo](https://github.com/sanghack81/KRCIT): 
    ```git clone https://github.com/sanghack81/KRCIT.git```
    
- Follow the instructions provided in [readme](https://github.com/sanghack81/KRCIT/blob/master/README.md)

- Keep the virtual environment activated

- Install gpflow in the activated conda venv:
    
    ```pip install gpflow```
    
- In order to avoid Deprecation Warning reinstall specific versions of Numpy and Seaborn
    - Numpy 1.16

    ```pip uninstall numpy```
    
    ```pip install numpy==1.16```
    
    - Seaborn 0.8.1

    ```pip uninstall seaborn```
    
    ```pip install seaborn==0.8.1```
    

# Install KRCIT 

- Go to KRCIT repor directory

    ```cd KRCIT```

- Install requirements

    ```conda install --yes --file requirements.txt```

- Install KRCIT package

    ```python setup.py install```



# Use KRCIT

- Go to the experiments folder of KRCIT repo: 

    ```cd uai2017experiments```

- Run simple experiments:

    ```python run_simple_experiments.py```
