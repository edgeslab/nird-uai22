#!/bin/bash
# conda create --name rci python=3.6 --yes
# conda activate rci

conda install six numpy wheel scipy matplotlib pandas  --yes 
conda install tensorflow-base==1.12.0=eigen_py36h4dcebc2_0  # https://github.com/tensorflow/tensorflow/issues/24172
# conda install tensorflow=1.13 --yes
pip install gpflow==1.3

cwd=`pwd`
cd ~/Downloads
git clone https://github.com/sanghack81/pyGK
git clone https://github.com/sanghack81/SDCIT
git clone https://github.com/sanghack81/pyRCDs
git clone https://github.com/sanghack81/KRCIT

cd SDCIT
conda install --yes --file requirements.txt
./setup.sh
python setup.py install
cd ../pyGK
conda install --yes --file requirements.txt
python setup.py install
cd ../pyRCDs
conda install --yes --file requirements.txt
python setup.py install
cd ../KRCIT
conda install --yes --file requirements.txt
python setup.py install
cd ..
rm -rf SDCIT pyGK pyRCDs KRCIT

yes | pip uninstall numpy
pip install numpy==1.16

yes | pip uninstall seaborn
pip install seaborn==0.8.1

pip install ghalton
pip install torch
pip install --upgrade git+https://github.com/cornellius-gp/gpytorch.git

# conda deactivate
cd $cwd
