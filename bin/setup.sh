#!/bin/bash
# This script sets up the Mamba environment for the project.
# Define what python version to use

PROPATH="/home/rafael/dev/projects"
enviro=$(basename "$PWD")
APPPATH=$PROPATH/$enviro

mamba create --name $enviro python==3.9.19 -y

echo " ------------------------------------   "
echo " mamba activate "$enviro
echo " -------------------------------------  "

# mamba install -n $enviro -c conda-forge --file $APPPATH/conf/conda-requirements.txt -y
# or
# mamba install -n $enviro -c conda-forge --file $APPPATH/conf/conda-env.yaml -y
# and
# pip install -r $APPPATH/conf/pip-requirements.txt 


