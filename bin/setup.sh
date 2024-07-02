#!/bin/bash
# This script sets up the Mamba environment for the project.
# Define what python version to use
# run source bin/setup.sh

enviro=$(basename "$PWD")

mamba create --name $enviro python==3.8.19 -y

# setup install conda in specific environment
mamba install -n ai-docs --file ../conf/c-requirements.txt -y

echo " ------------------------------------   "
echo " mamba activate "$enviro
echo " -------------------------------------  "

# setup install pip in specific environment, enviroment must be activated before this setp !
echo " pip install -r conf/p-requirements.txt "

