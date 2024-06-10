#!/bin/bash
# This script sets up the Conda environment for the project.
#
APP_PATH=$(dirname "$0")

# get enviroment name from the conda_config.yaml file
enviro=$(basename "$PWD")

# Check if the environment exists
if ! conda env list | grep -q "$enviro"; then
    echo "Creating a new conda environment named $enviro..."
    # Create a new conda environment named "$enviro" using the configuration file "conda_config.yaml"
    conda create --name $enviro --file requirements.txt -y
fi

echo " ------------------------------ "
echo "conda activate "$enviro
echo "pip install -r requirements.txt"
echo " ------------------------------ "
echo "To remove a conda environment, use the following command:"
echo "conda env remove --name myenv"
echo "or rm -rf /path/to/anaconda3/envs/myenv"