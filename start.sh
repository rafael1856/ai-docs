#!/bin/bash
#

#https://stackoverflow.com/questions/4332478/read-the-current-text-color-in-a-xterm/4332530#4332530
NORMAL=$(tput sgr0)
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)

# expected environment name
env_name=$(basename "$PWD")

CPATH="/home/rafael/miniforge-pypy3/bin"
# Get the name of the current conda environment
current_env=$($CPATH/conda env list | grep '*' | awk '{print $1}')


# Check if the environments match
if [ "$env_name" != "$current_env" ]; then
    printf "\n *** ERROR ***\n"
    printf "\nThe current conda environment is not the expected environment."
    printf "\n\n%40s" "Activate the enviroment running: ${RED}conda activate $env_name ${NORMAL}"
    printf "\n\n%40s\n" "If you don't have the environment, create it by running: ${RED}source bin/setup.sh ${NORMAL}"
  exit 1
fi

# clean old logs
rm logs/*.log > /dev/null 2>&1

# Check if the parameter is --doc or --url
if [ "$1" = "--doc" ] || [ "$1" = "--url" ]; then
    arg_for_python="$1 $2"
else
    printf " \n *** ERROR ***"
    printf " \n Invalid parameter. \n Please use    --doc name-of-the-doc \n or \n --url http://your_site/your_doc\n"
    printf " for example ./start.sh --doc path_to_my_doc/my_doc_name\n"
    printf " \nThe doc will be copied to the local 'data' folder for processing, if the doc does not exist in the data folder\n\n"
    exit 1
fi

# Stop the docker container
src/stop-docker.sh

sleep 5

# Start the docker containerfor the vector database
printf "\n *** starting docker for vector database  ***"

export KMP_DEVICE_THREAD_LIMIT=4
src/run-docker.sh

python src/main.py $arg_for_python


# OMP: Warning #96: Cannot form a team with 12 threads, using 1 instead.
# OMP: Hint Consider unsetting KMP_DEVICE_THREAD_LIMIT (KMP_ALL_THREADS), KMP_TEAMS_THREAD_LIMIT, and OMP_THREAD_LIMIT (if any are set).

