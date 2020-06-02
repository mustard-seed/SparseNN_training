#!/bin/bash 

set -e 

module load gnu8
module load openmpi3
export ENV_PREFIX=$PWD/env
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_CPU_OPERATIONS=MPI
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1

conda env create --prefix $ENV_PREFIX --file conda_environment.yml --force
