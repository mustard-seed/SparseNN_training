#!/bin/bash 

set -e 
export ENV_PREFIX=$PWD/env
export HOROVOD_WITH_MPI=1
export HOROVOD_CPU_OPERATIONS=MPI
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_OPERATIONS=NCCL

conda env create --prefix $ENV_PREFIX --file conda_environment.yml --force
