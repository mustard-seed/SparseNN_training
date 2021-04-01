#!/bin/bash 

set -e 
export ENV_PREFIX=$PWD/env

conda env create --prefix $ENV_PREFIX --file conda_environment.yml --force
