#!/bin/bash

###Set the processor queue
#PBS -q xeon
###Export all environment variables in the qsub command environment to the batch job environment.
#PBS -V
###Merge error and output files
#PBS -j oe
###Each node has  CPU (Intel 8280)
###Select processor node with 112 logical CPU cores each
#PBS -l select=8:ncpus=112
###Request exclusive placement on the node
#PBS -l place=excl
###Name to appear on the job list
#PBS -N qsub_train

### Load the modules
module load gnu8
module load openmpi3
#source /opt/intel/compilers_and_libraries_2018.0.128/linux/mpi/intel64/bin/mpivars.sh

###CD to the working directory
cd "$PBS_O_WORKDIR"

###Obtain number of cores per socket
export num_core_per_socket=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')

### OPA FABRIC ###
export OMPI_MCA_mtl=psm2
export OMPI_MCA_btl=^tcp,openib
# export I_MPI_FABRICS=shm:tmi
# export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export PSM2_IDENTIFY=1
# export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=$num_core_per_socket
export KMP_AFFINITY=granularity=fine,compact,1,0
export num_proc=16

### Use PBS's RSH instead of SSH
# export I_MPI_HYDRA_BOOTSTRAP=rsh
# export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

### Horovod timeline
export HOROVOD_TIMELINE="${PBS_JOBID}_timeline.json"
export HOROVOD_TIMELINE_MARK_CYCLES=0
export HOROVOD_FUSION_THRESHOLD=134217728

source activate /homes/jmusel/SparseNN_training/env

echo "PBS Job Number      " $(echo $PBS_JOBID | sed 's/\..*//')
echo "PBS batch run on    " $(hostname)
echo "Time it was started " $(date +%F_%T)
echo "Current Directory   " $(pwd)
echo "Submitted work dir  " $PBS_O_WORKDIR
echo "Number of Nodes     " $PBS_NP
echo "Nodefile List       " $PBS_NODEFILE

### Execute on multinode. One process per socket, and bind each process to all the cores on the socket
### See https://stackoverflow.com/questions/28216897/syntax-of-the-map-by-option-in-openmpi-mpirun-v1-8
### mpiexec won't work 
time mpirun -x LD_LIBRARY_PATH \
    -x OMP_NUM_THREADS \
    -x PATH \
    --map-by ppr:1:socket:pe=$num_core_per_socket --report-bindings \
    --oversubscribe -n $num_proc \
    python LeNetExperiment.py --mode train --config experiment_configs/config_LeNet5_prune_cluster4.yaml \
    --load_checkpoint 2 --checkpoint_path experiment_logs/LeNet5_sparsify_cluster4_2_log/ckpt_LeNet5_sparsify_cluster4_epoch39.pth.tar \
    --multiprocessing  | tee output.txt
# time mpirun -x LD_LIBRARY_PATH \
#     -x OMP_NUM_THREADS \
#     -x PATH -x I_MPI_FABRICS \
#     -x I_MPI_TMI_PROVIDER \
#     -x HFI_NO_CPUAFFINITY \
#     -x PSM2_IDENTIFY \
#     -x I_MPI_FALLBACK \
#     -x I_MPI_HYDRA_BOOTSTRAP \
#     -x I_MPI_HYDRA_BOOTSTRAP_EXEC \
#     -x HOROVOD_AUTOTUNE=1 \
#     -x HOROVOD_AUTOTUNE_LOG=autotune_log.csv \
#     --map-by socket:PE=1 --report-bindings \
#     --oversubscribe -n $num_proc python pytorch_mnist.py --no-cuda --epochs=100 --batch-size=1024 2>&1 | tee mnist_train_result.txt
 


