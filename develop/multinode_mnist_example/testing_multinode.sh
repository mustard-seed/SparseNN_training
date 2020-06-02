#!/bin/bash
#PBS -l select=4:ncpus=272 -lplace=excl

source /opt/intel/compilers_and_libraries_2017.1.132/linux/mpi/intel64/bin/mpivars.sh intel64

### OPA FABRIC ###
export I_MPI_FABRICS=shm:tmi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export PSM2_IDENTIFY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=66

# Use PBS's RSH instead of SSH
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

#IFS=' ' read -r -a nodenames <<< $PBS_NODEFILE
readarray -t nodenames < $PBS_NODEFILE
nodeconfig=~/nodeconfig_8.txt
rm -f ~/nodeconfig_8.txt
cnt=1
for n in "${nodenames[@]}"
do
        echo "-host ${n} -n 1 numactl -p 1 /export/software/caffe/build/tools/caffe train -solver /export/software/caffe/8_nodes_images/solver/solver_images_${cnt}.prototxt -engine "MKL2017" --param_server=mpi " >> $nodeconfig

cnt=$((cnt+1))
done

time mpirun -configfile $nodeconfig 2>&1 | tee ~/8_node_test.txt
