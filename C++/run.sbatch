#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=output.txt

export
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./a.out