#!/bin/bash  -v
#SBATCH --partition=SP3
#SBATCH --ntasks=1 		# number of tasks / mpi processes
#SBATCH --cpus-per-task=1 	# Number OpenMP Threads per process
#SBATCH -J aloca-1-cpu
#SBATCH --time=01:02:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo $SLURM_JOB_ID		#ID of job allocation
echo $SLURM_SUBMIT_DIR		#Directory job where was submitted
echo $SLURM_JOB_NODELIST	#File containing allocated hostnames
echo $SLURM_NTASKS		#Total number of cores for job

#module swap gnu intel/18.0.2.199
module load numpy, pandas, sklearn

#run the application:
srun   nohup nice python3 mainSplitDataFrame.py &