#!/bin/bash
# Job Name
#SBATCH -J or
# Walltime requested
#SBATCH -t 16:00:00
# Provide index values (TASK IDs)
#SBATCH --array=0-35
#SBATCH --account=carney-ashenhav-condo
# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
##SBATCH -e job_files/arrayjob-%J-%a.err
#SBATCH -o job_files/arrayjob-%A-%a.out
# Controls the minimum/maximum number of nodes allocated to the job
#SBATCH -N 1
# single core
#SBATCH -c 2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xiamin_leng@brown.edu
# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
module load miniforge
source /users/xleng/.bashrc
conda deactivate
conda deactivate
conda deactivate
conda activate lan_pipe2
echo "Running job array number 1: "$SLURM_ARRAY_TASK_ID
python runModel_p.py $SLURM_ARRAY_TASK_ID
