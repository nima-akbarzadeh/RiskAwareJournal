#!/bin/bash
#SBATCH --mail-user=nima.akbarzadeh@mail.mcgill.ca
#SBATCH --account=def-adulyasa
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=25
#SBATCH --output=~/projects/def-adulyasa/mcnima/SafeWhittleIndex/output.txt
#SBATCH --time=03:00:00

module load python/3.10

source ~/envs/restless_bandits/bin/activate

python main_planning.py
