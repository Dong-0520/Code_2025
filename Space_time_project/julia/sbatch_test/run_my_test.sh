#!/bin/bash
#SBATCH --mail-user=d336li@uwaterloo.ca
#SBATCH --mail-type=end,fail
#SBATCH --partition=cpu_pr3
#SBATCH --job-name="sbatch test 01"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=stdout-%j.log
#SBATCH --error=stderr-%j.log

/home/d336li/opt/julia-1.11/bin/julia /home/d336li/codes/mytest.jl
