#!/bin/bash
#SBATCH --job-name=mytest
#SBATCH --output=mytest.out
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d336li@uwaterloo.ca

/home/d336li/opt/julia-1.11/bin/julia /home/d336li/codes/mytest.jl
