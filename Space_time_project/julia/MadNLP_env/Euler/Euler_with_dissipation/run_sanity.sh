#!/bin/bash
#SBATCH --job-name=sanity_ma57
#SBATCH --chdir=/work/d336li/Code_2025/Space_time_project/julia/sbatch_test
#SBATCH --output=/work/d336li/Code_2025/Space_time_project/julia/MadNLP_env/Euler/Euler_with_dissipation/logs/mytest.%j.out
#SBATCH --error=/work/d336li/Code_2025/Space_time_project/julia/MadNLP_env/Euler/Euler_with_dissipation/logs/mytest.%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --partition=cpu_pr3           # ← 改成你查到的CPU分区名，例如 cpu_pr2
#SBATCH --cpus-per-task=4         # 可选：给 Julia 一些线程
set -euo pipefail

export JULIA_DEPOT_PATH=/work/d336li/.julia
export JULIA_PROJECT=$HOME/Code_2025/Space_time_project/julia/MadNLP_env

mkdir -p /work/d336li/Code_2025/Space_time_project/julia/MadNLP_env/Euler/Euler_with_dissipation/logs

#（可选）控制 Julia 线程数与 BLAS 线程数
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

$HOME/opt/julia-1.11.3/bin/julia \
  $HOME/Code_2025/Space_time_project/julia/MadNLP_env/Euler/Euler_with_dissipation/sanity_ma57.jl

echo "pkg downloading finished"
