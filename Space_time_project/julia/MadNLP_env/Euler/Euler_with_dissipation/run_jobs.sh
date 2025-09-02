#!/bin/bash
#SBATCH --job-name=optimization_iter_400
#SBATCH --chdir=/work/d336li/Code_2025/Space_time_project/julia/sbatch_test
#SBATCH --output=/work/d336li/Code_2025/Space_time_project/julia/MadNLP_env/Euler/Euler_with_dissipation/logs/mytest.%j.out
#SBATCH --error=/work/d336li/Code_2025/Space_time_project/julia/MadNLP_env/Euler/Euler_with_dissipation/logs/mytest.%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=2G
#SBATCH --partition=cpu_pr3           # ← 改成你查到的CPU分区名，例如 cpu_pr2

set -euo pipefail
export JULIA_DEPOT_PATH=/work/d336li/.julia       # 统一依赖目录
export JULIA_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_PKG_USE_CLI_GIT=true
export JULIA_PKG_SERVER=https://pkg.julialang.org

echo "starting running file"

$HOME/opt/julia-1.11.3/bin/julia \
  $HOME/Code_2025/Space_time_project/julia/MadNLP_env/Euler/Euler_with_dissipation/Euler_optimization_pseudo_transient_sbatch.jl

echo "end running file"
