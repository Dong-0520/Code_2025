#!/bin/bash
#SBATCH --job-name=mytest
#SBATCH --chdir=/work/d336li/Code_2025/Space_time_project/julia/sbatch_test
#SBATCH --output=/work/d336li/Code_2025/Space_time_project/julia/sbatch_test/logs/mytest.%j.out
#SBATCH --error=/work/d336li/Code_2025/Space_time_project/julia/sbatch_test/logs/mytest.%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d336li@uwaterloo.ca

set -euo pipefail
mkdir -p /work/d336li/Code_2025/Space_time_project/julia/sbatch_test/logs

# 正文里可以安全用 $HOME（在这台机器等于 /work/d336li）
echo "HOST=$(hostname)"
echo "HOME=$HOME"

JULIA="$HOME/opt/julia-1.11.3/bin/julia"
$JULIA -v
$JULIA "$HOME/Code_2025/Space_time_project/julia/sbatch_test/my_test.jl"

echo "Job completed 1"
