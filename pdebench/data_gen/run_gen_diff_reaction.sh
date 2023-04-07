#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate pdebench
which python
export DDE_BACKEND=jax
echo "DDE Backend: $DDE_BACKEND"
export CUDA_VISIBLE_DEVICES=8,9
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1

NJOBS=4
CONFIGS=("sim.Dv=3.494256e-03,1.268606e-02,2.605024e-02 sim.Du=8.214404e-04,1.654288e-02,1.434367e-03 sim.k=1.223760e-02,2.320338e-03,6.412087e-03"
    "sim.Dv=2.686215e-03,1.286294e-02,3.617290e-03 sim.Du=4.604165e-03,7.236904e-03,3.974254e-04 sim.k=1.753930e-02,1.661509e-03,7.000869e-03"
    "sim.Dv=1.950065e-03,1.545350e-02,4.597971e-03 sim.Du=7.935128e-03,1.704641e-02,1.421452e-02 sim.k=1.209407e-03,4.118001e-03,9.386063e-03"
    "sim.Dv=2.067045e-02,1.121939e-02,6.682122e-03 sim.Du=5.269728e-03,1.236232e-02,6.761766e-03 sim.k=4.137835e-04,8.988021e-03,1.162692e-02")

echo "Running ${NJOB} jobs"

mkdir -p /data/nithinc/generated_pdebench/2D/ReactionDiffusion/

for ((i = 0; i < ${#CONFIGS[@]}; i++))
do
    # echo launcher_${i}
    mkdir -p /data/nithinc/generated_pdebench/hydra/2D/ReactionDiffusion/${i}_launcher
    echo "tmux new-session -d -c '$(pwd)' -s '${i}_reactdiff_launcher'  'eval \"\$(conda shell.bash hook)\"; conda activate pdebench; \
export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export OPENBLAS_NUM_THREADS=4; \
export HYDRA_FULL_ERROR=1; python gen_diff_react.py --multirun hydra.job.name=${i}_launcher ${CONFIGS[$i]};' &&"
done


