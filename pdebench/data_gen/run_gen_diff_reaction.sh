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
# CONFIGS=("sim.Dv=3.494256e-03,1.268606e-02,2.605024e-02 sim.Du=8.214404e-04,1.654288e-02,1.434367e-03 sim.k=1.223760e-02,2.320338e-03,6.412087e-03"
#     "sim.Dv=2.686215e-03,1.286294e-02,3.617290e-03 sim.Du=4.604165e-03,7.236904e-03,3.974254e-04 sim.k=1.753930e-02,1.661509e-03,7.000869e-03"
#     "sim.Dv=1.950065e-03,1.545350e-02,4.597971e-03 sim.Du=7.935128e-03,1.704641e-02,1.421452e-02 sim.k=1.209407e-03,4.118001e-03,9.386063e-03"
#     "sim.Dv=2.067045e-02,1.121939e-02,6.682122e-03 sim.Du=5.269728e-03,1.236232e-02,6.761766e-03 sim.k=4.137835e-04,8.988021e-03,1.162692e-02")

CONFIGS=("sim.Dv=1.197443e-01,2.725704e-01,1.856076e-01,3.023561e-01,3.132344e-01,3.369890e-02,7.570828e-03 sim.Du=4.187508e-01,1.297511e-01,1.172420e-01,4.978229e-01,2.351847e-01,4.182471e-01,2.382290e-01 sim.k=6.426775e-02,1.591103e-02,6.385121e-02,8.693649e-02,5.279494e-02,7.438393e-02,6.746974e-02"
    "sim.Dv=3.295169e-02,3.793569e-01,2.959587e-01,1.513326e-01,1.647486e-02,4.328981e-01,2.369018e-01 sim.Du=3.594401e-01,4.394185e-01,3.570933e-01,4.605572e-01,1.975422e-01,4.004743e-01,2.223661e-01 sim.k=9.362309e-02,8.800780e-02,1.064798e-02,1.446092e-02,2.248171e-02,9.658253e-02,4.418002e-02"
    "sim.Dv=3.136975e-01,1.512121e-01,2.541142e-01,1.935473e-01,1.761043e-01,2.929520e-01,2.925416e-01 sim.Du=4.521105e-01,3.410229e-01,4.644799e-01,4.282146e-01,4.954957e-01,3.356696e-01,8.163350e-02 sim.k=8.620312e-02,9.649866e-02,9.056490e-02,5.734164e-02,7.166788e-02,2.190137e-02,8.332919e-02"
    "sim.Dv=2.871926e-01,1.431938e-01,3.266683e-02,4.271173e-01,4.949132e-01,4.517053e-02,4.004971e-01 sim.Du=2.052899e-01,7.546761e-02,1.470162e-01,3.844191e-01,4.363962e-01,2.219061e-02,3.073048e-01 sim.k=5.449084e-03,7.212561e-02,3.376446e-02,8.820963e-02,9.808294e-02,5.103662e-02,9.985239e-02")


echo "Running ${NJOB} jobs"

mkdir -p /data/nithinc/generated_pdebench/2D/ReactionDiffusion/

for ((i = 0; i < ${#CONFIGS[@]}; i++))
do
    # echo launcher_${i}
    mkdir -p /data/nithinc/residual_and_visualization_pdebench/hydra/2D/ReactionDiffusion/${i}_launcher
    echo "tmux new-session -d -c '$(pwd)' -s '${i}_reactdiff_launcher'  'eval \"\$(conda shell.bash hook)\"; conda activate pdebench; \
export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export OPENBLAS_NUM_THREADS=4; \
export HYDRA_FULL_ERROR=1; python gen_diff_react.py --multirun hydra.job.name=${i}_launcher ${CONFIGS[$i]};' &&"
done
# tmux new-session -d -c '/home/nithinc/Workspace/constrained-pdes/PDEBench/pdebench/data_gen' -s '0_reactdiff_launcher'  'eval "$(conda shell.bash hook)"; conda activate pdebench; export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export OPENBLAS_NUM_THREADS=4; export HYDRA_FULL_ERROR=1; python gen_diff_react.py --multirun hydra.job.name=0_launcher sim.Dv=1.197443e-01,2.725704e-01,1.856076e-01,3.023561e-01,3.132344e-01,3.369890e-02,7.570828e-03 sim.Du=4.187508e-01,1.297511e-01,1.172420e-01,4.978229e-01,2.351847e-01,4.182471e-01,2.382290e-01 sim.k=6.426775e-02,1.591103e-02,6.385121e-02,8.693649e-02,5.279494e-02,7.438393e-02,6.746974e-02;' &&
# tmux new-session -d -c '/home/nithinc/Workspace/constrained-pdes/PDEBench/pdebench/data_gen' -s '1_reactdiff_launcher'  'eval "$(conda shell.bash hook)"; conda activate pdebench; export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export OPENBLAS_NUM_THREADS=4; export HYDRA_FULL_ERROR=1; python gen_diff_react.py --multirun hydra.job.name=1_launcher sim.Dv=3.295169e-02,3.793569e-01,2.959587e-01,1.513326e-01,1.647486e-02,4.328981e-01,2.369018e-01 sim.Du=3.594401e-01,4.394185e-01,3.570933e-01,4.605572e-01,1.975422e-01,4.004743e-01,2.223661e-01 sim.k=9.362309e-02,8.800780e-02,1.064798e-02,1.446092e-02,2.248171e-02,9.658253e-02,4.418002e-02;' &&
# tmux new-session -d -c '/home/nithinc/Workspace/constrained-pdes/PDEBench/pdebench/data_gen' -s '2_reactdiff_launcher'  'eval "$(conda shell.bash hook)"; conda activate pdebench; export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export OPENBLAS_NUM_THREADS=4; export HYDRA_FULL_ERROR=1; python gen_diff_react.py --multirun hydra.job.name=2_launcher sim.Dv=3.136975e-01,1.512121e-01,2.541142e-01,1.935473e-01,1.761043e-01,2.929520e-01,2.925416e-01 sim.Du=4.521105e-01,3.410229e-01,4.644799e-01,4.282146e-01,4.954957e-01,3.356696e-01,8.163350e-02 sim.k=8.620312e-02,9.649866e-02,9.056490e-02,5.734164e-02,7.166788e-02,2.190137e-02,8.332919e-02;' &&
# tmux new-session -d -c '/home/nithinc/Workspace/constrained-pdes/PDEBench/pdebench/data_gen' -s '3_reactdiff_launcher'  'eval "$(conda shell.bash hook)"; conda activate pdebench; export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4; export OPENBLAS_NUM_THREADS=4; export HYDRA_FULL_ERROR=1; python gen_diff_react.py --multirun hydra.job.name=3_launcher sim.Dv=2.871926e-01,1.431938e-01,3.266683e-02,4.271173e-01,4.949132e-01,4.517053e-02,4.004971e-01 sim.Du=2.052899e-01,7.546761e-02,1.470162e-01,3.844191e-01,4.363962e-01,2.219061e-02,3.073048e-01 sim.k=5.449084e-03,7.212561e-02,3.376446e-02,8.820963e-02,9.808294e-02,5.103662e-02,9.985239e-02;' &&

