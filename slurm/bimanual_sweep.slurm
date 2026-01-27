#!/bin/bash
#SBATCH --job-name="cms_bimanual_sweep"
#SBATCH --output="/projects/bcyd/ywang41/cosmos-predict2.5/slurm/outputs/%x/%j.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest
#SBATCH --account=bfqx-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0

cd /projects/bcyd/ywang41/cosmos-predict2.5/
source .venv/bin/activate

torchrun --nproc_per_node=2 --master_port=12343 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=bimanual_sweep_2b_128_128 ~dataloader_train.dataloaders
