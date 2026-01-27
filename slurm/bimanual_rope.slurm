#!/bin/bash
#SBATCH --job-name="cms_bimanual_rope"
#SBATCH --output="/projects/bcyd/ywang41/cosmos-predict2.5/slurm/outputs/%x/%j.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bfqx-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl               # tells MuJoCo to use EGL
export EGL_DEVICE_ID=0             # or set CUDA_VISIBLE_DEVICES=0 instead
# If you use dm_control, MUJOCO_EGL_DEVICE_ID is also respected:
export MUJOCO_EGL_DEVICE_ID=0

cd /projects/bcyd/ywang41/cosmos-predict2.5/
source .venv/bin/activate

torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=bimanual_rope_2b_128_128 ~dataloader_train.dataloaders
