CHECKPOINT_DIR=/work/nvme/bcyd/ywang41/cosmos_predict2_action_conditioned/custom_tasks/bimanual_sweep/checkpoints/iter_000012000
python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

python -m scripts.compare_video_baselines \
    dataset=real_aloha_dataset dataset.shape_meta.action.shape=[8] \
    dataset.dataset_dir=/media/yixuan/yixuan_4T/projects/diffusion-forcing/data/real_aloha/bimanual_rope_1201 \
    dataset.obs_keys=['camera_0_color'] dataset.horizon=192 dataset.skip_frame=1 \
    dataset.skip_idx=1 +dataset.val_horizon=192 dataset.resolution=128 \
    +cosmos_experiment=bimanual_rope_2b_128_128 \
    +cosmos_ckpt='/home/yixuan/cosmos-predict2.5/cosmos_predict2_action_conditioned/custom_tasks/bimanual_rope/checkpoints/iter_000004000' \
    +cosmos_chunk_size=12 +cosmos_guidance=3.0 +batch_per_eval=2 +save_vis=true +save_dir='compare_video_baselines/bimanual_rope_debug'

