python scripts/convert_hdf5_to_cosmos.py \
    --input_dir /media/yixuan/yixuan_4T/projects/diffusion-forcing/data/real_aloha/bimanual_box_1221 \
    --output_dir /home/yixuan/cosmos-predict2.5/datasets/bimanual_box_1221 \
    --task_name bimanual_box \
    --camera_key obs/images/camera_0_color \
    --state_key obs/joint_pos \
    --action_key action \
    --resize 128 128 \
    --fps 10 \
    --no_relative_actions

python scripts/convert_hdf5_to_cosmos.py \
    --input_dir /media/yixuan/yixuan_4T/projects/diffusion-forcing/data/real_aloha/bimanual_rope_1201 \
    --output_dir /home/yixuan/cosmos-predict2.5/datasets/bimanual_rope_1201 \
    --task_name bimanual_rope \
    --camera_key obs/images/camera_0_color \
    --state_key obs/joint_pos \
    --action_key action \
    --resize 128 128 \
    --fps 10 \
    --no_relative_actions

python scripts/convert_hdf5_to_cosmos.py \
    --input_dir /media/yixuan/yixuan_4T/projects/diffusion-forcing/data/real_aloha/bimanual_sweep_0103 \
    --output_dir /home/yixuan/cosmos-predict2.5/datasets/bimanual_sweep_0103 \
    --task_name bimanual_sweep \
    --camera_key obs/images/camera_0_color \
    --state_key obs/joint_pos \
    --action_key action \
    --resize 128 128 \
    --fps 10 \
    --no_relative_actions

python scripts/convert_hdf5_to_cosmos.py \
    --input_dir /media/yixuan/yixuan_4T/projects/diffusion-forcing/data/real_aloha/pusht_1000_1101 \
    --output_dir /home/yixuan/cosmos-predict2.5/datasets/pusht_1000_1101 \
    --task_name pusht \
    --camera_key obs/images/camera_0_color \
    --state_key obs/joint_pos \
    --action_key action \
    --resize 128 128 \
    --fps 10 \
    --no_relative_actions

python scripts/convert_hdf5_to_cosmos.py \
    --input_dir /media/yixuan/yixuan_4T/projects/diffusion-forcing/data/real_aloha/single_chain_in_box_1224 \
    --output_dir /home/yixuan/cosmos-predict2.5/datasets/single_chain_in_box_1224 \
    --task_name single_chain_in_box \
    --camera_key obs/images/camera_0_color \
    --state_key obs/joint_pos \
    --action_key action \
    --resize 128 128 \
    --fps 10 \
    --no_relative_actions

python scripts/convert_hdf5_to_cosmos.py \
    --input_dir /media/yixuan/yixuan_4T/projects/diffusion-forcing/data/real_aloha/single_grasp_1213 \
    --output_dir /home/yixuan/cosmos-predict2.5/datasets/single_grasp_1213 \
    --task_name single_grasp \
    --camera_key obs/images/camera_0_color \
    --state_key obs/joint_pos \
    --action_key action \
    --resize 128 128 \
    --fps 10 \
    --no_relative_actions
