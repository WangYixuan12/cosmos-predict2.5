# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

try:
    from cosmos_predict2._src.predict2.action.configs.action_conditioned.experiment.gr00t_customized_gr1 import (
        register_gr00t_customized_gr1_data,
    )
except ImportError:
    register_gr00t_customized_gr1_data = None

# bridge dataset path
base_path = "datasets/bridge/"

train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


# experiment for next-frame prediction
bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
)
bridge_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
)

# experiment for action-sequence video prediction
bridge_13frame_480_640_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)
bridge_13frame_480_640_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
)


# ------------------------------------------------------------


# create dataloader for each dataset
def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def build_webdataset(webdataset_instance, **kwargs):
    """Helper function to build WebDataset from a WebDataset instance.

    WebDatasets need to call build_dataset() to get the actual iterable dataset
    that can be used with DataLoader.

    Args:
        webdataset_instance: An instantiated WebDataset object.
        **kwargs: Additional parameters to override on the webdataset instance
            before building. This allows experiment configs to override parameters
            like gripper_rescale_factor, num_action_per_chunk, etc.
    """
    # Apply any parameter overrides to the webdataset instance
    for key, value in kwargs.items():
        if hasattr(webdataset_instance, key):
            setattr(webdataset_instance, key, value)
    return webdataset_instance.build_dataset()


bridge_train_dataloader = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
)
bridge_val_dataloader = L(DataLoader)(
    dataset=bridge_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=True,
)

bridge_13frame_480_640_train_dataloader = L(DataLoader)(
    dataset=bridge_13frame_480_640_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_13frame_480_640_train_dataset),
    batch_size=1,
    drop_last=True,
)
bridge_13frame_480_640_val_dataloader = L(DataLoader)(
    dataset=bridge_13frame_480_640_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_13frame_480_640_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data():
    cs = ConfigStore.instance()
    from cosmos_predict2._src.predict2.configs.common.mock_data import MOCK_DATA_INTERLEAVE_CONFIG

    # Always register mock dataloaders to satisfy defaults when not overridden
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_train",
        node=bridge_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_val",
        node=bridge_val_dataloader,
    )

    # 13 frame 480 640
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_13frame_480_640_train",
        node=bridge_13frame_480_640_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_13frame_480_640_val",
        node=bridge_13frame_480_640_val_dataloader,
    )

    # Register custom task datasets
    # Scripted Sim ALOHA (4D actions, 128x128 images)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="scripted_sim_aloha_train",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/scripted_sim_aloha_100/annotation/train",
                val_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                test_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                video_path="datasets/scripted_sim_aloha_100/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="train",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/scripted_sim_aloha_100/annotation/train",
                    val_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                    test_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                    video_path="datasets/scripted_sim_aloha_100/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="train",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="scripted_sim_aloha_val",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/scripted_sim_aloha_100/annotation/train",
                val_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                test_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                video_path="datasets/scripted_sim_aloha_100/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="val",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/scripted_sim_aloha_100/annotation/train",
                    val_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                    test_annotation_path="datasets/scripted_sim_aloha_100/annotation/val",
                    video_path="datasets/scripted_sim_aloha_100/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="val",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    # Bimanual Rope (8D actions)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bimanual_rope_train",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/bimanual_rope_1201/annotation/train",
                val_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                test_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                video_path="datasets/bimanual_rope_1201/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="train",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/bimanual_rope_1201/annotation/train",
                    val_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                    test_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                    video_path="datasets/bimanual_rope_1201/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="train",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bimanual_rope_val",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/bimanual_rope_1201/annotation/train",
                val_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                test_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                video_path="datasets/bimanual_rope_1201/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="val",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/bimanual_rope_1201/annotation/train",
                    val_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                    test_annotation_path="datasets/bimanual_rope_1201/annotation/val",
                    video_path="datasets/bimanual_rope_1201/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="val",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    # Bimanual Box (14D actions)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bimanual_box_train",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/bimanual_box_1221/annotation/train",
                val_annotation_path="datasets/bimanual_box_1221/annotation/val",
                test_annotation_path="datasets/bimanual_box_1221/annotation/val",
                video_path="datasets/bimanual_box_1221/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="train",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/bimanual_box_1221/annotation/train",
                    val_annotation_path="datasets/bimanual_box_1221/annotation/val",
                    test_annotation_path="datasets/bimanual_box_1221/annotation/val",
                    video_path="datasets/bimanual_box_1221/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="train",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bimanual_box_val",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/bimanual_box_1221/annotation/train",
                val_annotation_path="datasets/bimanual_box_1221/annotation/val",
                test_annotation_path="datasets/bimanual_box_1221/annotation/val",
                video_path="datasets/bimanual_box_1221/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="val",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/bimanual_box_1221/annotation/train",
                    val_annotation_path="datasets/bimanual_box_1221/annotation/val",
                    test_annotation_path="datasets/bimanual_box_1221/annotation/val",
                    video_path="datasets/bimanual_box_1221/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="val",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    # Bimanual Sweep (4D actions)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bimanual_sweep_train",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/bimanual_sweep_0103/annotation/train",
                val_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                test_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                video_path="datasets/bimanual_sweep_0103/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="train",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/bimanual_sweep_0103/annotation/train",
                    val_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                    test_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                    video_path="datasets/bimanual_sweep_0103/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="train",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bimanual_sweep_val",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/bimanual_sweep_0103/annotation/train",
                val_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                test_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                video_path="datasets/bimanual_sweep_0103/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="val",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/bimanual_sweep_0103/annotation/train",
                    val_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                    test_annotation_path="datasets/bimanual_sweep_0103/annotation/val",
                    video_path="datasets/bimanual_sweep_0103/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="val",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    # PushT (4D actions)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="pusht_train",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/pusht_1000_1101/annotation/train",
                val_annotation_path="datasets/pusht_1000_1101/annotation/val",
                test_annotation_path="datasets/pusht_1000_1101/annotation/val",
                video_path="datasets/pusht_1000_1101/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="train",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/pusht_1000_1101/annotation/train",
                    val_annotation_path="datasets/pusht_1000_1101/annotation/val",
                    test_annotation_path="datasets/pusht_1000_1101/annotation/val",
                    video_path="datasets/pusht_1000_1101/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="train",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="pusht_val",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/pusht_1000_1101/annotation/train",
                val_annotation_path="datasets/pusht_1000_1101/annotation/val",
                test_annotation_path="datasets/pusht_1000_1101/annotation/val",
                video_path="datasets/pusht_1000_1101/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="val",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/pusht_1000_1101/annotation/train",
                    val_annotation_path="datasets/pusht_1000_1101/annotation/val",
                    test_annotation_path="datasets/pusht_1000_1101/annotation/val",
                    video_path="datasets/pusht_1000_1101/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="val",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    # Single Chain in Box (4D actions)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="single_chain_in_box_train",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/single_chain_in_box_1224/annotation/train",
                val_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                test_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                video_path="datasets/single_chain_in_box_1224/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="train",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/single_chain_in_box_1224/annotation/train",
                    val_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                    test_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                    video_path="datasets/single_chain_in_box_1224/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="train",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="single_chain_in_box_val",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/single_chain_in_box_1224/annotation/train",
                val_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                test_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                video_path="datasets/single_chain_in_box_1224/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="val",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/single_chain_in_box_1224/annotation/train",
                    val_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                    test_annotation_path="datasets/single_chain_in_box_1224/annotation/val",
                    video_path="datasets/single_chain_in_box_1224/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="val",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    # Single Grasp (4D actions)
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="single_grasp_train",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/single_grasp_1213/annotation/train",
                val_annotation_path="datasets/single_grasp_1213/annotation/val",
                test_annotation_path="datasets/single_grasp_1213/annotation/val",
                video_path="datasets/single_grasp_1213/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="train",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/single_grasp_1213/annotation/train",
                    val_annotation_path="datasets/single_grasp_1213/annotation/val",
                    test_annotation_path="datasets/single_grasp_1213/annotation/val",
                    video_path="datasets/single_grasp_1213/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="train",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="single_grasp_val",
        node=L(DataLoader)(
            dataset=L(Dataset_3D)(
                train_annotation_path="datasets/single_grasp_1213/annotation/train",
                val_annotation_path="datasets/single_grasp_1213/annotation/val",
                test_annotation_path="datasets/single_grasp_1213/annotation/val",
                video_path="datasets/single_grasp_1213/",
                fps_downsample_ratio=1,
                num_action_per_chunk=12,
                cam_ids=[0],
                accumulate_action=False,
                video_size=[128, 128],
                val_start_frame_interval=1,
                mode="val",
            ),
            sampler=L(get_sampler)(
                dataset=L(Dataset_3D)(
                    train_annotation_path="datasets/single_grasp_1213/annotation/train",
                    val_annotation_path="datasets/single_grasp_1213/annotation/val",
                    test_annotation_path="datasets/single_grasp_1213/annotation/val",
                    video_path="datasets/single_grasp_1213/",
                    fps_downsample_ratio=1,
                    num_action_per_chunk=12,
                    cam_ids=[0],
                    accumulate_action=False,
                    video_size=[128, 128],
                    val_start_frame_interval=1,
                    mode="val",
                ),
            ),
            batch_size=1,
            drop_last=True,
        ),
    )

    # Register gr00t_customized_gr1 data
    if register_gr00t_customized_gr1_data is not None:
        register_gr00t_customized_gr1_data()
