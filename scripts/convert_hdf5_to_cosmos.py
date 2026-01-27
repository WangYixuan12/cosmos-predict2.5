#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert HDF5 datasets to Cosmos Bridge-like format (JSON annotations + MP4 videos).

Usage:
    python scripts/convert_hdf5_to_cosmos.py \
        --input_dir /path/to/hdf5/data \
        --output_dir datasets/my_task \
        --task_name my_task \
        --camera_key obs/images/camera_0_color \
        --action_key action \
        --train_ratio 0.9 \
        --resize 256 320
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def center_crop_batch(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[1:3]
    th, tw = crop_size
    if h / w > th / tw:
        # image is taller than crop
        crop_w = w
        crop_h = int(round(w * th / tw))
    elif h / w < th / tw:
        # image is wider than crop
        crop_h = h
        crop_w = int(round(h * tw / th))
    else:
        return img
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    return img[:, y1 : y1 + crop_h, x1 : x1 + crop_w, :]

def create_video_from_frames(
    frames: np.ndarray,
    output_path: str,
    fps: int = 15,
    codec: str = 'mp4v'
) -> None:
    """
    Create MP4 video from numpy array of frames.

    Args:
        frames: Array of shape (T, H, W, 3) with uint8 RGB images
        output_path: Path to save the MP4 file
        fps: Frames per second
        codec: Video codec fourcc code
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(T):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def resize_frames(frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frames to target size.

    Args:
        frames: Array of shape (T, H, W, 3)
        target_size: (height, width) tuple

    Returns:
        resized_frames: Array of shape (T, target_H, target_W, 3)
    """
    T, H, W, C = frames.shape
    target_H, target_W = target_size

    resized = np.zeros((T, target_H, target_W, C), dtype=frames.dtype)
    for i in range(T):
        resized[i] = cv2.resize(frames[i], (target_W, target_H), interpolation=cv2.INTER_LINEAR)

    return resized


def convert_hdf5_episode(
    hdf5_path: str,
    output_base_dir: str,
    episode_idx: int,
    split: str,
    camera_key: str,
    action_key: str,
    state_key: Optional[str] = None,
    resize_resolution: Optional[Tuple[int, int]] = None,
    fps: int = 15,
    use_relative_actions: bool = True,
    task_name: str = "custom_task"
) -> Dict:
    """
    Convert a single HDF5 episode to Cosmos format.

    Returns:
        annotation: Dictionary containing the episode annotation
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load images
        camera_data = f
        for key in camera_key.split('/'):
            camera_data = camera_data[key]
        frames = np.array(camera_data)  # Shape: (T, H, W, 3)

        # Resize if needed
        if resize_resolution is not None:
            frames = center_crop_batch(frames, resize_resolution)
            frames = resize_frames(frames, resize_resolution)

        # Load actions
        actions = np.array(f[action_key])  # Shape: (T, action_dim)
        
        if use_relative_actions:
            # Convert absolute actions to relative actions
            relative_actions = actions[1:] - actions[:-1]
            actions = relative_actions

        # Load or create states
        if state_key is not None and state_key in f:
            states = np.array(f[state_key])
        else:
            # If no explicit state, use actions as proxy
            states = actions.copy()

    # Create video
    video_dir = os.path.join(output_base_dir, 'videos', split, str(episode_idx))
    video_path = os.path.join(video_dir, 'rgb.mp4')
    create_video_from_frames(frames, video_path, fps=fps)

    # Prepare annotation
    T = frames.shape[0]
    action_dim = actions.shape[1]

    # Create state array (use actions as state proxy if no explicit state)
    # In Bridge format: state is [x, y, z, roll, pitch, yaw, gripper]
    # We'll adapt based on your action dimension
    state_list = states.tolist() if states.shape[1] >= action_dim else actions.tolist()

    # Extract gripper states (assume last dimension is gripper)
    if states.shape[1] == 14:
        continuous_gripper_state = [[s[6], s[13]] for s in state_list]
    elif states.shape[1] == 7:
        continuous_gripper_state = [s[6] for s in state_list]

    annotation = {
        'task': task_name,
        'texts': [f'{task_name} demonstration'],
        'videos': [{'video_path': f'videos/{split}/{episode_idx}/rgb.mp4'}],
        'action': actions.tolist(),
        'state': state_list,
        'continuous_gripper_state': continuous_gripper_state,
        'episode_id': f'{episode_idx}',
        'latent_videos': []
    }

    return annotation


def convert_dataset(
    input_dir: str,
    output_dir: str,
    task_name: str,
    camera_key: str = 'obs/images/camera_0_color',
    action_key: str = 'action',
    state_key: Optional[str] = 'obs/ee_pos',
    resize_resolution: Optional[Tuple[int, int]] = None,
    fps: int = 15,
    use_relative_actions: bool = True
) -> None:
    """
    Convert entire HDF5 dataset to Cosmos format.

    Args:
        input_dir: Directory containing train/ and val/ subdirectories with episode_*.hdf5
        output_dir: Output directory for converted dataset
        task_name: Name of the task
        camera_key: HDF5 path to camera images (e.g., 'obs/images/camera_0_color')
        action_key: HDF5 key for actions
        state_key: HDF5 key for states (optional)
        train_ratio: Ratio of train/val split (only used if val/ doesn't exist)
        resize_resolution: Target resolution as (height, width) tuple
        fps: Video frame rate
        use_relative_actions: Whether to compute relative actions
    """
    os.makedirs(output_dir, exist_ok=True)

    # Process train and val splits
    for split in ['train', 'val']:
        split_input_dir = os.path.join(input_dir, split)

        if not os.path.exists(split_input_dir):
            print(f"Warning: {split_input_dir} does not exist, skipping {split} split")
            continue

        # Find all episode HDF5 files
        episode_files = sorted([
            f for f in os.listdir(split_input_dir)
            if f.startswith('episode_') and f.endswith('.hdf5')
        ])

        print(f"\nProcessing {split} split: {len(episode_files)} episodes")

        # Create annotation directory
        annotation_dir = os.path.join(output_dir, 'annotation', split)
        os.makedirs(annotation_dir, exist_ok=True)

        # Process each episode
        for i, episode_file in enumerate(tqdm(episode_files, desc=f'Converting {split}')):
            episode_path = os.path.join(split_input_dir, episode_file)

            try:
                annotation = convert_hdf5_episode(
                    hdf5_path=episode_path,
                    output_base_dir=output_dir,
                    episode_idx=i,
                    split=split,
                    camera_key=camera_key,
                    action_key=action_key,
                    state_key=state_key,
                    resize_resolution=resize_resolution,
                    fps=fps,
                    use_relative_actions=use_relative_actions,
                    task_name=task_name
                )

                # Save annotation
                annotation_path = os.path.join(annotation_dir, f'{i}.json')
                with open(annotation_path, 'w') as f:
                    json.dump(annotation, f, indent=2)

            except Exception as e:
                print(f"\nError processing {episode_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\nConversion complete! Dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 dataset to Cosmos format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing train/val subdirectories with episode_*.hdf5')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for converted dataset')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Name of the task')
    parser.add_argument('--camera_key', type=str, default='obs/images/camera_0_color',
                        help='HDF5 path to camera images')
    parser.add_argument('--action_key', type=str, default='action',
                        help='HDF5 key for actions')
    parser.add_argument('--state_key', type=str, default=None,
                        help='HDF5 key for states (optional)')
    parser.add_argument('--resize', type=int, nargs=2, default=None, metavar=('HEIGHT', 'WIDTH'),
                        help='Resize resolution as HEIGHT WIDTH (e.g., --resize 256 320)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Video frame rate')
    parser.add_argument('--no_relative_actions', action='store_true',
                        help='Do not compute relative actions (use absolute actions)')

    args = parser.parse_args()

    resize_resolution = tuple(args.resize) if args.resize else None

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task_name=args.task_name,
        camera_key=args.camera_key,
        action_key=args.action_key,
        state_key=args.state_key,
        resize_resolution=resize_resolution,
        fps=args.fps,
        use_relative_actions=not args.no_relative_actions
    )


if __name__ == '__main__':
    main()
