import time
from pathlib import Path
from typing import Dict, Optional, Callable

import cv2
import numpy as np
import torch
import torch.utils
from einops import rearrange
from omegaconf import DictConfig
from sewar.full_ref import uqi
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)
from tqdm import tqdm

import sys
curr_file_path = Path(__file__).resolve().parent
sys.path.append(f"{curr_file_path}/../../diffusion-forcing")

from datasets.latent_dynamics import (
    RealAlohaDataset,
    SimAlohaDataset,
)
from algorithms.common.metrics import FrechetInceptionDistance, FrechetVideoDistance

from cosmos_predict2._src.predict2.action.inference.inference_pipeline import ActionVideo2WorldInference

def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

# set seed for reproducibility
seed = 42  # Choose your desired seed
torch.manual_seed(seed)
np.random.seed(seed)


def torch_pred_to_video(pred: torch.Tensor, dir_name: str) -> None:
    """Convert a tensor to a video and save it."""
    assert pred.ndim == 5  # (B, T, C, H, W)
    assert pred.max() <= 1.0 and pred.min() >= 0.0
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    for i in range(pred.shape[0]):
        vid_writer = cv2.VideoWriter(
            f"{dir_name}/video_{i}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            2,
            (pred.shape[4], pred.shape[3]),
        )
        for j in range(pred.shape[1]):
            frame = pred[i, j].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vid_writer.write(frame)
        vid_writer.release()


def build_dataset(cfg: DictConfig, split: str) -> Optional[torch.utils.data.Dataset]:
    # build the dataset
    compatible_datasets = {
        "sim_aloha_dataset": SimAlohaDataset,
        "real_aloha_dataset": RealAlohaDataset,
    }
    dataset = compatible_datasets[cfg.dataset._name](cfg.dataset)  # noqa
    if split == "training":
        return dataset
    elif split == "validation":
        return dataset.get_validation_dataset()
    elif split == "test":
        return dataset
    else:
        raise NotImplementedError(f"split '{split}' is not implemented")

def load_cosmos_wm(experiment_name: str, ckpt_path: str) -> ActionVideo2WorldInference:
    """Load Cosmos action-conditioned world model from checkpoint.

    Args:
        experiment_name: Experiment config name (e.g., 'bimanual_rope_2b_128_128')
        ckpt_path: Path to checkpoint (local or S3)

    Returns:
        cosmos_wm: Cosmos inference pipeline
    """
    cosmos_wm = ActionVideo2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=ckpt_path,
        s3_credential_path="",  # Empty for local checkpoints
        context_parallel_size=1,  # Single GPU inference
    )
    return cosmos_wm


@torch.no_grad()
def cosmos_infer(
    cosmos_wm: ActionVideo2WorldInference,
    batch: Dict[str, torch.Tensor],
    obs_key: str,
    chunk_size: int = 12,
    guidance: int = 0,
    action_scale: float = 20.0,
) -> torch.Tensor:
    """Inference with Cosmos action-conditioned world model.

    Args:
        cosmos_wm: Cosmos inference pipeline
        batch: Data batch with obs and action
        obs_key: Key for observations in batch
        chunk_size: Number of frames to predict per chunk
        guidance: Guidance scale for CFG (0 = no guidance)
        action_scale: Scaling factor for actions

    Returns:
        pred: Predicted video (B, T, C, H, W) in [0, 1]
    """
    device = next(iter(cosmos_wm.model.parameters())).device

    # Get batch dimensions
    B, T = batch["obs"][obs_key].shape[:2]
    frames = batch["obs"][obs_key]  # (B, T, C, H, W) in [0, 1]
    actions = batch["action"]  # (B, T, A)

    # Process each batch item separately (Cosmos expects batch_size=1)
    all_preds = []

    for b in range(B):
        # Get first frame as numpy array (H, W, C) uint8
        first_frame = frames[b, 0].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        first_frame = (first_frame * 255.0).astype(np.uint8)

        # Scale actions
        batch_actions = actions[b].cpu().numpy()  # (T, A)
        batch_actions = batch_actions * action_scale

        # Collect predicted frames
        pred_frames = [first_frame]
        current_frame = first_frame

        # Autoregressive rollout with chunks
        for t in range(0, T - 1, chunk_size):
            end_t = min(t + chunk_size, T - 1)
            chunk_actions = batch_actions[t:end_t]  # (chunk_size, A)

            # Run inference for this chunk
            next_frame, video_chunk = cosmos_wm.step_inference(
                img_array=current_frame,
                action=chunk_actions,
                guidance=guidance,
                seed=t,
                num_latent_conditional_frames=1,
            )

            # Extract predicted frames (skip first frame which is conditioning)
            # video_chunk: (chunk_len, H, W, C) uint8
            for i in range(1, video_chunk.shape[0]):
                if len(pred_frames) < T:
                    pred_frames.append(video_chunk[i])

            # Update current frame for next chunk
            current_frame = next_frame

            if len(pred_frames) >= T:
                break

        # Ensure we have exactly T frames
        pred_frames = pred_frames[:T]
        while len(pred_frames) < T:
            pred_frames.append(pred_frames[-1])  # Repeat last frame if needed

        # Convert to tensor (T, H, W, C) -> (T, C, H, W)
        pred_video = np.stack(pred_frames, axis=0)  # (T, H, W, C) uint8
        pred_video = torch.from_numpy(pred_video).float() / 255.0  # [0, 1]
        pred_video = pred_video.permute(0, 3, 1, 2)  # (T, C, H, W)
        all_preds.append(pred_video)

    # Stack batch dimension
    pred = torch.stack(all_preds, dim=0)  # (B, T, C, H, W)

    return pred


def mse_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """MSE Loss between prediction and ground truth.

    Args:
        pred: predicted tensor (B, T, C, H, W)
        gt: ground truth tensor (B, T, C, H, W)

    Returns:
        mse: MSE loss between prediction and ground truth in (B, T)
    """
    assert pred.shape == gt.shape
    mse = torch.nn.functional.mse_loss(pred, gt, reduction="none")
    return mse.mean(dim=(2, 3, 4))


def psnr_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """PSNR metric between prediction and ground truth.

    Args:
        pred: predicted tensor (B, T, C, H, W)
        gt: ground truth tensor (B, T, C, H, W)
    """
    assert pred.shape == gt.shape
    psnr = torch.zeros((pred.shape[0], pred.shape[1]))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            psnr[i, j] = peak_signal_noise_ratio(
                pred[i, j : j + 1], gt[i, j : j + 1], data_range=1.0
            )
    return psnr


def ssim_metric(pred: torch.Tensor, gt: torch.Tensor, n_hist: int = 0) -> torch.Tensor:
    """SSIM metric between prediction and ground truth.

    Args:
        pred: predicted tensor (B, T, C, H, W)
        gt: ground truth tensor (B, T, C, H, W)
        n_hist: number of history frames to ignore
    """
    assert pred.shape == gt.shape
    B = pred.shape[0]
    pred = rearrange(pred[:, n_hist:], "b t c h w -> (b t) c h w")
    gt = rearrange(gt[:, n_hist:], "b t c h w -> (b t) c h w")
    ssim = structural_similarity_index_measure(pred, gt, reduction="none")
    return ssim.reshape(B, -1)


def lpips_metric(pred: torch.Tensor, gt: torch.Tensor, n_hist: int = 0) -> torch.Tensor:
    loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net="alex").to(pred.device)

    lpips = torch.zeros((pred.shape[0], pred.shape[1]))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            lpips[i, j] = loss_fn_alex(pred[i, j : j + 1], gt[i, j : j + 1])
    return lpips


def uiqi_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """UIQI metric between prediction and ground truth.

    Args:
        pred: predicted tensor (B, T, C, H, W)
        gt: ground truth tensor (B, T, C, H, W)
    """
    assert pred.shape == gt.shape
    B, T = pred.shape[0:2]
    pred = rearrange(pred, "b t c h w -> (b t) c h w")
    gt = rearrange(gt, "b t c h w -> (b t) c h w")
    uiqi = torch.zeros((B * T))
    for i in range(pred.shape[0]):
        pred_i = pred[i].permute(1, 2, 0).detach().cpu().numpy()
        pred_i = (pred_i * 255.0).astype(np.uint8)
        gt_i = gt[i].permute(1, 2, 0).detach().cpu().numpy()
        gt_i = (gt_i * 255.0).astype(np.uint8)
        uiqi[i] = uqi(pred_i, gt_i)
    uiqi = uiqi.reshape(B, T)
    return uiqi


def fvd_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    fvd_model = FrechetVideoDistance().to(pred.device)
    pred = (pred - 0.5) * 2.0
    gt = (gt - 0.5) * 2.0
    pred = rearrange(pred, "b t c h w -> t b c h w")
    gt = rearrange(gt, "b t c h w -> t b c h w")
    if pred.shape[0] < 16:
        pred = pred.repeat_interleave(repeats=4, dim=0)
        gt = gt.repeat_interleave(repeats=4, dim=0)
    fvd_ls = []
    for i in range(pred.shape[1]):
        fvd_ls.append(fvd_model.compute(pred[:, i : i + 1], gt[:, i : i + 1]))
    return torch.tensor(fvd_ls)


def fid_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    fid_model = FrechetInceptionDistance().to(pred.device)
    pred = (pred * 255.0).to(torch.uint8)
    gt = (gt * 255.0).to(torch.uint8)
    fid_ls = []
    for i in range(pred.shape[0]):
        fid_model.update(pred[i], real=False)
        fid_model.update(gt[i], real=True)
        fid_ls.append(fid_model.compute())
        fid_model.reset()
    return torch.tensor(fid_ls)


@torch.no_grad()
def eval_one_batch(
    batch: Dict,
    batch_idx: int,
    methods: list,
    cosmos_wm: Optional[ActionVideo2WorldInference],
    cosmos_chunk_size: int,
    cosmos_guidance: int,
    obs_keys: list,
    save_vis: bool = True,
    save_dir: Optional[str] = None,
) -> np.ndarray:
    obs_key = obs_keys[0]
    B = batch["obs"][obs_key].shape[0]
    dict_keys = list(batch["obs"].keys())
    for k in dict_keys:
        if k not in obs_keys:
            del batch["obs"][k]

    dinowm_tf = Compose(
        [
            Resize(128, interpolation=InterpolationMode.BILINEAR, antialias=True),
            CenterCrop((128, 128)),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    
    # infer Cosmos
    if "cosmos" in methods:
        print("Inferring Cosmos...")
        cosmos_time_s = time.time()
        cosmos_pred = cosmos_infer(
            cosmos_wm, batch, obs_key,
            chunk_size=cosmos_chunk_size,
            guidance=cosmos_guidance
        )
        cosmos_time_e = time.time()
        print("Cosmos inference done.")

    GT = rearrange(batch["obs"][obs_key], "b t c h w -> (b t) c h w")
    GT = dinowm_tf(GT)
    GT = rearrange(GT, "(b t) c h w -> b t c h w", b=B)
    GT = torch.clamp(GT, -1, 1) / 2.0 + 0.5

    if "cosmos" in methods:
        print(f"Cosmos time: {cosmos_time_e - cosmos_time_s:.4f}s")

    # save the predictions
    if save_vis:
        save_dir_batch_i = f"{save_dir}/batch_{batch_idx}"
        if "cosmos" in methods:
            torch_pred_to_video(cosmos_pred, f"{save_dir_batch_i}/cosmos")
        torch_pred_to_video(GT, f"{save_dir_batch_i}/GT")

    # MSE metrics
    pred_ls = []
    res = np.zeros((len(methods), 7))
    if "cosmos" in methods:
        pred_ls.append(cosmos_pred)
    for i, pred in enumerate(pred_ls):
        mse = mse_metric(pred, GT)
        psnr = psnr_metric(pred, GT)
        ssim = ssim_metric(pred, GT)
        lpips = lpips_metric(pred, GT)
        uiqi = uiqi_metric(pred, GT)
        fvd = fvd_metric(pred, GT)
        fid = fid_metric(pred, GT)

        res[i, 0] = mse.mean()
        res[i, 1] = lpips.mean()
        res[i, 2] = fid.mean()
        res[i, 3] = psnr.mean()
        res[i, 4] = ssim.mean()
        res[i, 5] = uiqi.mean()
        res[i, 6] = fvd.mean()
    return res

def main() -> None:
    """Collect demonstration for the Push-T task.

    Usage: python demo_pusht.py -o data/pusht_demo.zarr

    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area.
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    obs_keys = ["camera_0_color"]
    device = "cuda"
    save_vis = True
    B = 1
    save_dir = "/home/yixuan/diffusion-forcing/compare_video_baselines/cosmos_debug"
    methods = ["cosmos"]
    names = ["Cosmos"]

    # load Cosmos world model
    cosmos_experiment = "bimanual_rope_2b_128_128"
    cosmos_ckpt = "/home/yixuan/cosmos-predict2.5/cosmos_predict2_action_conditioned/custom_tasks/bimanual_rope/checkpoints/iter_000004000"
    cosmos_chunk_size = 12
    cosmos_guidance = 3.0
    print(f"Loading Cosmos model: {cosmos_experiment} from {cosmos_ckpt}")
    cosmos_wm = load_cosmos_wm(cosmos_experiment, cosmos_ckpt)
    print("Cosmos model loaded successfully")

    # build algo and load ckpt
    import ipdb; ipdb.set_trace()
    dataset: SimAlohaDataset = build_dataset(cfg, split="validation")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False)

    all_res = []
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # if batch_idx > 1:
        #     break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        res = eval_one_batch(
            batch,
            batch_idx,
            methods,
            cosmos_wm,
            cosmos_chunk_size,
            cosmos_guidance,
            device,
            obs_keys,
            save_vis=save_vis,
            save_dir=save_dir,
        )
        all_res.append(res)
    all_res_np = np.stack(all_res).mean(axis=0)  # (N, 7)
    # print first row of a blank cell and all metric names
    print(
        "Method \t \u2193 MSE \t \u2193 LPIPS \t \u2193 FID \t \u2191 PSNR \t \u2191 SSIM \t \u2191 UIQI \t \u2193 FVD"  # noqa
    )
    for i, name in enumerate(names):
        print(f"{name}", end="\t")
        for j in range(all_res_np.shape[1]):
            print(f"{all_res_np[i, j]:.4f}", end="\t")
        print("", end="\n")
        np.save(f"{save_dir}/{name}.npy", all_res_np[i])


if __name__ == "__main__":
    main()
