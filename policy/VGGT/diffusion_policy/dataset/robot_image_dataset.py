from typing import Dict
import numba
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            # keys=['head_camera', 'front_camera', 'state', 'action'],
            # keys=["head_camera", "state", "action"],
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        # front_cam = np.moveaxis(sample['front_camera'],-1,1)/255
        # left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        # right_cam = np.moveaxis(sample['right_camera'],-1,1)/255

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                # 'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0

        head_cam = resize_and_preprocess_torch_images(head_cam)
        front_cam = resize_and_preprocess_torch_images(front_cam)
        left_cam = resize_and_preprocess_torch_images(left_cam)
        right_cam = resize_and_preprocess_torch_images(right_cam)

        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                'front_cam': front_cam, # B, T, 3, H, W
                'left_cam': left_cam, # B, T, 3, H, W
                'right_cam': right_cam, # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
            },
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)


def resize_and_preprocess_torch_images(image_tensor, mode="crop", target_size=518):
    """
    Resize and preprocess torch images for model input.
    
    Args:
        image_tensor (torch.Tensor): Tensor of shape [B, S, 3, H, W], values in [0, 1].
        mode (str): Either "crop" or "pad".
        target_size (int): Target size for preprocessing.
    
    Returns:
        torch.Tensor: Tensor of shape [B, S, 3, H', W'] ready for model input.
    """
    if image_tensor.ndim != 5:
        raise ValueError("Input must be of shape [B, S, 3, H, W]")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'.")

    B, S, C, H, W = image_tensor.shape
    images_out = []
    shapes = set()

    # Process each image individually
    for b in range(B):
        seq_out = []
        for s in range(S):
            img_tensor = image_tensor[b, s]  # [3, H, W]

            _, H_i, W_i = img_tensor.shape

            # Resize logic
            if mode == "pad":
                if W_i >= H_i:
                    new_W = target_size
                    new_H = round(H_i * (new_W / W_i) / 14) * 14
                else:
                    new_H = target_size
                    new_W = round(W_i * (new_H / H_i) / 14) * 14
            else:  # mode == "crop"
                new_W = target_size
                new_H = round(H_i * (new_W / W_i) / 14) * 14

            # Resize
            img_resized = TF.resize(img_tensor, [new_H, new_W], interpolation=TF.InterpolationMode.BICUBIC)

            # Crop in 'crop' mode
            if mode == "crop" and new_H > target_size:
                start_y = (new_H - target_size) // 2
                img_resized = img_resized[:, start_y : start_y + target_size, :]

            # Pad in 'pad' mode
            if mode == "pad":
                pad_h = target_size - img_resized.shape[1]
                pad_w = target_size - img_resized.shape[2]
                if pad_h > 0 or pad_w > 0:
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    img_resized = F.pad(
                        img_resized,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=1.0,
                    )

            shapes.add((img_resized.shape[1], img_resized.shape[2]))
            seq_out.append(img_resized)
        images_out.append(torch.stack(seq_out))  # [S, 3, H', W']

    # Align shapes if they differ across images
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_H = max(h for h, w in shapes)
        max_W = max(w for h, w in shapes)

        padded_out = []
        for seq in images_out:
            padded_seq = []
            for img in seq:
                pad_h = max_H - img.shape[1]
                pad_w = max_W - img.shape[2]
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
                padded_seq.append(img)
            padded_out.append(torch.stack(padded_seq))
        images_out = padded_out

    # Stack back to [B, S, 3, H', W']
    images_out = torch.stack(images_out)

    return images_out
