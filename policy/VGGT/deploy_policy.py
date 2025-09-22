import numpy as np
from .dp_model import DP
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def resize_and_preprocess_single_image(image_tensor, mode="crop", target_size=518):
    """
    Resize and preprocess a single torch image [3, H, W].

    Args:
        image_tensor (torch.Tensor): Tensor of shape [3, H, W], values in [0, 1].
        mode (str): Either "crop" or "pad".
        target_size (int): Target size for preprocessing.

    Returns:
        torch.Tensor: Tensor of shape [3, H', W'] ready for model input.
    """
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor)

    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError("Input must be of shape [3, H, W]")

    image_tensor = image_tensor.float()  # ensure float tensor

    _, H, W = image_tensor.shape

    # Resize logic
    if mode == "pad":
        if W >= H:
            new_W = target_size
            new_H = round(H * (new_W / W) / 14) * 14
        else:
            new_H = target_size
            new_W = round(W * (new_H / H) / 14) * 14
    else:  # mode == "crop"
        new_W = target_size
        new_H = round(H * (new_W / W) / 14) * 14

    # Resize
    img_resized = TF.resize(image_tensor, [new_H, new_W], interpolation=TF.InterpolationMode.BICUBIC)

    # Crop in 'crop' mode
    if mode == "crop" and new_H > target_size:
        start_y = (new_H - target_size) // 2
        img_resized = img_resized[:, start_y:start_y + target_size, :]

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

    return img_resized.cpu().numpy()


def encode_obs(observation):
    front_cam = (np.moveaxis(observation["observation"]["front_camera"]["rgb"], -1, 0) / 255)
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)

    front_cam = resize_and_preprocess_single_image(front_cam)
    head_cam = resize_and_preprocess_single_image(head_cam)
    left_cam = resize_and_preprocess_single_image(left_cam)
    right_cam = resize_and_preprocess_single_image(right_cam)

    obs = dict(
        front_cam=front_cam,
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    ckpt_file = f"./policy/VGGT/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    print(f"Loaded from {ckpt_file}")
    load_config_path = f'./policy/VGGT/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
