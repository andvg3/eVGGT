"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    # Head camera
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]

    # Front camera
    front_camera_type = cfg.front_camera_type
    front_camera_cfg = get_camera_config(front_camera_type)
    cfg.task.shape_meta.obs.front_cam.shape = [
        3,
        front_camera_cfg["h"],
        front_camera_cfg["w"],
    ]

    # Left camera
    left_camera_type = cfg.left_camera_type
    left_camera_cfg = get_camera_config(left_camera_type)
    cfg.task.shape_meta.obs.left_cam.shape = [
        3,
        left_camera_cfg["h"],
        left_camera_cfg["w"],
    ]

    # Right camera
    right_camera_type = cfg.right_camera_type
    right_camera_cfg = get_camera_config(right_camera_type)
    cfg.task.shape_meta.obs.right_cam.shape = [
        3,
        right_camera_cfg["h"],
        right_camera_cfg["w"],
    ]
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
