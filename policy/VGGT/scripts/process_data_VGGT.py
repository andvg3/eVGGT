import pickle, os
import numpy as np
import json
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]

        # Collect all 4 cameras
        all_cam_data = {}
        for cam in ["front_camera", "head_camera", "left_camera", "right_camera"]:
            cam_group = root[f"/observation/{cam}"]
            all_cam_data[cam] = dict(
                rgb=cam_group["rgb"][()],
                depth=cam_group["depth"][()],
                actor_segmentation=cam_group["actor_segmentation"][()],
                mesh_segmentation=cam_group["mesh_segmentation"][()],
                cam2world_gl=cam_group["cam2world_gl"][()],
                extrinsic_cv=cam_group["extrinsic_cv"][()],
                intrinsic_cv=cam_group["intrinsic_cv"][()],
            )

            # Collect point cloud data
            pointcloud = root["/pointcloud"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, all_cam_data, pointcloud


def save_all_cameras(camera_data, pointcloud, episode_idx, step_in_episode, save_base="data/scene"):
    """
    Save all four cameras' data at a given timestep under:
    data/scene/episode<episode_idx>/<step_in_episode>/
    """
    step_dir = os.path.join(save_base, f"episode{episode_idx}", str(step_in_episode))
    os.makedirs(step_dir, exist_ok=True)

    for cam in ["front_camera", "head_camera", "left_camera", "right_camera"]:
        cam_data = camera_data[cam]

        # Save RGB image
        rgb_bytes = cam_data["rgb"][step_in_episode]
        rgb_img = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(step_dir, f"{cam}.png"), rgb_img)

        # Save depth and segmentations
        np.save(os.path.join(step_dir, f"{cam}_depth.npy"), cam_data["depth"][step_in_episode])
        np.save(os.path.join(step_dir, f"{cam}_actor_seg.npy"), cam_data["actor_segmentation"][step_in_episode])
        np.save(os.path.join(step_dir, f"{cam}_mesh_seg.npy"), cam_data["mesh_segmentation"][step_in_episode])

        # Save camera matrices as YAML
        cam_info = {
            "cam2world_gl": cam_data["cam2world_gl"][step_in_episode].tolist(),
            "extrinsic_cv": cam_data["extrinsic_cv"][step_in_episode].tolist(),
            "intrinsic_cv": cam_data["intrinsic_cv"][step_in_episode].tolist()
        }
        with open(os.path.join(step_dir, f"{cam}_cam.yaml"), "w") as f:
            yaml.dump(cam_info, f, default_flow_style=False)

        np.save(os.path.join(step_dir, f"pointcloud.npy"), pointcloud[step_in_episode])


def save_instructions(instructions, episode_idx, save_base="data/scene"):
    """
    Save unseen instructions at a given timestep under:
    data/scene/episode<episode_idx>/<step_in_episode>/unseen_instructions.yaml
    """
    step_dir = os.path.join(save_base, f"episode{episode_idx}")
    os.makedirs(step_dir, exist_ok=True)

    instruction_path = os.path.join(step_dir, "unseen_instructions.yaml")
    with open(instruction_path, "w") as f:
        yaml.dump({"unseen": instructions}, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("task_name", type=str, help="The name of the task (e.g., beat_block_hammer)")
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process (e.g., 50)")
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = f"../../data/{task_name}/{task_config}"
    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays, state_arrays, joint_action_arrays = [], [], []
    episode_ends_arrays = []

    current_ep = 0
    total_count = 0
    step_idx = 0

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            camera_data_all,
            pointcloud_all
        ) = load_hdf5(load_path)

        instruction_path = os.path.join(load_dir, f"instructions/episode{current_ep}.json")
        with open(instruction_path, "r") as f:
            instruction_data = json.load(f)

        save_instructions(instruction_data.get("unseen", []), current_ep, save_base=f"data/{task_name}-{task_config}-{num}.zarr/scene")

        for j in range(0, left_gripper_all.shape[0]):
            joint_state = vector_all[j]
            if j != left_gripper_all.shape[0] - 1:
                head_rgb = camera_data_all["head_camera"]["rgb"][j]
                head_img = cv2.imdecode(np.frombuffer(head_rgb, np.uint8), cv2.IMREAD_COLOR)
                head_camera_arrays.append(head_img)
                state_arrays.append(joint_state)

                # Save all cameras at this timestep
                save_all_cameras(camera_data_all, pointcloud_all, current_ep, j, save_base=f"data/{task_name}-{task_config}-{num}.zarr/scene")

            if j != 0:
                joint_action_arrays.append(joint_state)

            step_idx += 1

        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    head_camera_arrays = np.moveaxis(np.array(head_camera_arrays), -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset("head_camera", data=head_camera_arrays, chunks=(100, *head_camera_arrays.shape[1:]), overwrite=True, compressor=compressor)
    zarr_data.create_dataset("state", data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("action", data=joint_action_arrays, chunks=(100, joint_action_arrays.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arrays, dtype="int64", overwrite=True, compressor=compressor)

if __name__ == "__main__":
    main()
