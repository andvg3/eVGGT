<h1 align="center">
  <a href="https://robotwin-benchmark.github.io">Improving Robotic Manipulation with Efficient Geometry-Aware Vision Encoder<br></a>
</h1>
This repository is under active development and will be updated.


# Installation

Create a conda environment

```
$ conda create -n eVGGT python=3.10 -y
$ conda activate eVGGT
```

To install basic environments and CuRobo:

```
$ bash script/_install.sh
```

Download assets for RoboTwin:

```
$ bash script/_download_assets.sh
```

Install required packages:
```
$ cd policy/VGGT/vggt/
$ pip install -r requirements.txt
$ pip install -e . # Install VGGT dependency
cd ..
$ pip install -e . # Install diffusion_policy depenedency
$ cd ../..
```

For weights of eVGGT used for training policy, please download at [this link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/an_vuong_mbzuai_ac_ae/EdfMz2qQyP5LrcSt44rkr2UBtjSdMvIsieKf7STiRam0yQ?e=v6Q04Y). Place it at:

```
policy/VGGT/checkpoints/distillation
```

For more details of RoboTwin, follow [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).


# Training & Evaluation

Step 1: Collect data:
```
$ cd ../..
$ bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

- Step 2: Process data:
```
$ cd policy/VGGT
$ bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh beat_block_hammer demo_randomized 50
```

- Step 3: Train policy:
```
$ bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}
# bash train.sh beat_block_hammer demo_randomized 50 0 14 0
# For `aloha-agilex` embodiment, the action_dim is 14
```

- Step 4: Eval policy:
```
$ bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# bash eval.sh beat_block_hammer demo_randomized demo_randomized 50 0 0
# This command trains the policy using the `demo_randomized` setting ($ckpt_setting)
# and evaluates it using the same `demo_randomized` setting ($task_config).
#
# To evaluate a policy trained on the `demo_randomized` setting and tested on the `demo_clean` setting, run:
# bash eval.sh beat_block_hammer demo_clean demo_randomized 50 0 0
```


For other policies, please refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).


<!-- # Citations
If you find this work useful, please consider citing:

<b>RoboTwin 2.0</b>: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
``` -->


# Acknowledgement

Thanks to [Tianxing Chen et al.](https://tianxingchen.github.io) for their amazing RoboTwin platform.
