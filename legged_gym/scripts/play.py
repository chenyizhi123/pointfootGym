# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from export_policy_as_onnx import *

import numpy as np
import torch

# 修改说明：
# 1. 从速度跟踪改为位置跟踪：命令从3D速度(vx,vy,vyaw)改为2D位置(target_x,target_y)
# 2. 添加了位置跟踪相关的状态监控：距离、剩余时间、到达状态等
# 3. 实现了动态目标设置：到达当前目标后自动生成新的随机目标


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 位置跟踪测试设置
    # 禁用自动重采样，手动设置固定的位置目标来测试策略
    env.cfg.commands.resampling_time = 1000.0  # 设置很长时间避免重采样
    
    # 手动设置位置目标来测试策略的位置跟踪能力
    # 设置一个中等距离的目标位置
    current_pos = env.root_states[:, :2].clone()  # 获取当前位置
    env.commands[:, 0] = current_pos[:, 0] + 2.0  # 目标X位置：当前位置 + 2米
    env.commands[:, 1] = current_pos[:, 1] + 1.0  # 目标Y位置：当前位置 + 1米
    
    # 重置时间相关状态
    env.target_time_remaining[:] = env.cfg.commands.target_timeout
    env.is_target_reached[:] = False
    env.target_reached_time[:] = 0.0
    
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        export_policy_as_onnx(args)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        # 定期更新位置命令确保不被重采样覆盖，并在到达目标后设置新目标
        if i % 100 == 0:  # 每100步检查一次（约每5秒@20Hz）
            current_pos = env.root_states[:, :2]
            target_pos = env.commands[:, :2]
            distance_to_target = torch.norm(target_pos - current_pos, dim=1)
            
            # 如果到达目标（距离<0.2米），设置新的随机目标
            reached_mask = distance_to_target < 0.2
            if torch.any(reached_mask):
                # 为到达目标的环境设置新的随机目标
                n_reached = reached_mask.sum().item()
                new_targets_x = current_pos[reached_mask, 0] + torch.rand(n_reached, device=env.device) * 4.0 - 2.0  # [-2, 2]米偏移
                new_targets_y = current_pos[reached_mask, 1] + torch.rand(n_reached, device=env.device) * 4.0 - 2.0  # [-2, 2]米偏移
                
                env.commands[reached_mask, 0] = new_targets_x
                env.commands[reached_mask, 1] = new_targets_y
                env.target_time_remaining[reached_mask] = env.cfg.commands.target_timeout
                env.is_target_reached[reached_mask] = False
                env.target_reached_time[reached_mask] = 0.0
        
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            # 计算当前位置和目标的相关信息
            current_pos_x = env.root_states[robot_index, 0].item()
            current_pos_y = env.root_states[robot_index, 1].item()
            target_pos_x = env.commands[robot_index, 0].item()
            target_pos_y = env.commands[robot_index, 1].item()
            distance_to_target = np.sqrt((target_pos_x - current_pos_x)**2 + (target_pos_y - current_pos_y)**2)
            
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    # 位置跟踪相关状态
                    'target_pos_x': target_pos_x,
                    'target_pos_y': target_pos_y,
                    'current_pos_x': current_pos_x,
                    'current_pos_y': current_pos_y,
                    'distance_to_target': distance_to_target,
                    'time_remaining': env.target_time_remaining[robot_index].item(),
                    'is_target_reached': env.is_target_reached[robot_index].item(),
                    'target_reached_time': env.target_reached_time[robot_index].item(),
                    # 机器人运动状态
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
