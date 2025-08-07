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
    
    # 位置跟踪任务设置
    # 禁用重采样以便手动控制目标位置
    env.cfg.commands.resampling_time = 1000.0  # 设置很长时间避免重采样
    env.cfg.commands.min_norm = 0.0  # 允许小目标（不自动置零）
    
    # 手动设置初始位置偏移来测试策略
    env.commands[:, 0] = 3.0  # X方向偏移 3.0 m (相对于当前位置)
    env.commands[:, 1] = 2.0  # Y方向偏移 2.0 m (相对于当前位置)
    
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
        # 动态改变位置偏移来测试跟踪能力
        if i % 400 == 0:  # 每400步（约20秒@20Hz）改变偏移指令
            if i == 0:
                env.commands[:, 0] = 3.0  # 第一个偏移：向前右移动
                env.commands[:, 1] = 2.0
            elif i == 400:
                env.commands[:, 0] = -2.0  # 第二个偏移：向后左移动  
                env.commands[:, 1] = -3.0
            elif i == 800:
                env.commands[:, 0] = 1.0   # 第三个偏移：小幅右前移动
                env.commands[:, 1] = 1.0
            elif i == 1200:
                env.commands[:, 0] = -1.0   # 第四个偏移：小幅左后移动
                env.commands[:, 1] = -1.0
        
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # 每5秒打印一次位置跟踪状态
        if i % 100 == 0:
            current_offset = env.base_position[robot_index, :2] - env.command_base_position[robot_index, :2]
            target_offset = env.commands[robot_index, :2]
            distance = torch.norm(target_offset - current_offset).item()
            print(f"Step {i}: Target_offset=({target_offset[0].item():.2f}, {target_offset[1].item():.2f}), "
                  f"Current_offset=({current_offset[0].item():.2f}, {current_offset[1].item():.2f}), "
                  f"Distance={distance:.2f}m")
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            # 计算位置跟踪相关的指标
            current_offset = env.base_position[robot_index, :2] - env.command_base_position[robot_index, :2]
            target_offset = env.commands[robot_index, :2]
            distance_to_target = torch.norm(target_offset - current_offset).item()
            
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    # 位置跟踪相关
                    'target_offset_x': target_offset[0].item(),
                    'target_offset_y': target_offset[1].item(),
                    'current_offset_x': current_offset[0].item(),
                    'current_offset_y': current_offset[1].item(),
                    'distance_to_target': distance_to_target,
                    'base_pos_abs_x': env.base_position[robot_index, 0].item(),
                    'base_pos_abs_y': env.base_position[robot_index, 1].item(),
                    # 速度信息
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
