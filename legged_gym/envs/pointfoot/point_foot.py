import os
import sys
from typing import Dict
import numpy as np

import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.terrain import Terrain


class PointFoot:
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg()
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_propriceptive_obs
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.proprioceptive_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self._include_feet_height_rewards = self._check_if_include_feet_height_rewards()
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def get_observations(self):
        return self.proprioceptive_obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.proprioceptive_obs_buf = torch.clip(self.proprioceptive_obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.proprioceptive_obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.measured_heights = self._get_heights()
        self._compute_feet_states()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _check_if_include_feet_height_rewards(self):
        members = [attr for attr in dir(self.cfg.rewards.scales) if not attr.startswith("__")]
        for scale in members:
            if "feet_height" in scale:
                return True
        return False

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample(env_ids)

        self._reset_buffers(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_target_distance"] = max(abs(self.command_ranges["target_x"][0]), abs(self.command_ranges["target_x"][1]))
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_buffers(self, env_ids):
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_feet_air_time[env_ids] = 0.
        self.current_max_feet_height[env_ids] = 0.
        self.last_max_feet_height[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # 重置步态相关缓冲区
        self.gait_indices[env_ids] = 0
        # 重置位置跟踪状态
        self.target_time_remaining[env_ids] = self.cfg.commands.target_timeout  # 重置为满时间
        self.is_target_reached[env_ids] = False  # 重置到达状态
        self.target_reached_time[env_ids] = 0.0  # 重置持续时间

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.compute_proprioceptive_observations()
        self.compute_privileged_observations()

        self._add_noise_to_obs()

    def _add_noise_to_obs(self):
        # add noise if needed
        if self.add_noise:
            obs_noise_vec, privileged_extra_obs_noise_vec = self.noise_scale_vec
            obs_noise_buf = (2 * torch.rand_like(self.proprioceptive_obs_buf) - 1) * obs_noise_vec
            self.proprioceptive_obs_buf += obs_noise_buf
            if self.num_privileged_obs is not None:
                privileged_extra_obs_buf = (2 * torch.rand_like(
                    self.privileged_obs_buf[:, len(self.noise_scale_vec[0]):]) - 1) * privileged_extra_obs_noise_vec
                self.privileged_obs_buf += torch.cat((obs_noise_buf, privileged_extra_obs_buf), dim=1)

    def compute_privileged_observations(self):
        if self.num_privileged_obs is not None:
            self._compose_privileged_obs_buf_no_height_measure()
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights_critic:
                self.privileged_obs_buf = self._add_height_measure_to_buf(self.privileged_obs_buf)
            if self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
                raise RuntimeError(
                    f"privileged_obs_buf size ({self.privileged_obs_buf.shape[1]}) does not match num_privileged_obs ({self.num_privileged_obs})")

    def _compose_privileged_obs_buf_no_height_measure(self):
        # 计算相对目标位置和归一化剩余时间（与proprioceptive观测保持一致）
        current_pos = self.root_states[:, :2]
        target_pos = self.commands[:, :2]
        relative_target_pos = target_pos - current_pos
        normalized_remaining_time = (self.target_time_remaining / self.cfg.commands.target_timeout).clamp(0, 1)
        
        # 组合命令观测：[相对位置x, 相对位置y, 归一化剩余时间]
        combined_commands = torch.cat([
            relative_target_pos * self.commands_scale,  # 2维：相对位置
            normalized_remaining_time.unsqueeze(1)      # 1维：剩余时间
        ], dim=1)
        
        self.privileged_obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                             self.projected_gravity,
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                             self.dof_vel * self.obs_scales.dof_vel,
                                             self.actions,
                                             combined_commands,  # 3维：相对位置+时间
                                             ), dim=-1)

    def compute_proprioceptive_observations(self):
        self._compose_proprioceptive_obs_buf_no_height_measure()
        if self.cfg.terrain.measure_heights_actor:
            self.proprioceptive_obs_buf = self._add_height_measure_to_buf(self.proprioceptive_obs_buf)
        if self.proprioceptive_obs_buf.shape[1] != self.num_obs:
            raise RuntimeError(
                f"obs_buf size ({self.proprioceptive_obs_buf.shape[1]}) does not match num_obs ({self.num_obs})")

    def _add_height_measure_to_buf(self, buf):
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                             1.) * self.obs_scales.height_measurements
        buf = torch.cat(
            (buf, heights), dim=-1
        )
        return buf

    def _compose_proprioceptive_obs_buf_no_height_measure(self):
        '''
        位置跟踪任务的观测组合
        
        为什么这样设计观测：
        1. 使用相对目标位置而非绝对位置 - 提高泛化能力
        2. 包含剩余时间信息 - 让机器人了解时间约束
        3. 保持其他原有观测 - 确保运动能力不受影响
        
        观测维度分析：
        - 相对目标位置比绝对位置更有意义，因为机器人只需知道往哪个方向走多远
        - 剩余时间让机器人能合理分配动作的紧急程度
        - 其他观测保持稳定，确保基本运动控制不受影响
        
        观测包含：
        - base angular velocity (3维)
        - projected gravity (3维)  
        - DOF position relative to default (8维)
        - DOF velocity (8维)
        - actions (8维)
        - relative target position (2维) - 相对目标位置
        - normalized remaining time (1维) - 归一化剩余时间
        - clock inputs (2维)
        - gait parameters (4维)
        '''
        # 计算相对目标位置（核心观测）
        current_pos = self.root_states[:, :2]  # 当前x,y位置
        target_pos = self.commands[:, :2]      # 目标x,y位置  
        relative_target_pos = target_pos - current_pos  # 相对目标位置向量
        
        # 归一化剩余时间 [0,1]，1表示刚开始，0表示即将超时
        normalized_remaining_time = (self.target_time_remaining / self.cfg.commands.target_timeout).clamp(0, 1)
        
        self.proprioceptive_obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,                    # 3维：角速度
            self.projected_gravity,                                         # 3维：重力方向
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 8维：关节位置
            self.dof_vel * self.obs_scales.dof_vel,                        # 8维：关节速度
            self.actions,                                                   # 8维：动作
            relative_target_pos * self.commands_scale,                     # 2维：相对目标位置（缩放后）
            normalized_remaining_time.unsqueeze(1),                        # 1维：归一化剩余时间
            self.clock_inputs_sin.view(self.num_envs, 1),                 # 1维：步态相位sin
            self.clock_inputs_cos.view(self.num_envs, 1),                 # 1维：步态相位cos  
            self.gaits,                                                     # 4维：步态参数
        ), dim=-1)  # 总维度：3+3+8+8+8+2+1+1+1+4 = 39维

    def create_sim(self):
        """ Creates simulation, terrain and environments
        You can modify this function to add your own terrain and environment settings.
        For example, you can add a custom terrain mixed with heightfield and trimesh.
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
            self.base_mass[env_id] = props[0].mass
        if self.cfg.domain_rand.randomize_base_com:
            com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
            props[0].com.x += np.random.uniform(-com_x, com_x)
            props[0].com.y += np.random.uniform(-com_y, com_y)
            props[0].com.z += np.random.uniform(-com_z, com_z)
        return props

    def _post_physics_step_callback(self):
        """ 位置跟踪任务的状态更新
        
        主要逻辑：
        1. 更新位置跟踪状态（到达判定、时间管理）
        2. 处理超时的环境（强制换新目标）  
        3. 按原计划重新采样目标
        4. 更新步态控制
        """
        # 更新位置跟踪状态
        self._update_position_tracking_state()
        
        # 处理按时间计划的目标重新采样
        scheduled_resample_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        
        # 处理超时的环境（强制换新目标）
        timeout_ids = (self.target_time_remaining <= 0).nonzero(as_tuple=False).flatten()
        
        # 处理持续到达足够久的环境（成功完成任务，换新目标）
        hold_success_ids = (self.target_reached_time >= self.cfg.commands.hold_time_threshold).nonzero(as_tuple=False).flatten()
        
        # 合并需要重新采样的环境ID
        all_resample_ids = torch.cat([scheduled_resample_ids, timeout_ids, hold_success_ids]).unique()
        self._resample(all_resample_ids)
        
        # 步态控制更新
        self._step_contact_targets()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
            
    def _update_position_tracking_state(self):
        """ 更新位置跟踪状态
        
        为什么需要这个函数：
        1. 实时判断是否到达目标，用于奖励计算
        2. 更新剩余时间，实现时间约束  
        3. 跟踪持续到达时间，奖励稳定性
        
        设计思路：
        - 到达判定基于欧氏距离
        - 时间倒计时确保任务效率
        - 持续时间累计奖励稳定保持
        """
        # 更新剩余时间（倒计时）
        self.target_time_remaining -= self.dt
        
        # 计算当前位置到目标的距离
        current_pos = self.root_states[:, :2]  # 当前x,y位置
        target_pos = self.commands[:, :2]      # 目标x,y位置
        distance_to_target = torch.norm(target_pos - current_pos, dim=1)
        
        # 判断是否到达目标
        currently_reached = distance_to_target < self.cfg.commands.reach_tolerance
        
        # 更新持续到达时间
        # 如果当前到达且之前也到达，累加时间；如果刚到达，开始计时；如果未到达，清零
        self.target_reached_time = torch.where(
            currently_reached & self.is_target_reached,
            self.target_reached_time + self.dt,  # 继续累加
            torch.where(
                currently_reached,
                torch.full_like(self.target_reached_time, self.dt),  # 开始计时
                torch.zeros_like(self.target_reached_time)  # 清零
            )
        )
        
        # 更新到达状态
        self.is_target_reached = currently_reached

    def _resample(self, env_ids):
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

    def _resample_commands(self, env_ids):
        """ 生成2D位置跟踪目标命令
        
        逻辑思路：
        1. 基于当前位置生成相对偏移的目标位置
        2. 确保目标距离在合理范围内(min_target_distance ~ max_target_distance)  
        3. 重置目标时间，开始新的倒计时
        
        为什么这样设计：
        - 相对偏移避免目标过于集中在某个区域
        - 距离限制确保任务既有挑战性又可完成
        - 时间重置给每个目标充足但有限的时间

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
            
        # 获取当前位置作为生成目标的基准
        current_pos_x = self.root_states[env_ids, 0]  # 当前x位置
        current_pos_y = self.root_states[env_ids, 1]  # 当前y位置
        
        # 生成相对偏移（在配置范围内）
        target_offset_x = torch_rand_float(
            self.command_ranges["target_x"][0], 
            self.command_ranges["target_x"][1], 
            (len(env_ids), 1), device=self.device).squeeze(1)
        target_offset_y = torch_rand_float(
            self.command_ranges["target_y"][0], 
            self.command_ranges["target_y"][1], 
            (len(env_ids), 1), device=self.device).squeeze(1)
        
        # 计算初步目标位置
        target_x = current_pos_x + target_offset_x
        target_y = current_pos_y + target_offset_y
        
        # 检查目标距离是否在合理范围内
        target_distance = torch.sqrt(target_offset_x**2 + target_offset_y**2)
        too_close = target_distance < self.cfg.commands.min_target_distance
        too_far = target_distance > self.cfg.commands.max_target_distance
        
        # 对距离不合理的目标重新生成
        invalid_mask = too_close | too_far
        if torch.any(invalid_mask):
            # 生成合理距离和随机角度
            n_invalid = invalid_mask.sum()
            angles = torch.rand(n_invalid, device=self.device) * 2 * 3.14159
            # 距离在min_target_distance到max_target_distance之间均匀分布
            distances = torch.rand(n_invalid, device=self.device) * \
                       (self.cfg.commands.max_target_distance - self.cfg.commands.min_target_distance) + \
                       self.cfg.commands.min_target_distance
            
            # 重新计算无效目标的位置
            target_x[invalid_mask] = current_pos_x[invalid_mask] + distances * torch.cos(angles)
            target_y[invalid_mask] = current_pos_y[invalid_mask] + distances * torch.sin(angles)
        
        # 设置命令（绝对目标位置）
        self.commands[env_ids, 0] = target_x  # 目标x坐标
        self.commands[env_ids, 1] = target_y  # 目标y坐标
        
        # 重置这些环境的目标时间（开始新的倒计时）
        self.target_time_remaining[env_ids] = self.cfg.commands.target_timeout

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (
                    actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """Random pushes the robots."""
        max_push_force = (
                self.base_mass.mean().item()
                * self.cfg.domain_rand.max_push_vel_xy
                / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )
        self.rigid_body_external_forces[:, 0, 0:3] = quat_rotate(
            self.base_quat, rigid_body_external_forces
        )
        self.rigid_body_external_forces[:, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ 位置跟踪任务的课程学习：根据成功率逐步增加目标距离

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # 基于位置跟踪和任务完成的成功率来调整难度
        if hasattr(self.reward_scales, "position_tracking") and len(env_ids) > 0:
            # 计算位置跟踪的平均奖励
            avg_position_reward = torch.mean(self.episode_sums["position_tracking"][env_ids]) / self.max_episode_length
            target_reward = 0.8 * self.reward_scales["position_tracking"]
            
            if avg_position_reward > target_reward:
                # 成功率高，增加难度：扩大目标距离范围
                current_max = max(abs(self.command_ranges["target_x"][0]), abs(self.command_ranges["target_x"][1]))
                new_max = min(current_max + 0.5, self.cfg.commands.max_curriculum_distance)
                
                self.command_ranges["target_x"] = [-new_max, new_max]
                self.command_ranges["target_y"] = [-new_max, new_max]

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        obs_noise_vec = torch.zeros(self.cfg.env.num_propriceptive_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        obs_noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        obs_noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        obs_noise_vec[6:9] = noise_scales.gravity * noise_level
        command_end_idx = 9 + self.cfg.commands.num_commands
        obs_noise_vec[9:command_end_idx] = 0.  # commands
        dof_pos_end_idx = command_end_idx + self.num_dof
        obs_noise_vec[command_end_idx:dof_pos_end_idx] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        dof_vel_end_idx = dof_pos_end_idx + self.num_dof
        obs_noise_vec[dof_pos_end_idx:dof_vel_end_idx] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        last_action_end_idx = dof_vel_end_idx + self.num_actions
        obs_noise_vec[dof_vel_end_idx:last_action_end_idx] = 0.  # previous actions
        if self.cfg.env.num_privileged_obs is not None:
            privileged_extra_obs_noise_vec = torch.zeros(
                self.cfg.env.num_privileged_obs - self.cfg.env.num_propriceptive_obs, device=self.device)
        else:
            privileged_extra_obs_noise_vec = None

        if self.cfg.terrain.measure_heights_actor:
            measure_heights_end_idx = last_action_end_idx + len(self.cfg.terrain.measured_points_x) * len(
                self.cfg.terrain.measured_points_y)
            obs_noise_vec[
            last_action_end_idx:measure_heights_end_idx] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.terrain.measure_heights_critic:
            if self.cfg.env.num_privileged_obs is not None:
                privileged_extra_obs_noise_vec[
                :len(self.cfg.terrain.measured_points_x) * len(
                    self.cfg.terrain.measured_points_y)] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return obs_noise_vec, privileged_extra_obs_noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.foot_positions = self.rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.last_foot_positions = torch.zeros_like(self.foot_positions)
        self.foot_heights = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.foot_velocities = torch.zeros_like(self.foot_positions)
        self.foot_velocities_f = torch.zeros_like(self.foot_positions)
        self.foot_relative_velocities = torch.zeros_like(self.foot_velocities)
        self.foot_quat = torch.zeros(self.num_envs, len(self.feet_indices), 4, dtype=torch.float, device=self.device)
        self.foot_ang_vel = torch.zeros_like(self.foot_positions)
        self.base_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.last_base_position = torch.zeros_like(self.base_position)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        # 位置跟踪命令：target_x, target_y (绝对坐标)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        # 位置命令的观测缩放（位置使用米为单位，不需要额外缩放）
        self.commands_scale = torch.tensor([1.0, 1.0],  # 保持原始的米单位
                                           device=self.device, requires_grad=False)
        
        # 位置跟踪状态管理
        self.target_time_remaining = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 剩余时间
        self.is_target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)     # 是否到达目标
        self.target_reached_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 到达目标的持续时间
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                        device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.last_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                device=self.device, requires_grad=False)
        self.current_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                   device=self.device, requires_grad=False)
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # 添加步态控制相关的缓冲区
        self.gaits = torch.zeros(
            self.num_envs,
            4,  # frequency, offset, duration, swing_height
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs_sin = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_inputs_cos = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        
        # 添加步态范围配置
        self.gaits_ranges = class_to_dict(self.cfg.gait.ranges)


    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights_actor and not self.terrain.cfg.measure_heights_critic:
            return
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_below_foot(self):
        """ Samples heights of the terrain at required points around each foot.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.feet_state[:, :, :2]

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_terrain_heights_from_points(self, points):
        points = points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights

    def _get_foot_heights(self):
        """Samples heights of the terrain at foot positions.
        Copied from base_task.py for foot height calculation.
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.foot_positions[:, :, :2] + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        return heights

    def _compute_feet_states(self):
        # Update feet state
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        
        # Compute foot quaternions and positions
        self.foot_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[
            :, self.feet_indices, 3:7
        ]
        self.foot_positions = self.rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        
        # Compute foot velocities
        self.foot_velocities = (
            self.foot_positions - self.last_foot_positions
        ) / self.dt

        # Compute foot angular velocities in local frame
        self.foot_ang_vel = self.rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 10:13]
        for i in range(len(self.feet_indices)):
            self.foot_ang_vel[:, i] = quat_rotate_inverse(
                self.foot_quat[:, i], self.foot_ang_vel[:, i]
            )
            self.foot_velocities_f[:, i] = quat_rotate_inverse(
                self.foot_quat[:, i], self.foot_velocities[:, i]
            )

        # Compute base position
        self.base_position = self.root_states[:, 0:3]
        
        # Compute foot relative velocities
        foot_relative_velocities = (
            self.foot_velocities
            - (self.base_position - self.last_base_position)
            .unsqueeze(1)
            .repeat(1, len(self.feet_indices), 1)
            / self.dt
        )
        for i in range(len(self.feet_indices)):
            self.foot_relative_velocities[:, i, :] = quat_rotate_inverse(
                self.base_quat, foot_relative_velocities[:, i, :]
            )
        
        # Compute foot heights
        if hasattr(self.cfg.asset, 'foot_radius'):
            foot_radius = self.cfg.asset.foot_radius
        else:
            foot_radius = 0.0
            
        ground_heights = self._get_foot_heights()
        self.foot_heights = torch.clip(
            (
                self.foot_positions[:, :, 2]
                - foot_radius
                - ground_heights), 0, 1)
        
        # Update last positions for next iteration
        self.last_foot_positions = self.foot_positions.clone()
        self.last_base_position = self.base_position.clone()
        
        # Original air time and contact logic
        self.last_feet_air_time = self.feet_air_time * self.first_contact + self.last_feet_air_time * ~self.first_contact
        self.feet_air_time *= ~self.contact_filt
        if self._include_feet_height_rewards:
            self.last_max_feet_height = self.current_max_feet_height * self.first_contact + self.last_max_feet_height * ~self.first_contact
            self.current_max_feet_height *= ~self.contact_filt
            self.feet_height = self.feet_state[:, :, 2] - self._get_heights_below_foot()
            self.current_max_feet_height = torch.max(self.current_max_feet_height,
                                                     self.feet_height)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt

    # ==================== 第一类：原有的基础Reward函数 ====================
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward steps between proper duration
        rew_airTime_below_min = torch.sum(
            torch.min(self.feet_air_time - self.cfg.rewards.min_feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime = rew_airTime_below_min + rew_airTime_above_max
        return rew_airTime

    def _reward_feet_distance(self):
        reward = 0
        for i in range(self.feet_state.shape[1] - 1):
            for j in range(i + 1, self.feet_state.shape[1]):
                feet_distance = torch.norm(
                    self.feet_state[:, i, :2] - self.feet_state[:, j, :2], dim=-1
                )
            reward += torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        return (~self.reset_buf).float()
    # ==================== 第二类：TODO中的跟踪类Reward函数 ====================


    # ------- TODO: add reward function for velocity tracking task -------
    '''
    After adding the reward functions you need in this file, you need to add the corresponding reward scales in the config file.
    For example, if you want to add a reward function for linear velocity tracking, you need to add the following to the config file:
    
    rewards:
      scales:
        tracking_lin_vel: 1.0
        tracking_ang_vel: 1.0
        base_height: 1.0
        tracking_base_height: 1.0
        orientation: 1.0

    '''
    # ========== 位置跟踪任务奖励函数 (替换速度跟踪) ==========
    def _reward_position_tracking(self):
        """ 位置跟踪奖励 - 核心奖励函数
        
        设计思路：
        - 基于到目标的欧氏距离
        - 使用exp函数，距离越近奖励越高
        - sigma控制奖励的陡峭程度
        
        为什么用exp函数：
        - 距离为0时奖励为1（最大）
        - 距离增大时奖励指数衰减
        - 提供平滑的梯度便于学习
        """
        current_pos = self.root_states[:, :2]
        target_pos = self.commands[:, :2]
        distance_error = torch.norm(target_pos - current_pos, dim=1)
        return torch.exp(-distance_error**2 / self.cfg.rewards.position_tracking_sigma)
    
    def _reward_target_reached(self):
        """ 到达目标奖励 - 成功奖励
        
        设计思路：
        - 当机器人在容忍范围内时给予大奖励
        - 鼓励机器人真正到达目标而不是接近
        - 简单的0/1奖励，清晰明确
        """
        return self.is_target_reached.float()
    
    def _reward_time_efficiency(self):
        """ 时间效率奖励 - 速度奖励
        
        设计思路：
        - 奖励快速到达目标的行为
        - 基于剩余时间的比例
        - 到达后剩余时间越多奖励越高
        
        计算逻辑：
        - 只有到达目标时才给奖励
        - 奖励 = (剩余时间 / 总时间) 如果到达目标
        """
        time_ratio = self.target_time_remaining / self.cfg.commands.target_timeout
        efficiency_reward = time_ratio / self.cfg.rewards.efficiency_time_scale
        # 只有到达目标时才给效率奖励
        return efficiency_reward * self.is_target_reached.float()
    
    def _reward_timeout_penalty(self):
        """ 超时惩罚 - 负向激励
        
        设计思路：
        - 当剩余时间<=0且未到达目标时给予惩罚
        - 强制机器人在时限内完成任务
        - 避免无效的徘徊行为
        """
        is_timeout = (self.target_time_remaining <= 0) & (~self.is_target_reached)
        return is_timeout.float()  # 返回正值，在配置中设置负权重
    
    def _reward_task_completion(self):
        """ 任务完成奖励 - 终极目标奖励
        
        设计思路：
        - 当机器人保持在目标位置足够久时给予大奖励
        - 这是一个"里程碑"奖励，标志着任务的真正完成
        - 只在即将换新目标的那一步给予，避免重复给奖励
        
        实现逻辑：
        - 检查是否刚好达到hold_time_threshold
        - 使用小的容忍范围判断"刚达到"
        - 给予一次性的大奖励
        """
        # 检查是否刚达到保持时间阈值（在一个dt的误差范围内）
        just_completed = (
            (self.target_reached_time >= self.cfg.commands.hold_time_threshold) &
            (self.target_reached_time <= self.cfg.commands.hold_time_threshold + self.dt * 2)  # 允许1-2个dt的误差
        )
        return just_completed.float()

    # def _reward_tracking_base_height(self):
    #     # Tracking the base height command
    #     if self.commands.shape[1] > 3:  # 假设第4个命令是目标高度
    #         base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #         height_error = torch.square(self.commands[:, 3] - base_height)
    #         return torch.exp(-height_error / self.cfg.rewards.tracking_sigma)
    #     else:
    #         return torch.zeros(self.num_envs, device=self.device)

    def _reward_tracking_contacts_shaped_force(self):
        """基于接触力的步态奖励"""
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(len(self.feet_indices)):
            reward += (1 - desired_contact[:, i]) * torch.exp(
                -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        """基于脚部速度的步态奖励"""
        foot_velocities = torch.norm(self.feet_state[:, :, 7:10], dim=-1)  # 线性速度
        desired_contact = self.desired_contact_states
        
        reward = 0
        for i in range(len(self.feet_indices)):
            reward += desired_contact[:, i] * torch.exp(
                -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
            )
        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_height(self):
        # Reward foot height based on gait phase
        desired_contact = self.desired_contact_states
        swing_height = self.gaits[:, 3]
        
        reward = 0
        for i in range(len(self.feet_indices)):
            # Swinging foot should be higher
            target_height = (1 - desired_contact[:, i]) * swing_height
            height_error = torch.square(self.foot_heights[:, i] - target_height)
            reward += torch.exp(-height_error / self.cfg.rewards.tracking_sigma)
        return reward / len(self.feet_indices)

    # ==================== 第三类：后来新加的Reward函数 ====================
    def _reward_action_smooth(self):
        # Penalize changes in actions
        if self.last_actions.dim() == 3:
            return torch.sum(
                torch.square(
                    self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)
        else:
            # Fallback for 2D last_actions
            return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_feet_contact_number(self):
        # Reward appropriate number of feet in contact (single foot support)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        return torch.sum(contact, dim=1)
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to limits
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)
    # ------------ 步态控制相关方法 ----------------
    def _step_contact_targets(self):
        """更新步态接触目标状态"""
        frequencies = self.gaits[:, 0]
        offsets = self.gaits[:, 1]
        durations = torch.cat(
            [
                self.gaits[:, 2].view(self.num_envs, 1),
                self.gaits[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )

        self.clock_inputs_sin = torch.sin(2 * torch.pi * self.gait_indices)
        self.clock_inputs_cos = torch.cos(2 * torch.pi * self.gait_indices)

        # von mises distribution
        kappa = 10.0  # 如果没有在config中定义，使用默认值
        if hasattr(self.cfg.rewards, 'kappa_gait_probs'):
            kappa = self.cfg.rewards.kappa_gait_probs
        
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        foot_indices = torch.remainder(
            torch.cat(
                [
                    self.gait_indices.view(self.num_envs, 1),
                    (self.gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
            0.5 / durations[stance_idxs]
        )
        foot_indices[swing_idxs] = 0.5 + (
            torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        self.desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (
            1 - smoothing_cdf_start(foot_indices - 1.5)
        )
        
    def _resample_gaits(self, env_ids):
        """重新采样步态参数"""
        if len(env_ids) == 0:
            return
        self.gaits[env_ids, 0] = torch_rand_float(
            self.gaits_ranges["frequencies"][0],
            self.gaits_ranges["frequencies"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = torch_rand_float(
            self.gaits_ranges["offsets"][0],
            self.gaits_ranges["offsets"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # 双足机器人简化为0.5偏移
        self.gaits[env_ids, 1] = 0.5

        self.gaits[env_ids, 2] = torch_rand_float(
            self.gaits_ranges["durations"][0],
            self.gaits_ranges["durations"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 3] = torch_rand_float(
            self.gaits_ranges["swing_height"][0],
            self.gaits_ranges["swing_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)