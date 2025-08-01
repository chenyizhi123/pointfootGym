"""
增强版RL控制器 - 支持Isaac Gym训练模型的部署
主要改进：
1. 扩展观测空间到33维，匹配Isaac Gym
2. 添加步态控制功能
3. 参数对齐Isaac Gym配置
"""

import os
import sys
import copy
import numpy as np
import yaml
import time
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

class PointfootControllerEnhanced:
    def __init__(self, model_dir, robot, robot_type, start_controller):
        # Initialize robot and type information
        self.robot = robot
        self.robot_type = robot_type
        self.is_point_foot = self.robot_type.startswith("PF")
        self.is_wheel_foot = self.robot_type.startswith("WF")
        self.is_sole_foot = self.robot_type.startswith("SF")

        # Load configuration and model file paths based on robot type
        self.config_file = f'{model_dir}/{self.robot_type}/params.yaml'
        self.model_file = f'{model_dir}/{self.robot_type}/policy/policy.onnx'

        # Load configuration settings from the YAML file
        self.load_config(self.config_file)

        # Load the ONNX model and set up input and output names
        self.policy_session = ort.InferenceSession(self.model_file)
        self.policy_input_names = [self.policy_session.get_inputs()[0].name]
        self.policy_output_names = [self.policy_session.get_outputs()[0].name]

        # Prepare robot command structure with default values for mode, q, dq, tau, Kp, Kd
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [self.control_cfg['stiffness'] for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [self.control_cfg['damping'] for x in range(0, self.joint_num)]

        # Prepare robot state structure
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]
        self.robot_state_tmp = copy.deepcopy(self.robot_state)

        # Initialize IMU (Inertial Measurement Unit) data structure
        self.imu_data = datatypes.ImuData()
        self.imu_data.quat[0] = 0
        self.imu_data.quat[1] = 0
        self.imu_data.quat[2] = 0
        self.imu_data.quat[3] = 1
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Set up a callback to receive updated robot state data
        self.robot_state_callback_partial = partial(self.robot_state_callback)
        self.robot.subscribeRobotState(self.robot_state_callback_partial)

        # Set up a callback to receive updated IMU data
        self.imu_data_callback_partial = partial(self.imu_data_callback)
        self.robot.subscribeImuData(self.imu_data_callback_partial)

        # Set up a callback to receive updated SensorJoy
        self.sensor_joy_callback_partial = partial(self.sensor_joy_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)

        # Set up a callback to receive diagnostic data
        self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)

        # Initialize the calibration state to -1, indicating no calibration has occurred.
        self.calibration_state = -1

        # Flag to start the controller
        self.start_controller = start_controller
        
        # 新增：步态控制相关初始化
        self.gait_indices = 0.0
        self.gaits = np.array([2.0, 0.5, 0.5, 0.05])  # [frequency, offset, duration, swing_height]
        self.clock_inputs_sin = 0.0
        self.clock_inputs_cos = 1.0
        self.desired_contact_states = np.zeros(2)  # 双足机器人
        self.dt = 1.0 / self.loop_frequency

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Assign configuration parameters to controller variables
        self.joint_names = config['PointfootCfg']['joint_names']
        self.init_state = config['PointfootCfg']['init_state']['default_joint_angle']
        self.stand_duration = config['PointfootCfg']['stand_mode']['stand_duration']
        self.control_cfg = config['PointfootCfg']['control']
        self.rl_cfg = config['PointfootCfg']['normalization']
        self.obs_scales = config['PointfootCfg']['normalization']['obs_scales']
        
        # 注意：这里需要更新为33维
        self.actions_size = config['PointfootCfg']['size']['actions_size']
        self.observations_size = 33  # 强制设为33维，匹配Isaac Gym
        
        self.imu_orientation_offset = np.array(list(config['PointfootCfg']['imu_orientation_offset'].values()))
        self.user_cmd_cfg = config['PointfootCfg']['user_cmd_scales']
        
        # 参数对齐Isaac Gym
        self.loop_frequency = 200  # 从500Hz改为200Hz
        self.control_cfg['decimation'] = 4  # 从10改为4
        self.control_cfg['damping'] = 1.5   # 从2.0改为1.5
        
        # Initialize variables for actions, observations, and commands
        self.actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(3)  # command to the robot (e.g., velocity, rotation)
        self.scaled_commands = np.zeros(3)
        self.base_lin_vel = np.zeros(3)  # base linear velocity
        self.base_position = np.zeros(3)  # robot base position
        self.loop_count = 0  # loop iteration count
        self.stand_percent = 0  # percentage of time the robot has spent in stand mode
        self.policy_session = None  # ONNX model session for policy inference
        self.joint_num = len(self.joint_names)  # number of joints
        self.commands = np.zeros(3)

        if self.is_wheel_foot:
          self.joint_pos_idxs = config['PointfootCfg']['size']['jointpos_idxs']
          self.wheel_joint_damping = config['PointfootCfg']['control']['wheel_joint_damping']
          self.wheel_joint_torque_limit = config['PointfootCfg']['control']['wheel_joint_torque_limit']

        # Initialize joint angles based on the initial configuration
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]
        
        # Set initial mode to "STAND"
        self.mode = "STAND"
        
        # 新增：步态参数配置
        self.gait_cfg = {
            'frequency_range': [1.5, 2.5],
            'offset': 0.5,  # 双足固定偏移
            'duration_range': [0.4, 0.6],
            'swing_height_range': [0.0, 0.1]
        }
        
        # 初始化步态参数
        self._init_gait_params()

    def _init_gait_params(self):
        """初始化步态参数"""
        self.gaits[0] = np.random.uniform(*self.gait_cfg['frequency_range'])  # frequency
        self.gaits[1] = self.gait_cfg['offset']  # offset
        self.gaits[2] = np.random.uniform(*self.gait_cfg['duration_range'])  # duration
        self.gaits[3] = np.random.uniform(*self.gait_cfg['swing_height_range'])  # swing_height
        
    def _update_gait_params(self):
        """更新步态参数，与Isaac Gym的_step_contact_targets对齐"""
        frequencies = self.gaits[0]
        self.gait_indices = (self.gait_indices + self.dt * frequencies) % 1.0
        
        # 计算时钟信号
        self.clock_inputs_sin = np.sin(2 * np.pi * self.gait_indices)
        self.clock_inputs_cos = np.cos(2 * np.pi * self.gait_indices)
        
        # 计算期望接触状态（简化版本，适配双足）
        offset = self.gaits[1]
        duration = self.gaits[2]
        
        # 左右脚的相位
        foot_phases = np.array([
            self.gait_indices,
            (self.gait_indices + offset) % 1.0
        ])
        
        # 简单的接触状态计算
        self.desired_contact_states = (foot_phases < duration).astype(float)
        
    def _resample_gaits(self):
        """定期重新采样步态参数"""
        if self.loop_count % (5 * self.loop_frequency) == 0:  # 每5秒重采样
            print("Resampling gait parameters...")
            self._init_gait_params()

    def compute_observation(self):
        # Convert IMU orientation from quaternion to Euler angles (ZYX convention)
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')  # Quaternion to Euler ZYX conversion
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()  # Get the inverse rotation matrix

        # Project the gravity vector (pointing downwards) into the body frame
        gravity_vector = np.array([0, 0, -1])  # Gravity in world frame (z-axis down)
        projected_gravity = np.dot(inverse_rot, gravity_vector)  # Transform gravity into body frame

        # Retrieve base angular velocity from the IMU data
        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        # Apply IMU orientation offset correction (using Euler angles)
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()  # Rotation matrix for offset correction
        base_ang_vel = np.dot(rot, base_ang_vel)  # Apply correction to angular velocity
        projected_gravity = np.dot(rot, projected_gravity)  # Apply correction to projected gravity

        # Retrieve joint positions and velocities from the robot state
        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)

        # Retrieve the last actions that were applied to the robot
        actions = np.array(self.last_actions)

        # Create a command scaler matrix for linear and angular velocities
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],  # Scale factor for linear velocity in x direction
            self.user_cmd_cfg['lin_vel_y'],  # Scale factor for linear velocity in y direction
            self.user_cmd_cfg['ang_vel_yaw']  # Scale factor for yaw (angular velocity)
        ])

        # Apply scaling to the command inputs (velocity commands)
        scaled_commands = np.dot(command_scaler, self.commands)

        # Populate observation vector
        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales['dof_pos']

        # In WF, joint pos does not include wheel speed, index(3, 7) needs to be removed
        if self.is_wheel_foot:
            joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs])
        else:
            joint_pos_input = joint_pos_value

        # 新增：更新步态参数
        self._update_gait_params()

        # 扩展观测空间到33维，匹配Isaac Gym
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],          # 3维: 基础角速度
            projected_gravity,                                   # 3维: 投影重力
            joint_pos_input,                                    # 6维: 关节位置
            joint_velocities * self.obs_scales['dof_vel'],     # 6维: 关节速度
            actions,                                            # 6维: 上一步动作
            scaled_commands,                                    # 3维: 缩放命令
            np.array([self.clock_inputs_sin]),                 # 1维: 步态时钟sin
            np.array([self.clock_inputs_cos]),                 # 1维: 步态时钟cos
            self.gaits                                          # 4维: 步态参数 [freq, offset, duration, swing_height]
        ])
        
        # 检查观测维度
        if len(obs) != 33:
            print(f"Warning: Observation size is {len(obs)}, expected 33")
            print(f"Components: ang_vel(3) + gravity(3) + joint_pos(6) + joint_vel(6) + actions(6) + commands(3) + clock(2) + gaits(4) = {3+3+6+6+6+3+2+4}")
        
        # Clip the observation values to within the specified limits for stability
        self.observations = np.clip(
            obs, 
            -self.rl_cfg['clip_scales']['clip_observations'],  # Lower limit for clipping
            self.rl_cfg['clip_scales']['clip_observations']  # Upper limit for clipping
        )

    def run(self):
        # Wait until the controller is started
        while not self.start_controller:
          time.sleep(1)

        # Initialize default joint angles for standing
        self.default_joint_angles = np.array([0.0] * len(self.joint_names))
        self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        self.mode = "STAND"
        self.loop_count = 0

        # Set the loop rate based on the frequency in the configuration
        rate = Rate(self.loop_frequency)
        print(f"Starting controller loop at {self.loop_frequency}Hz...")
        print(f"Observation size: {self.observations_size} dimensions")
        print(f"Gait parameters: {self.gaits}")
        
        while self.start_controller:
            self.update()
            # 新增：定期重采样步态参数
            self._resample_gaits()
            rate.sleep()
        
        # Reset robot command values to ensure a safe stop when exiting the loop
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [1.0 for x in range(0, self.joint_num)]
        self.robot.publishRobotCmd(self.robot_cmd)
        time.sleep(1)

    # 其他方法保持不变...
    def handle_stand_mode(self):
        if self.stand_percent < 1:
            for j in range(len(self.joint_names)):
                # Interpolate between initial and default joint angles during stand mode
                pos_des = self.default_joint_angles[j] * (1 - self.stand_percent) + self.init_state[self.joint_names[j]] * self.stand_percent
                self.set_joint_command(j, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])
            # Increment the stand percentage over time
            self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        else:
            # Switch to walk mode after standing
            self.mode = "WALK"

    def handle_walk_mode(self):
        # Update the temporary robot state and IMU data
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Execute actions every 'decimation' iterations
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_actions()
            # Clip the actions within predefined limits
            action_min = -self.rl_cfg['clip_scales']['clip_actions']
            action_max = self.rl_cfg['clip_scales']['clip_actions']
            self.actions = np.clip(self.actions, action_min, action_max)

        # Iterate over the joints and set commands based on actions
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)

        for i in range(len(joint_pos)):
            if self.is_point_foot or (i + 1) % 4 != 0:
                # Compute the limits for the action based on joint position and velocity
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])

                # Clip action within limits
                self.actions[i] = max(action_min / self.control_cfg['action_scale_pos'],
                                      min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))

                # Compute the desired joint position and set it
                pos_des = self.actions[i] * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])

                # Save the last action for reference
                self.last_actions[i] = self.actions[i]
            elif self.is_wheel_foot:
                action_min = joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                action_max = joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping
                self.last_actions[i] = self.actions[i]
                self.actions[i] = max(action_min / self.wheel_joint_damping,
                                      min(action_max / self.wheel_joint_damping, self.actions[i]))
                velocity_des = self.actions[i] * self.wheel_joint_damping
                self.set_joint_command(i, 0, velocity_des, 0, 0, self.wheel_joint_damping)

    def compute_actions(self):
        """
        Computes the actions based on the current observations using the policy session.
        """
        # Concatenate observations into a single tensor and convert to float32
        input_tensor = np.concatenate([self.observations], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        
        # 验证输入维度
        if len(input_tensor) != 33:
            print(f"Error: Input tensor size is {len(input_tensor)}, expected 33")
            # 如果维度不对，填充或截断到33维
            if len(input_tensor) < 33:
                input_tensor = np.pad(input_tensor, (0, 33 - len(input_tensor)), 'constant', constant_values=0)
            else:
                input_tensor = input_tensor[:33]
        
        # Create a dictionary of inputs for the policy session
        inputs = {self.policy_input_names[0]: input_tensor.reshape(1, -1)}  # 确保是2D
        
        # Run the policy session and get the output
        output = self.policy_session.run(self.policy_output_names, inputs)
        
        # Flatten the output and store it as actions
        self.actions = np.array(output).flatten()
        
    def set_joint_command(self, joint_index, q, dq, tau, kp, kd):
        """
        Sends a command to configure the state of a specific joint.
        """
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    def update(self):
        """
        Updates the robot's state based on the current mode and publishes the robot command.
        """
        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()
        
        # Increment the loop count
        self.loop_count += 1

        # Publish the robot command
        self.robot.publishRobotCmd(self.robot_cmd)
        
    # Callback functions (保持原样)
    def robot_state_callback(self, robot_state: datatypes.RobotState):
        self.robot_state = robot_state

    def imu_data_callback(self, imu_data: datatypes.ImuData):
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        
        # Rotate quaternion values
        self.imu_data.quat[0] = imu_data.quat[1]
        self.imu_data.quat[1] = imu_data.quat[2]
        self.imu_data.quat[2] = imu_data.quat[3]
        self.imu_data.quat[3] = imu_data.quat[0]

    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy):
        # 控制逻辑保持原样
        if not self.start_controller and self.calibration_state == 0 and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[3] == 1:
          print(f"L1 + Y: start_controller...")
          self.start_controller = True

        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
          print(f"L1 + X: stop_controller...")
          self.start_controller = False

        linear_x  = sensor_joy.axes[1]
        linear_y  = sensor_joy.axes[0]
        angular_z = sensor_joy.axes[2]

        linear_x  = 1.0 if linear_x > 1.0 else (-1.0 if linear_x < -1.0 else linear_x)
        linear_y  = 1.0 if linear_y > 1.0 else (-1.0 if linear_y < -1.0 else linear_y)
        angular_z = 1.0 if angular_z > 1.0 else (-1.0 if angular_z < -1.0 else angular_z)

        self.commands[0] = linear_x * 0.5
        self.commands[1] = linear_y * 0.5
        self.commands[2] = angular_z * 0.5

    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue):
      if diagnostic_value.name == "calibration":
        print(f"Calibration state: {diagnostic_value.code}")
        self.calibration_state = diagnostic_value.code

if __name__ == '__main__':
    # Get the robot type from the environment variable
    robot_type = os.getenv("ROBOT_TYPE")
    
    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    # Create a Robot instance of the specified type
    robot = Robot(RobotType.PointFoot)

    # Default IP address for the robot
    robot_ip = "127.0.0.1"
    
    # Check if command-line argument is provided for robot IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    # Initialize the robot with the provided IP address
    if not robot.init(robot_ip):
        sys.exit()

    # Determine if the simulation is running
    start_controller = robot_ip == "127.0.0.1"

    # Create and run the Enhanced PointfootController
    print("Starting Enhanced PointfootController with Isaac Gym compatibility...")
    controller = PointfootControllerEnhanced(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    controller.run()