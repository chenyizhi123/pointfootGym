from legged_gym.envs.base.base_config import BaseConfig

class PointFootRoughCfg(BaseConfig):
    class env:
        num_envs = 8192
        num_propriceptive_obs = 32  # 简化观测: 3(ang_vel) + 3(gravity) + 6(dof_pos) + 6(dof_vel) + 6(actions) + 2(target_pos) + 1(clock_sin) + 1(clock_cos) + 4(gaits) = 32
        num_privileged_obs = 153  # privileged观测: 32(基础观测) + 121(height_measurements 11x11) = 153
        num_actions = 6
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.6
        restitution = 0.8
        # rough terrain only:
        measure_heights_actor = False
        measure_heights_critic = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4,
                             0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = True
        num_commands = 2  # target_x, target_y (位置指令)
        resampling_time = 5.0  # time before command are changed[s]
        
        # 位置跟踪等级系统
        position_curriculum = True  # 启用位置跟踪等级系统
        max_init_position_level = 3  # 起始位置跟踪等级 (0-9)
        num_position_levels = 10  # 总共10个位置跟踪等级
        position_level_ranges = [
            # 每个等级对应的 [x_min, x_max, y_min, y_max] 范围
            [-1.0, 1.0, -1.0, 1.0],    # 等级0: ±1m
            [-1.5, 1.5, -1.5, 1.5],    # 等级1: ±1.5m
            [-2.0, 2.0, -2.0, 2.0],    # 等级2: ±2m
            [-2.5, 2.5, -2.5, 2.5],    # 等级3: ±2.5m
            [-3.0, 3.0, -3.0, 3.0],    # 等级4: ±3m
            [-4.0, 4.0, -4.0, 4.0],    # 等级5: ±4m
            [-5.0, 5.0, -5.0, 5.0],    # 等级6: ±5m
            [-6.0, 6.0, -6.0, 6.0],    # 等级7: ±6m
            [-7.5, 7.5, -7.5, 7.5],    # 等级8: ±7.5m
            [-10.0, 10.0, -10.0, 10.0] # 等级9: ±10m
        ]
        
        class ranges:
            target_x = [-5.0, 5.0]  # 目标X坐标范围[m]
            target_y = [-5.0, 5.0]  # 目标Y坐标范围[m]

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]
        
        class ranges:
            frequencies = [1.5, 2.5]
            offsets = [0, 1]  # offset is hard to learn
            durations = [0.5, 0.5]  # small durations(<0.4) is hard to learn
            swing_height = [0.0, 0.1]

    class init_state:
        pos = [0.0, 0.0, 0.62]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {
            "abad_L_Joint": 40,
            "hip_L_Joint": 40,
            "knee_L_Joint": 40,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 40,
            "hip_R_Joint": 40,
            "knee_R_Joint": 40,
            "foot_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,
            "foot_R_Joint": 0.0,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]

    class asset:
        import os
        import sys
        robot_type = os.getenv("ROBOT_TYPE")

        # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
        if not robot_type:
            print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
            sys.exit(1)
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pointfoot/' + robot_type + '/urdf/robot.urdf'
        name = robot_type
        foot_name = 'foot'
        terminate_after_contacts_on = ["abad", "base"]
        penalize_contacts_on = ["base", "abad", "hip", "knee"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        foot_radius = 0.03  # radius of the foot sphere for height calculation

    class domain_rand:
        randomize_friction = True
        friction_range = [0.0, 1.6]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        push_robots = True
        push_interval_s = 5 # time between pushes [s]
        max_push_vel_xy = 1.  

    class rewards:
        class scales:
            # termination related rewards
            keep_balance = 1.0

            # 位置跟踪相关奖励
            tracking_position = 2.0      # 位置跟踪主要奖励
            arrival_bonus = 1.0          # 到达目标奖励
            approach_efficiency = 0.5    # 朝向目标运动奖励
            tracking_time = 0.5          # 追踪时间奖励（在目标附近停留）

            # regulation related rewards
            base_height = -1.0  # 减弱高度惩罚，允许更灵活的运动 (原来-2.0)
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            torques = -0.00002
            dof_acc = -2.5e-7
            action_rate = -0.006
            dof_pos_limits = -2.0
            collision = -1  # 减弱碰撞惩罚，允许更灵活的运动 (原来-2.5)
            action_smooth = -0.001
            orientation = -5  # 减弱姿态惩罚，避免过度限制运动 (原来-8.0)
            feet_distance = -100
            feet_regulation = -0.00001 # 减弱脚部调节奖励 (原来-0.05)
            # foot_landing_vel = -0.1 # 减弱脚部着陆速度奖励 (原来-0.15)
            tracking_contacts_shaped_force = -0.2
            tracking_contacts_shaped_vel = -0.2
            
            # additional reward functions (set to 0 if not needed)
            feet_air_time = 0.0
            torque_limits = 0.0
            survival = 0.0
            feet_swing = 0.0
            # tracking_contacts_shaped_height = 0.0
            feet_contact_number = 0.0
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        
        # 位置跟踪相关参数
        position_tracking_sigma = 1.0    # 位置跟踪的容忍度，tracking reward = exp(-distance_error/sigma)
        arrival_threshold = 0.5          # 到达判定距离[m]
        arrival_bonus_value = 10.0       # 到达奖励值
        
        # 原有参数保持
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.68  # 0.58
        feet_height_target = 0.10
        min_feet_distance = 0.115
        about_landing_threshold = 0.08
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005
        min_feet_air_time = 0.25
        max_feet_air_time = 0.65
        # Mixed reward weights - no longer used since we removed filtering
        # filtered_weight = 0.7     # Weight for filtered velocity (stability)  
        # real_weight = 0.3         # Weight for real velocity (evaluation alignment)

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            # 位置跟踪相关的观测标准化
            position = 0.2           # 目标位置缩放 (5m范围 -> 1.0)

        clip_observations = 100.
        clip_actions = 100.
        # filter_weight = 0.05  # No longer used - removed filtering for direct evaluation alignment

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class PointFootRoughCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-5  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 100000  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = 'pointfoot_rough'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
