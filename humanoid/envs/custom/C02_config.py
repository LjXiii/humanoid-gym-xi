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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class C02Cfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15 # 策略网络（actor）保存的观察历史帧数量
        c_frame_stack = 3 # critic网络（价值函数）使用的观察历史帧数量
        num_single_obs = 47 # 单个时间步的观察向量维度
        num_observations = int(frame_stack * num_single_obs) # 机器人在单个时刻能够感知的所有必要信息
        single_num_privileged_obs = 73 # 单个时间步的特权观察向量维度
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs) # 提供仅在训练时可用的额外信息，帮助critic网络更准确评估状态价值
        num_actions = 12 # 
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions
                                  # 是否使用参考动作来辅助训练，当设为True时，会将预定义的参考动作（通常基于步态生成器）添加到策略输出的动作上，用作一种课程学习方法

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Casbot02/urdf/CASBOT02_LB.urdf'

        name = "legged_02" # 为机器人模型指定的名称，用于在仿真环境中识别机器人实例
        foot_name = "6_link" # "ankle_roll"链接（脚踝横滚关节）被指定为脚部
        knee_name = "4_link" # 指定膝关节的链接名称，用于特定的控制和观察计算

        terminate_after_contacts_on = ['base_link'] # 定义哪些部位接触会导致回合终止
        penalize_contacts_on = ["base_link"] # 定义哪些部位接触会受到惩罚（负奖励）
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter #控制自碰撞检测（机器人各部件之间的碰撞）
        flip_visual_attachments = False # 控制是否翻转视觉附件的方向，某些URDF文件可能需要这种调整以正确显示
        replace_cylinder_with_capsule = False # 决定是否将圆柱体碰撞几何体替换为胶囊体，胶囊体（有半球形端部的圆柱体）通常在物理仿真中表现更好，特别是处理碰撞时
        fix_base_link = False # 决定是否固定机器人的基座链接（躯干）如果设为True，机器人就变成了一个固定底座的操作臂，无法行走

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05 # 添加到关节位置观察的噪声量，相对低的值（0.05）表示关节位置测量相对准确
            dof_vel = 0.5 # 添加到关节速度观察的噪声量，明显高于位置噪声，这符合现实情况：速度通常是通过位置差分估计的，导致其噪声更高
            ang_vel = 0.1 # 添加到角速度（身体旋转速率）观察的噪声量，模拟陀螺仪传感器的噪声
            lin_vel = 0.05 # 添加到线速度（身体运动速度）观察的噪声量模拟速度估计中的噪声（可能来自IMU积分或视觉里程计）
            quat = 0.03 # 添加到四元数（身体姿态）观察的噪声量，较低的值表示姿态估计相对准确，这很重要，因为不准确的姿态估计会严重影响平衡
            height_measurements = 0.1 # 添加到高度测量的噪声量，可能用于模拟地形感知或身体高度估计的噪声

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.95] # 机器人基座中心在世界坐标系中的初始位置

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'leg_l1_joint': -5.16791005e-01, # leg pitch
            'leg_l2_joint': 0., # leg roll
            'leg_l3_joint': 0., # leg yaw
            'leg_l4_joint': 1.00567752e+00, # knee pitch 
            'leg_l5_joint': -4.88879716e-01, # ankle pitch
            'leg_l6_joint': 0., # ankle roll
            'leg_r1_joint': -5.16791005e-01, 
            'leg_r2_joint': 0., 
            'leg_r3_joint': 0.,
            'leg_r4_joint': 1.00567752e+00, 
            'leg_r5_joint': -4.88879716e-01, 
            'leg_r6_joint': 0.,
        }
    # 在这个配置中，所有关节的默认角度都设为0 --> 双腿垂直向下，膝盖完全伸直，脚踝使脚底与地面平行
    # 这是自然的站立姿势，但在实际训练中，可能会稍微调整这些值，比如微弯膝盖，以实现更稳定的初始状态

    class control(LeggedRobotCfg.control):
        # PD (Proportional-Derivative) 控制器用于将智能体的动作（目标关节角度）转换为关节扭矩
        # 公式：torque = P * (target_position - current_position) - D * current_velocity
        # PD Drive parameters:
        stiffness = {'leg_l1_joint': 350.0, 'leg_l2_joint': 200.0, 'leg_l3_joint': 200.0,
                     'leg_l4_joint': 350.0, 'leg_l5_joint': 15, 'leg_l6_joint': 15, 
                     'leg_r1_joint': 350.0, 'leg_r2_joint': 200.0, 'leg_r3_joint': 200.0,
                     'leg_r4_joint': 350.0, 'leg_r5_joint': 15, 'leg_r6_joint': 15}
        damping = {'leg_l1_joint': 10, 'leg_l2_joint': 10, 'leg_l3_joint': 10,
                     'leg_l4_joint': 10, 'leg_l5_joint': 10, 'leg_l6_joint': 10, 
                     'leg_r1_joint': 10, 'leg_r2_joint': 10, 'leg_r3_joint': 10,
                     'leg_r4_joint': 10, 'leg_r5_joint': 10, 'leg_r6_joint': 10}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25 # 0.25意味着动作空间被映射到±0.25弧度的范围（约±14.3度），更精细的控制，更平滑的运动，减少极端动作导致的不稳定性
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz 定义控制频率相对于仿真频率的比例，如果仿真运行在1000Hz（时间步长为0.001秒），控制器在每10个仿真步骤更新一次

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10 # 分配给PhysX引擎的CPU线程数，更多线程通常意味着更快的仿真速度，但有收益递减
            solver_type = 1  # 0: pgs, 1: tgs    物理求解器类型：1表示TGS(Temporal Gauss-Seidel)求解器，TGS相比PGS(Projected Gauss-Seidel)通常提供更好的稳定性，尤其是对于关节约束
            num_position_iterations = 4 # 位置约束求解迭代次数，影响关节和碰撞约束的精确性，更高的值提供更稳定、准确的结果，但需要更多计算
            num_velocity_iterations = 1 # 速度约束求解迭代次数，通常小于位置迭代次数，因为位置稳定性优先级更高
            contact_offset = 0.01  # [m] 碰撞检测的接触偏移距离（米）物体表面周围额外距离，在物体实际接触前碰撞检测就会触发，较大值可减少穿透但增加计算量
            rest_offset = 0.0   # [m] 物体静止时的偏移距离，设为0表示物体可以完全接触
            bounce_threshold_velocity = 0.1  # [m/s] 反弹阈值速度（米/秒），低于此速度的碰撞不会产生反弹效果，减少微小震荡
            max_depenetration_velocity = 1.0 # 最大去穿透速度，限制物体分离速度，防止爆炸性分离
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more，GPU处理的最大接触对数，大量并行环境时需要足够高的值
            default_buffer_size_multiplier = 5 
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2 # 接触信息收集模式：2表示在所有子步骤中收集接触数据，提供最详细的接触信息，对于接触驱动的学习很重要

    class domain_rand:
        randomize_friction = True # 启用地面摩擦系数随机化
        friction_range = [0.1, 2.0] # 摩擦系数范围，从非常光滑(0.1)到非常粗糙(2.0)
        randomize_base_mass = True # 启用基座质量随机化
        added_mass_range = [-5., 5.] # 在基本质量上增减-5kg到+5kg，模拟机器人携带不同负载的情况
        push_robots = True # 启用随机推力干扰
        push_interval_s = 4 # 每4秒推一次
        max_push_vel_xy = 0.2 # 最大水平推力速度（米/秒）
        max_push_ang_vel = 0.4 # 最大角推力速度（弧度/秒）
        # dynamic randomization 模拟真实机器人控制系统中的延迟和噪声
        action_delay = 0.5 # 行动延迟系数（按概率应用）
        action_noise = 0.02 # 动作噪声幅度

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4 # 命令向量包含4个元素：前后速度(x)、左右速度(y)、转向速度(yaw)和目标朝向(heading)
        resampling_time = 8.  # time before command are changed[s] 每8秒随机生成一次新命令，给机器人足够时间适应和执行每个命令
        heading_command = True  # if true: compute ang vel command from heading error
                                # 使用目标朝向命令而非直接角速度，系统会从当前朝向和目标朝向计算所需角速度

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.9244 # 理想的机器人基座高度（米）0.89
        min_dist = 0.2 # 最小足部/膝盖间距（米）
        max_dist = 0.5 # 最大足部间距（米）
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m 理想的足部抬高高度（米）
        cycle_time = 0.64                # sec 步态周期时间（秒）
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True # 将负总奖励限制为零，避免过早终止问题
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5 # 跟踪误差指数衰减系数，控制奖励曲线的陡峭程度
        max_contact_force = 700  # Forces above this value are penalized 最大接触力

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        # 不同类型的观察值乘以不同的缩放因子，将所有观察值统一到相似的数值范围，有助于神经网络学习
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18. # 将观察值限制在±18范围内
        clip_actions = 18. # 将动作值限制在±18范围内
        # 防止极端值影响训练稳


class C02CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration  每次迭代每个环境收集的步骤数
        max_iterations = 3001  # number of policy updates 每个更新周期的小批量数量

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations. 每100次迭代保存一次模型
        experiment_name = '02Bot_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
