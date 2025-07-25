#!/usr/bin/env python3
"""
PointFoot Model Evaluation Script
专门用于评估训练好的模型在四个评分标准上的表现：
1. 追踪指令精度 (Tracking Precision)
2. 地形跨越性能 (Terrain Traversal) 
3. 行走稳定性 (Walking Stability)
4. 抗外力干扰能力 (External Force Resistance)
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import json

class PointFootEvaluator:
    def __init__(self, args):
        self.args = args
        self.results = {}
        self.detailed_logs = defaultdict(list)
        
        # 加载环境和模型
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=args.task)
        self._setup_evaluation_config()
        
        # 创建环境和加载策略
        self.env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=self.env_cfg)
        self.train_cfg.runner.resume = True
        ppo_runner, _ = task_registry.make_alg_runner(env=self.env, name=args.task, args=args, train_cfg=self.train_cfg)
        self.policy = ppo_runner.get_inference_policy(device=self.env.device)
        
        print(f"🤖 Loaded model: {self.train_cfg.runner.experiment_name}")
        print(f"📊 Environment: {args.task} with {self.env.num_envs} environments")
    
    def _setup_evaluation_config(self):
        """配置评估环境参数"""
        # 使用较少的环境数量以便详细观察
        self.env_cfg.env.num_envs = min(self.env_cfg.env.num_envs, 100)
        
        # 禁用训练时的随机化，确保可重复的评估
        self.env_cfg.noise.add_noise = False
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.domain_rand.randomize_base_mass = False
        self.env_cfg.domain_rand.randomize_base_com = False
        
        # 设置较长的episode长度以充分评估
        self.env_cfg.env.episode_length_s = 30  # 30秒episode
        
        print("✅ Evaluation environment configured")
    
    def evaluate_tracking_precision(self, num_episodes=5):
        """评估1: 追踪指令精度"""
        print("\n🎯 评估1: 追踪指令精度 (Tracking Precision)")
        print("=" * 60)
        
        # 禁用推力和复杂地形，专注于追踪能力
        self.env_cfg.domain_rand.push_robots = False
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]  # 只用平坦地形
        
        # 重新创建环境
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        
        tracking_errors = []
        velocity_commands = []
        actual_velocities = []
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            
            obs = env.reset()
            episode_tracking_errors = []
            episode_commands = []
            episode_velocities = []
            
            for step in range(int(env.max_episode_length)):
                with torch.no_grad():
                    actions = self.policy(obs)
                
                obs, _, _, dones, _ = env.step(actions)
                
                # 计算追踪误差
                lin_vel_error = torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1)
                ang_vel_error = torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2])
                total_error = lin_vel_error + ang_vel_error
                
                episode_tracking_errors.append(total_error.cpu().numpy())
                episode_commands.append(env.commands[:, :3].cpu().numpy())
                episode_velocities.append(torch.cat([env.base_lin_vel[:, :2], env.base_ang_vel[:, 2:3]], dim=1).cpu().numpy())
                
                if step % 100 == 0:
                    mean_error = total_error.mean().item()
                    print(f"    Step {step}: Mean tracking error = {mean_error:.3f}")
            
            tracking_errors.extend(episode_tracking_errors)
            velocity_commands.extend(episode_commands)
            actual_velocities.extend(episode_velocities)
        
        # 分析结果
        all_errors = np.concatenate(tracking_errors)
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        success_rate = np.mean(all_errors < 0.5)  # 误差小于0.5算成功
        
        self.results['tracking_precision'] = {
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'success_rate': float(success_rate),
            'score': max(0, 100 * (1 - mean_error))  # 0-100分
        }
        
        print(f"  📊 平均追踪误差: {mean_error:.3f} ± {std_error:.3f}")
        print(f"  ✅ 成功率 (误差<0.5): {success_rate:.1%}")
        print(f"  🏆 追踪精度得分: {self.results['tracking_precision']['score']:.1f}/100")
        
        return env
    
    def evaluate_terrain_traversal(self, num_episodes=5):
        """评估2: 地形跨越性能"""
        print("\n🏔️ 评估2: 地形跨越性能 (Terrain Traversal)")
        print("=" * 60)
        
        # 启用复杂地形，禁用推力
        self.env_cfg.domain_rand.push_robots = False
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.terrain.terrain_proportions = [0.0, 0.2, 0.3, 0.3, 0.2]  # 复杂地形
        
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        
        traversal_distances = []
        terrain_levels = []
        success_episodes = 0
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            
            obs = env.reset()
            initial_positions = env.root_states[:, :2].clone()
            episode_distances = []
            
            for step in range(int(env.max_episode_length)):
                with torch.no_grad():
                    actions = self.policy(obs)
                
                obs, _, _, dones, _ = env.step(actions)
                
                # 计算行走距离
                current_positions = env.root_states[:, :2]
                distances = torch.norm(current_positions - initial_positions, dim=1)
                episode_distances.append(distances.cpu().numpy())
                
                if step % 100 == 0:
                    mean_distance = distances.mean().item()
                    print(f"    Step {step}: Mean distance = {mean_distance:.2f}m")
            
            # 分析episode结果
            final_distances = episode_distances[-1]
            traversal_distances.extend(final_distances)
            
            # 记录地形级别
            if hasattr(env, 'terrain_levels'):
                terrain_levels.extend(env.terrain_levels.cpu().numpy())
            
            # 成功标准：平均走过4米以上
            if np.mean(final_distances) > 4.0:
                success_episodes += 1
        
        # 分析结果
        mean_distance = np.mean(traversal_distances)
        std_distance = np.std(traversal_distances)
        success_rate = success_episodes / num_episodes
        
        self.results['terrain_traversal'] = {
            'mean_distance': float(mean_distance),
            'std_distance': float(std_distance),
            'success_rate': float(success_rate),
            'score': min(100, 100 * mean_distance / 8.0)  # 8米满分
        }
        
        print(f"  📊 平均穿越距离: {mean_distance:.2f} ± {std_distance:.2f} 米")
        print(f"  ✅ 成功率 (>4米): {success_rate:.1%}")
        print(f"  🏆 地形穿越得分: {self.results['terrain_traversal']['score']:.1f}/100")
        
        return env
    
    def evaluate_walking_stability(self, num_episodes=5):
        """评估3: 行走稳定性"""
        print("\n⚖️ 评估3: 行走稳定性 (Walking Stability)")
        print("=" * 60)
        
        # 中等难度地形，无推力
        self.env_cfg.domain_rand.push_robots = False
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.terrain.terrain_proportions = [0.3, 0.4, 0.2, 0.1, 0.0]  # 中等难度
        
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        
        stability_metrics = []
        fall_counts = []
        orientation_errors = []
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            
            obs = env.reset()
            episode_falls = 0
            episode_orientations = []
            
            for step in range(int(env.max_episode_length)):
                with torch.no_grad():
                    actions = self.policy(obs)
                
                obs, _, _, dones, _ = env.step(actions)
                
                # 记录跌倒次数
                current_falls = torch.sum(env.reset_buf).item()
                episode_falls += current_falls
                
                # 记录姿态稳定性 (roll, pitch角度)
                orientation_error = torch.norm(env.projected_gravity[:, :2], dim=1)
                episode_orientations.append(orientation_error.cpu().numpy())
                
                if step % 100 == 0:
                    mean_orientation = orientation_error.mean().item()
                    print(f"    Step {step}: Mean orientation error = {mean_orientation:.3f}, Falls = {episode_falls}")
            
            fall_counts.append(episode_falls)
            orientation_errors.extend(episode_orientations)
        
        # 分析结果
        all_orientations = np.concatenate(orientation_errors)
        mean_orientation_error = np.mean(all_orientations)
        total_falls = sum(fall_counts)
        stability_score = max(0, 100 - total_falls * 10 - mean_orientation_error * 100)
        
        self.results['walking_stability'] = {
            'mean_orientation_error': float(mean_orientation_error),
            'total_falls': int(total_falls),
            'falls_per_episode': float(total_falls / num_episodes),
            'score': float(stability_score)
        }
        
        print(f"  📊 平均姿态误差: {mean_orientation_error:.3f}")
        print(f"  🤕 总跌倒次数: {total_falls}")
        print(f"  📊 每episode跌倒: {total_falls/num_episodes:.1f}")
        print(f"  🏆 稳定性得分: {stability_score:.1f}/100")
        
        return env
    
    def evaluate_external_force_resistance(self, num_episodes=5):
        """评估4: 抗外力干扰能力"""
        print("\n💪 评估4: 抗外力干扰能力 (External Force Resistance)")
        print("=" * 60)
        
        # 启用强推力，中等地形
        self.env_cfg.domain_rand.push_robots = True
        self.env_cfg.domain_rand.push_interval_s = 3  # 每3秒推一次
        self.env_cfg.domain_rand.max_push_vel_xy = 2.0  # 强推力
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.terrain.terrain_proportions = [0.5, 0.3, 0.2, 0.0, 0.0]  # 简单地形专注抗推力
        
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        
        recovery_successes = []
        push_counts = []
        recovery_times = []
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            
            obs = env.reset()
            episode_pushes = 0
            episode_recoveries = 0
            push_recovery_start = None
            
            for step in range(int(env.max_episode_length)):
                with torch.no_grad():
                    actions = self.policy(obs)
                
                obs, _, _, dones, _ = env.step(actions)
                
                # 检测推力事件
                if hasattr(env, 'is_pushed') and torch.any(env.is_pushed):
                    if push_recovery_start is None:
                        push_recovery_start = step
                        episode_pushes += 1
                        print(f"    Push detected at step {step}")
                
                # 检测恢复成功 (2秒内保持稳定)
                if push_recovery_start is not None:
                    recovery_time = (step - push_recovery_start) * env.dt
                    
                    # 检查稳定性
                    orientation_stable = torch.all(torch.abs(env.projected_gravity[:, :2]) < 0.3, dim=1)
                    not_fallen = ~env.reset_buf
                    is_stable = orientation_stable & not_fallen
                    
                    if recovery_time >= 2.0:  # 2秒恢复期结束
                        if torch.all(is_stable):
                            episode_recoveries += 1
                            recovery_times.append(recovery_time)
                            print(f"    Recovery successful in {recovery_time:.1f}s")
                        else:
                            print(f"    Recovery failed after {recovery_time:.1f}s")
                        push_recovery_start = None
                
                if step % 100 == 0:
                    stable_count = torch.sum(~env.reset_buf).item()
                    print(f"    Step {step}: {stable_count}/{env.num_envs} robots stable")
            
            push_counts.append(episode_pushes)
            if episode_pushes > 0:
                recovery_successes.append(episode_recoveries / episode_pushes)
            else:
                recovery_successes.append(1.0)  # 没有推力就算完全成功
        
        # 分析结果
        mean_recovery_rate = np.mean(recovery_successes)
        total_pushes = sum(push_counts)
        mean_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        self.results['external_force_resistance'] = {
            'mean_recovery_rate': float(mean_recovery_rate),
            'total_pushes': int(total_pushes),
            'mean_recovery_time': float(mean_recovery_time),
            'score': float(mean_recovery_rate * 100)
        }
        
        print(f"  📊 平均恢复成功率: {mean_recovery_rate:.1%}")
        print(f"  💥 总推力次数: {total_pushes}")
        print(f"  ⏱️ 平均恢复时间: {mean_recovery_time:.1f}秒")
        print(f"  🏆 抗干扰得分: {mean_recovery_rate * 100:.1f}/100")
        
        return env
    
    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        print("\n" + "=" * 80)
        print("🏆 PointFoot 机器人性能评估报告")
        print("=" * 80)
        
        # 计算总分
        total_score = 0
        weights = {
            'tracking_precision': 0.3,
            'terrain_traversal': 0.25,
            'walking_stability': 0.25,
            'external_force_resistance': 0.2
        }
        
        for metric, weight in weights.items():
            if metric in self.results:
                total_score += self.results[metric]['score'] * weight
        
        print(f"\n📊 各项评分详情:")
        print(f"├─ 🎯 追踪指令精度:     {self.results['tracking_precision']['score']:.1f}/100 (权重30%)")
        print(f"├─ 🏔️ 地形跨越性能:     {self.results['terrain_traversal']['score']:.1f}/100 (权重25%)")
        print(f"├─ ⚖️ 行走稳定性:       {self.results['walking_stability']['score']:.1f}/100 (权重25%)")
        print(f"└─ 💪 抗外力干扰能力:   {self.results['external_force_resistance']['score']:.1f}/100 (权重20%)")
        
        print(f"\n🎖️ 综合评分: {total_score:.1f}/100")
        
        # 等级评定
        if total_score >= 90:
            grade = "S级 (优秀)"
        elif total_score >= 80:
            grade = "A级 (良好)"
        elif total_score >= 70:
            grade = "B级 (合格)"
        elif total_score >= 60:
            grade = "C级 (需改进)"
        else:
            grade = "D级 (不合格)"
        
        print(f"🏅 性能等级: {grade}")
        
        # 保存详细结果
        self.results['total_score'] = total_score
        self.results['grade'] = grade
        
        # 生成JSON报告
        report_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细报告已保存至: {report_path}")
        
        return total_score
    
    def run_full_evaluation(self):
        """运行完整评估流程"""
        print("🚀 开始 PointFoot 机器人全面性能评估")
        print("测试模型:", self.train_cfg.runner.experiment_name)
        print("评估时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
        
        try:
            # 1. 追踪指令精度
            self.evaluate_tracking_precision()
            
            # 2. 地形跨越性能  
            self.evaluate_terrain_traversal()
            
            # 3. 行走稳定性
            self.evaluate_walking_stability()
            
            # 4. 抗外力干扰能力
            self.evaluate_external_force_resistance()
            
            # 5. 生成综合报告
            final_score = self.generate_comprehensive_report()
            
            print(f"\n✅ 评估完成！最终得分: {final_score:.1f}/100")
            return final_score
            
        except Exception as e:
            print(f"❌ 评估过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0

def evaluate(args):
    """主评估函数"""
    evaluator = PointFootEvaluator(args)
    return evaluator.run_full_evaluation()

if __name__ == '__main__':
    args = get_args()
    evaluate(args) 