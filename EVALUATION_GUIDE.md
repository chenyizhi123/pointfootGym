# 🏆 PointFoot 机器人性能评估指南

## 📋 **评估脚本功能说明**

`play_evaluate.py` 是专门设计的模型评估脚本，用于全面测试训练好的PointFoot机器人在四个关键评分标准上的表现。

### **🎯 四大评估维度**

| 评估项目 | 测试内容 | 权重 | 评分标准 |
|---------|---------|------|---------|
| **🎯 追踪指令精度** | 在平坦地形测试速度指令跟踪能力 | 30% | 误差越小得分越高 |
| **🏔️ 地形跨越性能** | 在复杂地形测试穿越距离和成功率 | 25% | 行走距离和通过率 |
| **⚖️ 行走稳定性** | 测试姿态控制和跌倒频率 | 25% | 姿态误差和跌倒次数 |
| **💪 抗外力干扰** | 推力干扰下的恢复能力 | 20% | 恢复成功率和时间 |

## 🚀 **使用方法**

### **基本使用**
```bash
# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A

# 运行评估（自动加载最新训练的模型）
python legged_gym/scripts/play_evaluate.py --task=pointfoot_rough
```

### **指定具体模型**
```bash
# 指定特定的训练run
python legged_gym/scripts/play_evaluate.py --task=pointfoot_rough --run_name=Dec05_14-30-25_
```

### **自定义评估参数**
```bash
# 增加评估episodes数量（默认每项5个episodes）
python legged_gym/scripts/play_evaluate.py --task=pointfoot_rough --num_episodes=10
```

## 📊 **评估流程详解**

### **阶段1: 追踪指令精度测试 (30%权重)**

**测试环境设置:**
- ✅ 平坦地形 (terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0])
- ❌ 无推力干扰 (push_robots = False)
- ❌ 无噪声 (add_noise = False)

**评估指标:**
```python
# 计算线性和角速度追踪误差
lin_vel_error = torch.norm(commands[:, :2] - base_lin_vel[:, :2], dim=1)
ang_vel_error = torch.abs(commands[:, 2] - base_ang_vel[:, 2])
total_error = lin_vel_error + ang_vel_error

# 评分公式
score = max(0, 100 * (1 - mean_error))
```

**输出示例:**
```
🎯 评估1: 追踪指令精度 (Tracking Precision)
============================================================
  Episode 1/5
    Step 0: Mean tracking error = 1.234
    Step 100: Mean tracking error = 0.456
  ...
  📊 平均追踪误差: 0.312 ± 0.089
  ✅ 成功率 (误差<0.5): 78.5%
  🏆 追踪精度得分: 68.8/100
```

### **阶段2: 地形跨越性能测试 (25%权重)**

**测试环境设置:**
- 🏔️ 复杂地形 (terrain_proportions = [0.0, 0.2, 0.3, 0.3, 0.2])
- ❌ 无推力干扰
- 📏 30秒episode长度

**评估指标:**
```python
# 计算从起点的直线距离
distance = torch.norm(current_pos - initial_pos, dim=1)

# 评分公式 (8米为满分)
score = min(100, 100 * mean_distance / 8.0)
```

**输出示例:**
```
🏔️ 评估2: 地形跨越性能 (Terrain Traversal)
============================================================
  📊 平均穿越距离: 5.67 ± 1.23 米
  ✅ 成功率 (>4米): 80.0%
  🏆 地形穿越得分: 70.9/100
```

### **阶段3: 行走稳定性测试 (25%权重)**

**测试环境设置:**
- 🏕️ 中等难度地形 (terrain_proportions = [0.3, 0.4, 0.2, 0.1, 0.0])
- ❌ 无推力干扰

**评估指标:**
```python
# 姿态稳定性 (roll, pitch角度误差)
orientation_error = torch.norm(projected_gravity[:, :2], dim=1)

# 跌倒统计
falls = torch.sum(reset_buf).item()

# 评分公式
score = max(0, 100 - total_falls * 10 - mean_orientation * 100)
```

**输出示例:**
```
⚖️ 评估3: 行走稳定性 (Walking Stability)
============================================================
  📊 平均姿态误差: 0.089
  🤕 总跌倒次数: 3
  📊 每episode跌倒: 0.6
  🏆 稳定性得分: 61.1/100
```

### **阶段4: 抗外力干扰测试 (20%权重)**

**测试环境设置:**
- 💥 强推力 (push_interval_s = 3, max_push_vel_xy = 2.0)
- 🏕️ 简单地形专注抗干扰能力

**评估指标:**
```python
# 检测推力恢复 (2秒窗口内保持稳定)
is_stable = (~reset_buf) & \
           (torch.abs(projected_gravity[:, 0]) < 0.3) & \
           (torch.abs(projected_gravity[:, 1]) < 0.3)

# 评分公式
score = mean_recovery_rate * 100
```

**输出示例:**
```
💪 评估4: 抗外力干扰能力 (External Force Resistance)
============================================================
  📊 平均恢复成功率: 75.0%
  💥 总推力次数: 28
  ⏱️ 平均恢复时间: 1.8秒
  🏆 抗干扰得分: 75.0/100
```

## 🏆 **综合评估报告**

评估完成后会生成详细的综合报告:

```
================================================================================
🏆 PointFoot 机器人性能评估报告
================================================================================

📊 各项评分详情:
├─ 🎯 追踪指令精度:     68.8/100 (权重30%)
├─ 🏔️ 地形跨越性能:     70.9/100 (权重25%)
├─ ⚖️ 行走稳定性:       61.1/100 (权重25%)
└─ 💪 抗外力干扰能力:   75.0/100 (权重20%)

🎖️ 综合评分: 68.4/100
🏅 性能等级: C级 (需改进)

💾 详细报告已保存至: /path/to/logs/evaluation_report.json
```

### **等级评定标准**
- **S级 (优秀)**: 90+ 分
- **A级 (良好)**: 80-89 分  
- **B级 (合格)**: 70-79 分
- **C级 (需改进)**: 60-69 分
- **D级 (不合格)**: <60 分

## 📁 **输出文件**

### **JSON详细报告** (`evaluation_report.json`)
```json
{
  "tracking_precision": {
    "mean_error": 0.312,
    "std_error": 0.089,
    "success_rate": 0.785,
    "score": 68.8
  },
  "terrain_traversal": {
    "mean_distance": 5.67,
    "std_distance": 1.23,
    "success_rate": 0.8,
    "score": 70.9
  },
  "walking_stability": {
    "mean_orientation_error": 0.089,
    "total_falls": 3,
    "falls_per_episode": 0.6,
    "score": 61.1
  },
  "external_force_resistance": {
    "mean_recovery_rate": 0.75,
    "total_pushes": 28,
    "mean_recovery_time": 1.8,
    "score": 75.0
  },
  "total_score": 68.4,
  "grade": "C级 (需改进)"
}
```

## 🔧 **高级配置**

### **自定义评估参数**

如需修改评估标准，可以编辑脚本中的相关参数:

```python
# 在 PointFootEvaluator 类中修改
class PointFootEvaluator:
    def evaluate_tracking_precision(self, num_episodes=5):
        # 修改成功率阈值
        success_rate = np.mean(all_errors < 0.3)  # 更严格的标准
        
    def evaluate_terrain_traversal(self, num_episodes=5):
        # 修改距离评分标准
        score = min(100, 100 * mean_distance / 10.0)  # 10米满分
```

### **环境配置调整**

```python
def _setup_evaluation_config(self):
    # 使用更多环境进行统计
    self.env_cfg.env.num_envs = 200
    
    # 更长的episode测试
    self.env_cfg.env.episode_length_s = 60  # 60秒
```

## 🎯 **使用建议**

1. **训练阶段评估**: 每隔几千步进行快速评估，监控训练进度
2. **模型选择**: 用于比较不同训练策略的效果
3. **调参参考**: 根据各项得分调整reward权重和训练参数
4. **最终验证**: 训练完成后的全面性能验证

## ⚠️ **注意事项**

- 确保已有训练好的模型权重文件
- 评估过程中会禁用随机化，确保结果可重复
- 建议在GPU上运行以获得更快的评估速度
- 每次完整评估约需10-15分钟（取决于hardware）

这个评估脚本将帮助你全面了解模型在各个关键性能指标上的表现，为进一步优化提供数据支持！ 