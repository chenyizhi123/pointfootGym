# Isaac Gym到MuJoCo模型部署方案

## 问题分析

### 观测空间对比
| 组件 | Isaac Gym | MuJoCo | 状态 |
|------|-----------|--------|------|
| base_ang_vel | 3维 | 3维 | ✅ 匹配 |
| projected_gravity | 3维 | 3维 | ✅ 匹配 |
| joint_pos (相对) | 6维 | 6维 | ✅ 匹配 |
| joint_vel | 6维 | 6维 | ✅ 匹配 |
| actions | 6维 | 6维 | ✅ 匹配 |
| commands | 3维 | 3维 | ✅ 匹配 |
| **clock_inputs_sin** | **1维** | **❌ 缺失** | **需添加** |
| **clock_inputs_cos** | **1维** | **❌ 缺失** | **需添加** |
| **gaits参数** | **4维** | **❌ 缺失** | **需添加** |

**总计**：Isaac Gym (33维) vs MuJoCo (27维) - **缺少6维步态控制观测**

## 解决方案

### 方案A：扩展MuJoCo观测空间 (推荐)

#### 1. 修改observation计算
```python
# 在rl_controller.py的compute_observation()方法中添加：

def compute_observation(self):
    # ... 现有代码 ...
    
    # 新增：步态控制参数
    self._update_gait_params()
    
    # 扩展观测空间
    obs = np.concatenate([
        base_ang_vel * self.obs_scales['ang_vel'],
        projected_gravity,
        joint_pos_input,
        joint_velocities * self.obs_scales['dof_vel'],
        actions,
        scaled_commands,
        # 新增步态观测
        np.array([self.clock_inputs_sin]),  # 1维
        np.array([self.clock_inputs_cos]),  # 1维
        self.gaits                          # 4维 [freq, offset, duration, swing_height]
    ])
```

#### 2. 添加步态控制模块
```python
# 在PointfootController类中添加方法：

def __init__(self, ...):
    # ... 现有代码 ...
    
    # 步态控制相关初始化
    self.gait_indices = 0.0
    self.gaits = np.array([2.0, 0.5, 0.5, 0.05])  # [frequency, offset, duration, swing_height]
    self.clock_inputs_sin = 0.0
    self.clock_inputs_cos = 1.0
    self.desired_contact_states = np.zeros(2)  # 双足
    
def _update_gait_params(self):
    """更新步态参数，与Isaac Gym对齐"""
    frequencies = self.gaits[0]
    self.gait_indices = (self.gait_indices + self.dt * frequencies) % 1.0
    
    self.clock_inputs_sin = np.sin(2 * np.pi * self.gait_indices)
    self.clock_inputs_cos = np.cos(2 * np.pi * self.gait_indices)
    
    # 计算期望接触状态（简化版）
    offset = self.gaits[1]
    duration = self.gaits[2]
    
    foot_indices = np.array([
        self.gait_indices,
        (self.gait_indices + offset) % 1.0
    ])
    
    self.desired_contact_states = (foot_indices < duration).astype(float)

def _resample_gaits(self):
    """定期重新采样步态参数"""
    if self.loop_count % (5 * self.loop_frequency) == 0:  # 每5秒
        self.gaits[0] = np.random.uniform(1.5, 2.5)  # frequency
        self.gaits[1] = 0.5  # offset (双足固定0.5)
        self.gaits[2] = np.random.uniform(0.4, 0.6)  # duration
        self.gaits[3] = np.random.uniform(0.0, 0.1)  # swing_height
```

#### 3. 修改配置文件
```yaml
# params.yaml中更新
PointfootCfg:
  size:
    observations_size: 33  # 从27改为33
    
  gait_params:
    frequency_range: [1.5, 2.5]
    offset: 0.5  # 双足固定
    duration_range: [0.4, 0.6]
    swing_height_range: [0.0, 0.1]
```

### 方案B：修改模型适配MuJoCo (备选)

如果无法修改MuJoCo代码，可以：

1. **重新训练模型**：用27维观测空间训练新模型
2. **模型适配层**：在ONNX模型前添加预处理层，填充缺失的6维

### 方案C：混合方案 (逐步迁移)

1. **第一阶段**：使用方案B快速验证基本功能
2. **第二阶段**：实施方案A，完整移植步态控制
3. **第三阶段**：微调参数，优化性能

## 实施步骤

### Step 1: 参数对齐
```python
# 修改参数匹配Isaac Gym
self.control_cfg['decimation'] = 4  # 从10改为4
self.control_cfg['damping'] = 1.5   # 从2.0改为1.5
self.loop_frequency = 200           # 从500改为200
```

### Step 2: 添加步态控制
- 实施上述_update_gait_params()方法
- 添加步态参数重采样
- 扩展观测空间到33维

### Step 3: 验证和调试
- 对比Isaac Gym和MuJoCo的观测值
- 确保数值范围和scaling一致
- 测试模型推理结果

### Step 4: 性能优化
- 微调控制参数
- 调整步态参数范围
- 优化实时性能

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 观测不匹配导致性能下降 | 中 | 高 | 仔细对比数值，逐步验证 |
| 步态控制实现复杂 | 低 | 中 | 从简单版本开始，逐步完善 |
| 实时性能问题 | 低 | 低 | 优化计算，减少不必要操作 |

## 预期效果

- ✅ 完全兼容Isaac Gym训练的模型
- ✅ 保持原有的步态控制能力  
- ✅ 实现平滑的sim-to-sim迁移
- ✅ 为后续real-to-sim奠定基础