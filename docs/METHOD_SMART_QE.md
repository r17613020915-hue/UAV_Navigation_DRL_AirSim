# Smart QE (Q-value + Entropy) 攻击方法说明

## 概述

Smart QE是一种在Smart攻击基础上融合Q值差异和动作熵的新型对抗攻击方法，专门针对使用SAC/TD3等Actor-Critic架构的深度强化学习策略。

## 核心思想

### Smart攻击基础
Smart攻击的核心逻辑：
1. **风险感知**：只在UAV接近障碍物时触发攻击
2. **动作置信度**：检测策略是否强烈偏向某一动作

### Smart QE创新点
在Smart基础上增加两个关键指标：
1. **Q值差异**：检测双Q网络评估的不一致性
2. **动作熵**：检测动作分布的不确定性

## 触发条件

Smart QE的触发条件为：
```
高风险 AND (高置信度 OR (高Q差异 AND 低熵))
```

### 各条件详解

#### 1. 高风险 (High Risk)
- 基于与障碍物的距离计算
- 公式：`risk = (crash_distance + risk_margin - min_distance) / (crash_distance + risk_margin)`
- 阈值：`strict_risk_threshold = min(0.95, risk_threshold * 1.5)`

#### 2. 高置信度 (High Confidence)
- 检查策略输出的动作是否接近动作空间边界
- 归一化动作到[0,1]，计算距离中心的最大距离
- 阈值：`strict_conf_threshold = min(0.95, confidence_threshold * 1.3)`

#### 3. 高Q值差异 (High Q-Diff)
- 获取SAC/TD3的双Q网络评估值
- 计算：`q_diff = |Q1 - Q2|`
- 阈值：`q_diff >= q_diff_threshold`

#### 4. 低动作熵 (Low Entropy)
- 多次采样策略动作，使用核密度估计计算熵
- 归一化熵值（相对于均匀分布）
- 阈值：`entropy <= entropy_threshold`

## 攻击强度动态调整

Smart QE攻击会根据各指标动态调整攻击强度ε：

```
intensity_factor = 1.0
+ risk * 0.3                    # 风险越高，攻击越强
+ min(q_diff, 1.0) * 0.4       # Q差异越大，攻击越强
+ (1 - entropy) * 0.3          # 熵越低（动作越集中），攻击越强
+ confidence * 0.2             # 置信度越高，攻击越强

attack_epsilon = base_epsilon * intensity_factor
```

## 损失函数

```
loss = -MSE(action_adv, action_clean)  # 最大化动作差异
      - 0.1 * (Q1 + Q2) / 2             # 鼓励高Q值动作（攻击者视角）
```

## 与其他方法的对比

| 方法 | 触发条件 | 攻击强度 |
|------|---------|---------|
| **Smart** | 高风险 + 高置信度 | 固定/动态 |
| **Smart_Q** | 高风险 + 高Q差异 | 固定/动态 |
| **Smart_Q_Entropy** | 融合分数 >= 阈值 | 动态（基于融合分数） |
| **Smart_QE** | 高风险 × (高置信度 ∨ (高Q差异 ∧ 低熵)) | 动态（基于各指标） |
| **SmartC** | 高风险 + (高熵 ∨ 高Q差异) | 固定/动态 |

## 配置参数

### 必需参数

```ini
# 攻击类型和触发模式
attack_type = smart_qe
attack_trigger_mode = smart_qe

# 基础参数
attack_epsilon = 0.08
attack_confidence_threshold = 0.5
risk_threshold = 0.6
risk_margin = 3.0
```

### Smart QE特定参数

```ini
# Smart QE触发参数
smart_qe_entropy_threshold = 0.4      # 熵阈值
smart_qe_q_diff_threshold = 0.3       # Q值差异阈值
smart_qe_strict_risk_multiplier = 1.5 # 严格风险阈值倍数
smart_qe_strict_conf_multiplier = 1.3 # 严格置信度阈值倍数
```

### 参数调优建议

#### 保守设置（减少误触）
```ini
smart_qe_entropy_threshold = 0.3      # 更严格的熵阈值
smart_qe_q_diff_threshold = 0.5       # 更高的Q差异阈值
smart_qe_strict_risk_multiplier = 2.0 # 更高的风险倍数
smart_qe_strict_conf_multiplier = 1.5 # 更高的置信度倍数
```

#### 激进设置（增加攻击频率）
```ini
smart_qe_entropy_threshold = 0.5      # 更宽松的熵阈值
smart_qe_q_diff_threshold = 0.2       # 更低的Q差异阈值
smart_qe_strict_risk_multiplier = 1.2 # 更低的风险倍数
smart_qe_strict_conf_multiplier = 1.1 # 更低的置信度倍数
```

## 使用示例

### 基本使用

1. **配置文件**：使用提供的`config_smart_qe_attack.ini`

2. **运行评估**：
```bash
python scripts/start_evaluate_with_plot.py --config configs/config_smart_qe_attack.ini
```

### 高级调优

```python
# 在代码中自定义参数
from scripts.utils.thread_evaluation import check_smart_qe_trigger

# 调用触发检查
should_attack, risk, entropy, q_diff, confidence = check_smart_qe_trigger(
    env=env,
    model=model,
    obs=obs,
    device=device,
    action_space=env.action_space,
    perception_type='depth',
    risk_threshold=0.6,
    risk_margin=3.0,
    confidence_threshold=0.5,
    entropy_threshold=0.4,           # 自定义熵阈值
    q_diff_threshold=0.3,             # 自定义Q差异阈值
    strict_risk_multiplier=1.5,      # 自定义严格风险倍数
    strict_conf_multiplier=1.3,      # 自定义严格置信度倍数
    debug=True                        # 开启调试输出
)
```

## 统计信息

Smart QE会记录以下统计信息用于分析：

```python
smart_qe_stats = {
    'risks': [],           # 所有步骤的风险值
    'confidences': [],     # 所有步骤的置信度
    'entropies': [],       # 所有步骤的熵值
    'q_diffs': [],         # 所有步骤的Q值差异
    'triggers': 0          # 触发攻击的总次数
}
```

### 分析示例

```python
import numpy as np

# 计算各指标的统计信息
avg_risk = np.mean(smart_qe_stats['risks'])
avg_entropy = np.mean(smart_qe_stats['entropies'])
avg_q_diff = np.mean(smart_qe_stats['q_diffs'])
trigger_rate = smart_qe_stats['triggers'] / len(smart_qe_stats['risks'])

print(f"平均风险: {avg_risk:.3f}")
print(f"平均熵: {avg_entropy:.3f}")
print(f"平均Q差异: {avg_q_diff:.3f}")
print(f"触发率: {trigger_rate:.2%}")
```

## 与Smart_Q_Entropy的区别

虽然Smart_QE和Smart_Q_Entropy都使用了风险、熵和Q值，但它们的融合方式不同：

### Smart_Q_Entropy（融合方式）
```
fusion_score = risk_weight * risk
             + entropy_weight * entropy
             + q_weight * q_combined

触发条件: fusion_score >= threshold
```

### Smart_QE（逻辑组合方式）
```
触发条件: risk >= strict_risk_threshold
       AND (confidence >= strict_conf_threshold
         OR (q_diff >= q_diff_threshold AND entropy <= entropy_threshold))
```

**区别**：
- Smart_Q_Entropy：加权融合，各因素互补
- Smart_QE：逻辑与/或组合，要求更严格的条件组合

## 注意事项

1. **模型兼容性**：Smart QE依赖SAC/TD3的critic网络，需要使用支持双Q网络的算法
2. **计算开销**：每次触发需要多次采样动作和计算Q值，会有一定计算开销
3. **阈值调优**：不同环境和策略可能需要不同的阈值参数
4. **调试建议**：首次使用时建议开启debug模式，观察触发条件

## 未来改进方向

1. **自适应阈值**：根据运行时的统计数据自动调整阈值
2. **多步攻击**：在触发后连续攻击多步
3. **目标导向**：结合目标状态信息引导攻击方向
4. **多模态融合**：结合图像和向量观测的熵/Q值分析

