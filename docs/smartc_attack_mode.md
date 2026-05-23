# SmartC攻击模式文档

## 概述

SmartC（Smart + Statistical Critical State）是一种新的攻击触发模式，它结合了Smart模式的保守风险感知和Statistical模式的精确统计特征评估。

## 设计理念

SmartC模式旨在提供一种**既保守又精确**的攻击策略：
- **保守性**：通过Smart模式的三重风险过滤，避免在低风险区域无谓攻击
- **精确性**：通过Statistical模式的统计特征评估，确保只在模型真正不确定时攻击

## 触发逻辑详解

### 三重过滤机制

#### 第一重：风险预筛选 (Risk Pre-filtering)
```python
should_attack_risk, risk, min_dist, risk_dist = check_risk_trigger(
    env, risk_threshold, risk_margin
)
```
- 使用基础风险阈值进行初步筛选
- 只有在环境风险足够高时才继续后续评估

#### 第二重：严格风险评估 (Strict Risk Assessment)
```python
strict_risk_threshold = min(0.9, risk_threshold)
is_high_risk = risk >= strict_risk_threshold
```
- 应用更严格的风险阈值（最高0.9）
- 确保只在真正危险的环境中考虑攻击

#### 第三重：统计特征评估 (Statistical Feature Assessment)
```python
# 使用更严格的统计阈值
strict_entropy_scale = max(statistical_entropy_scale, 10.0)
strict_q_diff_scale = max(statistical_q_diff_scale, 3.0)
strict_variance_scale = max(statistical_variance_scale, 3.0)

should_attack_statistical = check_statistical_critical_state(...)
```
- **Q值差异检测**：检查双Q网络输出的一致性
- **动作熵评估**：分析动作分布的确定性
- **动作方差分析**：评估策略的稳定性

### 统计特征计算

#### Q值差异 (Q-Value Difference)
```python
# 计算双Q网络输出差异
q_outputs = critic(obs_t, action_t)
if len(q_outputs) >= 2:
    q_value_diff = abs(q_outputs[0] - q_outputs[1]).mean()
```

#### 动作熵 (Action Entropy)
```python
# 基于动作方差计算熵值
action_variances = var(actions_samples, axis=0)
normalized_variance = mean(variances / action_ranges^2)
action_entropy = -log(clip(normalized_variance, 1e-6, 1.0))
```

### 动态攻击强度调整

与Smart模式相同，根据风险等级动态调整epsilon：
```python
scale = 1.0 + 1.5 * risk  # risk从0→1时，scale从1.0→2.5
eps = attack_epsilon * scale
```

## 配置参数

```ini
[options]
# 启用SmartC模式
attack_trigger_mode = smartc

# 风险感知参数
risk_margin = 8.5      # 风险边际距离
risk_threshold = 0.1   # 基础风险阈值

# 统计特征评估参数
statistical_entropy_scale = 7.0   # 熵评估阈值放大倍数
statistical_q_diff_scale = 2.5    # Q值差异阈值放大倍数
statistical_variance_scale = 2.0  # 方差阈值放大倍数

# 攻击参数
attack_epsilon = 0.07  # 基础攻击强度
attack_type = fgsm     # 攻击类型
```

## 决策流程

```
开始攻击判断
    │
    ├── 1. 风险预筛选
    │   ├── ✅ 环境风险 ≥ risk_threshold → 继续
    │   └── ❌ 环境风险 < risk_threshold → 不攻击
    │
    ├── 2. 严格风险评估
    │   ├── ✅ 环境风险 ≥ min(0.9, risk_threshold) → 继续
    │   └── ❌ 环境风险 < 严格阈值 → 不攻击
    │
    └── 3. 统计特征评估
        ├── ✅ Q值差异大 OR 动作熵高 OR 方差大 → 执行攻击
        └── ❌ 所有特征都正常 → 不攻击
```

## 与其他模式的对比

| 特性 | Smart | SmartC | Statistical | Risk |
|------|-------|--------|------------|------|
| **风险感知** | ✅ 三重过滤 | ✅ 三重过滤 | ❌ 无 | ✅ 风险权重 |
| **统计精确性** | ❌ 置信度评估 | ✅ 统计特征 | ✅ 统计特征 | ❌ 无 |
| **攻击频率** | 最低 | 中等 | 高 | 高 |
| **计算复杂度** | 中等 | 高 | 高 | 低 |
| **保守程度** | 最高 | 高 | 中等 | 低 |

## 适用场景

- **生产环境**：需要平衡安全性和攻击效果
- **关键系统**：要求攻击决策高度可靠
- **资源受限**：希望减少不必要的攻击计算
- **精确控制**：需要基于模型状态的智能决策

## 调试信息

启用debug模式时，SmartC会输出详细的触发信息：
```
[Attack Trigger] SmartC mode: attack triggered at step 123
(risk: 0.85, entropy: 6.2, q_diff: 3.1, variance: 0.15)
```

这有助于理解触发决策的具体依据。
