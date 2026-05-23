# 置信度攻击（Confidence-based Attack）逻辑详解

## 1. 整体思路

置信度攻击的核心思想是：**只在智能体对动作非常确定（高置信度）的时候才进行攻击**。这样做的原因是：
- 当智能体不确定时，攻击效果不明显
- 当智能体非常确定时，小的扰动可能导致大的动作变化
- 节省计算资源，只在关键时刻攻击

## 2. 攻击流程

```
┌─────────────────────────────────────────────────────────┐
│  每个时间步（Step）的评估流程                              │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  获取当前观测 obs     │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  模型预测动作         │
        │  action = model.predict(obs) │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  检查置信度           │
        │  check_action_confidence() │
        └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   置信度高？              置信度低？
        │                       │
        ▼                       ▼
   生成对抗样本              使用原始观测
    obs_adv = attack()      obs_adv = obs
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  使用 obs_adv 预测    │
        │  action = model.predict(obs_adv) │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  执行动作，环境更新   │
        └───────────────────────┘
```

## 3. 置信度检测逻辑（check_action_confidence）

### 方法1：动作值边界检测

**原理**：如果动作值接近动作空间的边界，说明智能体对动作很确定。

**步骤**：
1. **归一化动作**：将动作从原始范围映射到 [0, 1]
   ```python
   action_normalized = (action - action_low) / (action_high - action_low)
   ```
   
   例如：
   - 动作空间：v_xy ∈ [0.5, 5.0], yaw_rate ∈ [-0.52, 0.52]
   - 如果 action = [4.5, 0.4]
   - 归一化后：action_normalized = [(4.5-0.5)/(5.0-0.5), (0.4-(-0.52))/(0.52-(-0.52))]
   - 结果：≈ [0.89, 0.88]

2. **计算距离中心的距离**
   ```python
   distances_from_center = |action_normalized - 0.5| * 2
   ```
   - 中心是 0.5（归一化后的中间值）
   - 乘以2是为了将范围从 [0, 0.5] 扩展到 [0, 1]
   
   例如：
   - action_normalized = [0.89, 0.88]
   - distances = [|0.89-0.5|*2, |0.88-0.5|*2] = [0.78, 0.76]
   - max_distance = 0.78

3. **判断是否超过阈值**
   ```python
   should_attack = max_distance >= threshold  # 默认 threshold=0.5
   ```
   - 如果 max_distance = 0.78 >= 0.5，则触发攻击 ✓

### 方法2：Policy确定性检测（新增）

**原理**：通过多次采样计算动作的方差，方差小说明policy很确定。

**步骤**：
1. **多次采样动作**
   ```python
   for i in range(10):  # 采样10次
       action_i = model.predict(obs, deterministic=False)  # 非确定性采样
       actions_samples.append(action_i)
   ```

2. **计算方差**
   ```python
   actions_array = np.array(actions_samples)  # shape: (10, action_dim)
   action_variance = np.mean(np.var(actions_array, axis=0))  # 每个维度的方差，取平均
   ```

3. **归一化方差**
   ```python
   action_range = mean(action_high - action_low)
   normalized_variance = action_variance / (action_range ** 2)
   ```

4. **计算置信度**
   ```python
   policy_confidence = 1.0 - clip(normalized_variance, 0, 1)
   ```
   - 方差越小 → 置信度越高
   - 例如：方差=0.1 → 置信度=0.9

5. **判断是否超过阈值**
   ```python
   should_attack = (max_distance >= threshold) OR (policy_confidence >= threshold)
   ```

## 4. 攻击生成（以FGSM为例）

当置信度高时，生成对抗样本：

### FGSM攻击步骤：

1. **获取干净动作**
   ```python
   action_clean = model.predict(obs, deterministic=True)
   ```

2. **前向传播获取对抗动作**
   ```python
   obs_t.requires_grad_(True)  # 需要梯度
   action_adv = model.actor(obs_t)  # 通过actor网络
   ```

3. **计算损失**
   ```python
   loss = MSE(action_adv, action_clean)
   ```
   - 目标是让对抗动作与干净动作差异最大

4. **反向传播**
   ```python
   loss.backward()  # 计算 obs_t.grad
   ```

5. **生成对抗样本**
   ```python
   grad_sign = sign(obs_t.grad)  # 梯度的符号
   obs_adv = obs + epsilon * grad_sign  # FGSM公式
   ```

6. **裁剪到有效范围**
   ```python
   obs_adv = clip(obs_adv, obs - epsilon, obs + epsilon)  # L∞约束
   obs_adv = clip(obs_adv, 0, 1)  # 观测空间约束
   ```

## 5. 具体例子

### 例子1：动作接近边界

假设：
- 动作空间：v_xy ∈ [0.5, 5.0], yaw_rate ∈ [-0.52, 0.52]
- 当前动作：action = [4.8, 0.45]  # 接近最大值
- 阈值：threshold = 0.5

计算：
1. 归一化：action_normalized = [(4.8-0.5)/(5.0-0.5), (0.45-(-0.52))/(0.52-(-0.52))]
   = [0.96, 0.93]
2. 距离中心：distances = [|0.96-0.5|*2, |0.93-0.5|*2] = [0.92, 0.86]
3. max_distance = 0.92 >= 0.5 ✓ **触发攻击**

### 例子2：动作在中间

假设：
- 当前动作：action = [2.5, 0.0]  # 在中间
- 阈值：threshold = 0.5

计算：
1. 归一化：action_normalized = [(2.5-0.5)/(5.0-0.5), (0.0-(-0.52))/(0.52-(-0.52))]
   = [0.44, 0.50]
2. 距离中心：distances = [|0.44-0.5|*2, |0.50-0.5|*2] = [0.12, 0.0]
3. max_distance = 0.12 < 0.5 ✗ **不触发攻击**

### 例子3：Policy确定性高

假设：
- 多次采样动作：10次采样结果都很接近
- 方差：action_variance = 0.05（很小）
- 归一化方差：normalized_variance = 0.05 / (4.5²) ≈ 0.002
- 置信度：policy_confidence = 1.0 - 0.002 = 0.998
- 阈值：threshold = 0.5

判断：0.998 >= 0.5 ✓ **触发攻击**

## 6. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `attack_confidence_threshold` | 0.5 | 置信度阈值，越低越容易触发攻击 |
| `attack_epsilon` | 0.1 | 攻击强度，越大扰动越大 |
| `attack_on_confidence` | False | 是否启用置信度攻击 |
| `debug_attack` | False | 是否打印调试信息 |

## 7. 为什么攻击效果不明显？

可能的原因：

1. **阈值太高**：threshold=0.7 太高，攻击很少触发
   - 解决：降低到 0.3-0.5

2. **epsilon太小**：攻击强度不够
   - 解决：增加到 0.2-0.3

3. **置信度检测不准确**：只检查动作值，没检查policy确定性
   - 解决：已修复，现在会检查policy方差

4. **攻击方法不够强**：FGSM是单步攻击
   - 解决：使用PGD或MIM（多步迭代攻击）

5. **模型鲁棒性强**：模型对扰动不敏感
   - 解决：增加epsilon或使用更强的攻击方法

## 8. 调试建议

启用调试模式：
```ini
[options]
debug_attack = True
attack_on_confidence = True
attack_confidence_threshold = 0.3  # 降低阈值，更容易触发
```

查看输出：
```
[Attack Debug] Step 123: Attack triggered! Max action distance: 0.85, Threshold: 0.3
episode: 1 reward: -10.5 success: False attack_rate: 45.2%
```

- `attack_rate`：攻击触发频率，应该在20-60%之间比较合理
- 如果attack_rate太低（<10%），说明阈值太高
- 如果attack_rate太高（>80%），说明阈值太低，攻击太频繁


