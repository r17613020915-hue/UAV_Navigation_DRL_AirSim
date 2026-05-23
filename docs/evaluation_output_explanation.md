# 评估输出详细说明

## 评估输出的完整结构

评估结束后会打印一个综合报告，包含以下几个部分：

---

## 1. 基础统计信息 (EVALUATION SUMMARY)

### `Total Episodes`
- **含义**: 总共评估的episode数量
- **示例**: `Total Episodes: 10`
- **说明**: 这是你设置的评估轮数

### `Average Episode Reward`
- **含义**: 所有episode的平均奖励
- **示例**: `Average Episode Reward: 15.2345`
- **说明**: 
  - 数值越大越好
  - 如果为负值，说明agent表现很差
  - 这个值综合了所有episode的奖励

### `Success Rate`
- **含义**: 成功率（到达目标的episode比例）
- **示例**: `Success Rate: 0.6000 (60.00%)`
- **说明**: 
  - 范围：0.0 到 1.0（或 0% 到 100%）
  - 越高越好，理想情况下应该接近100%
  - 如果为0，说明agent从未成功到达目标

### `Crash Rate`
- **含义**: 碰撞率（撞到障碍物的episode比例）
- **示例**: `Crash Rate: 0.3000 (30.00%)`
- **说明**: 
  - 范围：0.0 到 1.0（或 0% 到 100%）
  - 越低越好，理想情况下应该接近0%
  - 如果为100%，说明agent总是撞到障碍物

### `Average Episode Steps`
- **含义**: 所有episode的平均步数
- **示例**: `Average Episode Steps: 125.50`
- **说明**: 
  - 表示agent平均存活了多长时间
  - 如果这个值很小（如20-30），说明agent很快就失败了
  - 如果这个值很大，说明agent能够存活较长时间

### `Average Success Episode Steps`
- **含义**: 成功episode的平均步数
- **示例**: `Average Success Episode Steps: 98.33`
- **说明**: 
  - 只统计成功到达目标的episode
  - 表示成功完成任务平均需要多少步
  - 越小越好（说明完成任务更快）

### `Success Episodes`
- **含义**: 成功episode的数量和总数
- **示例**: `Success Episodes: 6/10`
- **说明**: 显示有多少个episode成功完成了任务

### `Average Attack Rate` (如果启用了攻击)
- **含义**: 平均攻击率（在置信度攻击模式下）
- **示例**: `Average Attack Rate: 45.23%`
- **说明**: 
  - 表示在所有步骤中，有多少比例触发了攻击
  - 如果使用置信度攻击，这个值应该在0-100%之间
  - 如果总是攻击，这个值会是100%

---

## 2. 攻击统计信息 (ATTACK STATISTICS)

**注意**: 这部分只在启用了攻击时才会显示

### `Average Attack Rate`
- **含义**: 所有episode的平均攻击率
- **示例**: `Average Attack Rate: 45.23%`
- **说明**: 与上面基础统计中的相同

### `Attack Rate Range`
- **含义**: 攻击率的范围（最小值和最大值）
- **示例**: `Attack Rate Range: [12.50%, 78.90%]`
- **说明**: 
  - 显示不同episode之间攻击率的差异
  - 如果范围很大，说明不同episode的置信度差异很大

### `Average Max Distance (Method 1)`
- **含义**: 方法1的平均最大距离
- **示例**: `Average Max Distance (Method 1): 0.6234`
- **说明**: 
  - **方法1**: 基于动作值的置信度检测
  - 将动作归一化到[0,1]，计算距离0.5的最大距离
  - 值越大，说明动作越极端（越接近边界），置信度越高
  - 范围：0.0 到 1.0

### `Max Distance Range`
- **含义**: 最大距离的范围
- **示例**: `Max Distance Range: [0.1234, 0.9876]`
- **说明**: 显示不同步骤之间最大距离的差异

### `Average Policy Confidence (Method 2)`
- **含义**: 方法2的平均策略置信度
- **示例**: `Average Policy Confidence (Method 2): 0.7823`
- **说明**: 
  - **方法2**: 基于策略确定性的置信度检测
  - 多次采样策略输出，计算方差，方差越小置信度越高
  - 值越大，说明策略越确定，置信度越高
  - 范围：0.0 到 1.0

### `Policy Confidence Range`
- **含义**: 策略置信度的范围
- **示例**: `Policy Confidence Range: [0.1234, 0.9876]`
- **说明**: 显示不同步骤之间策略置信度的差异

### `Method 1 Triggers (Total)`
- **含义**: 方法1触发的总次数
- **示例**: `Method 1 Triggers (Total): 234`
- **说明**: 在所有评估步骤中，方法1（动作值检测）触发了多少次

### `Method 2 Triggers (Total)`
- **含义**: 方法2触发的总次数
- **示例**: `Method 2 Triggers (Total): 189`
- **说明**: 在所有评估步骤中，方法2（策略确定性检测）触发了多少次

### `Method 2 Failures (Total)`
- **含义**: 方法2失败的总次数
- **示例**: `Method 2 Failures (Total): 12`
- **说明**: 
  - 方法2需要多次采样策略，如果采样失败（如内存不足），会记录失败次数
  - 这个值应该很小，如果很大说明有问题

### `Attack Type`
- **含义**: 使用的攻击类型
- **示例**: `Attack Type: fgsm` 或 `pgd`, `mim`, `bim`, `deepfool`, `cw`, `random`
- **说明**: 显示使用了哪种对抗攻击方法

### `Attack Epsilon`
- **含义**: 攻击的扰动强度
- **示例**: `Attack Epsilon: 0.01`
- **说明**: 
  - 值越大，攻击越强，但可能更容易被检测到
  - 通常范围：0.001 到 0.1

### `Confidence Threshold`
- **含义**: 置信度阈值（仅在置信度攻击模式下显示）
- **示例**: `Confidence Threshold: 0.5`
- **说明**: 
  - 当置信度超过这个阈值时，才会触发攻击
  - 范围：0.0 到 1.0
  - 值越小，越容易触发攻击

### `Attack Type: Always Attack`
- **含义**: 如果显示这个，说明没有使用置信度攻击
- **说明**: 每步都会攻击，攻击率为100%

---

## 3. 奖励分布统计 (REWARD DISTRIBUTION)

### `Min Reward`
- **含义**: 所有episode中的最小奖励
- **示例**: `Min Reward: -45.2345`
- **说明**: 显示最差的episode表现

### `Max Reward`
- **含义**: 所有episode中的最大奖励
- **示例**: `Max Reward: 125.6789`
- **说明**: 显示最好的episode表现

### `Median Reward`
- **含义**: 所有episode的中位数奖励
- **示例**: `Median Reward: 15.1234`
- **说明**: 
  - 中位数比平均值更能反映典型表现
  - 如果中位数和平均值差异很大，说明奖励分布不均匀

### `Std Reward`
- **含义**: 奖励的标准差
- **示例**: `Std Reward: 35.6789`
- **说明**: 
  - 标准差越大，说明episode之间的表现差异越大
  - 如果标准差很大，说明训练不稳定

### `Episodes with Positive Reward`
- **含义**: 获得正奖励的episode数量和比例
- **示例**: `Episodes with Positive Reward: 7/10 (70.00%)`
- **说明**: 
  - 显示有多少episode获得了正奖励
  - 这个比例应该尽可能高

### `Episodes with Negative Reward`
- **含义**: 获得负奖励的episode数量和比例
- **示例**: `Episodes with Negative Reward: 3/10 (30.00%)`
- **说明**: 
  - 显示有多少episode获得了负奖励
  - 这个比例应该尽可能低

---

## 4. 结果数组 (Results Array)

### 格式
```python
Results Array: [avg_reward, success_rate, crash_rate, avg_success_steps, avg_attack_rate]
```

### 各项含义
1. **avg_reward**: 平均奖励
2. **success_rate**: 成功率
3. **crash_rate**: 碰撞率
4. **avg_success_steps**: 成功episode的平均步数
5. **avg_attack_rate**: 平均攻击率（如果启用了攻击）

### 保存位置
- 结果会保存到评估日志文件夹的 `results.npy` 文件中
- 可以用 `np.load()` 加载查看

---

## 5. 每个Episode结束时的输出

在每个episode结束时，会打印一行信息：

```
episode: 1 reward: 15.2345 success: True attack_rate: 45.2% | max_dist_avg: 0.623, policy_conf_avg: 0.782 | M1:23 M2:18 M2_fail:0
```

### 各项含义
- **episode**: episode编号
- **reward**: 这个episode的总奖励
- **success**: 是否成功到达目标（True/False）
- **attack_rate**: 这个episode的攻击率（如果启用了置信度攻击）
- **max_dist_avg**: 这个episode的平均最大距离（方法1）
- **policy_conf_avg**: 这个episode的平均策略置信度（方法2）
- **M1**: 方法1触发的次数
- **M2**: 方法2触发的次数
- **M2_fail**: 方法2失败的次数

---

## 如何解读评估结果

### 好的评估结果应该：
1. ✅ **Success Rate > 80%**: 大部分episode都能成功
2. ✅ **Crash Rate < 20%**: 很少碰撞
3. ✅ **Average Episode Reward > 0**: 平均奖励为正
4. ✅ **Average Episode Steps > 100**: 能够存活较长时间
5. ✅ **Episodes with Positive Reward > 70%**: 大部分episode获得正奖励

### 不好的评估结果可能：
1. ❌ **Success Rate = 0%**: agent从未成功
2. ❌ **Crash Rate = 100%**: agent总是碰撞
3. ❌ **Average Episode Reward < 0**: 平均奖励为负
4. ❌ **Average Episode Steps < 30**: 很快就失败
5. ❌ **Std Reward 很大**: 训练不稳定

### 攻击相关的解读：
- **Attack Rate 接近 100%**: 说明agent总是很"自信"，可能训练过度
- **Attack Rate 接近 0%**: 说明agent总是很"不确定"，可能训练不足
- **Attack Rate 在 30-70%**: 比较正常，说明agent在关键时刻更自信
- **Method 1 和 Method 2 触发次数差异大**: 说明两种检测方法的结果不一致

---

## 示例输出

```
================================================================================
EVALUATION SUMMARY - All Episodes Statistics
================================================================================
Total Episodes: 10
Average Episode Reward: 15.2345
Success Rate: 0.6000 (60.00%)
Crash Rate: 0.3000 (30.00%)
Average Episode Steps: 125.50
Average Success Episode Steps: 98.33
Success Episodes: 6/10
Average Attack Rate: 45.23%

--------------------------------------------------------------------------------
ATTACK STATISTICS
--------------------------------------------------------------------------------
Average Attack Rate: 45.23%
Attack Rate Range: [12.50%, 78.90%]
Average Max Distance (Method 1): 0.6234
Max Distance Range: [0.1234, 0.9876]
Average Policy Confidence (Method 2): 0.7823
Policy Confidence Range: [0.1234, 0.9876]
Method 1 Triggers (Total): 234
Method 2 Triggers (Total): 189
Method 2 Failures (Total): 0
Attack Type: fgsm
Attack Epsilon: 0.01
Confidence Threshold: 0.5

--------------------------------------------------------------------------------
REWARD DISTRIBUTION
--------------------------------------------------------------------------------
Min Reward: -45.2345
Max Reward: 125.6789
Median Reward: 15.1234
Std Reward: 35.6789
Episodes with Positive Reward: 7/10 (70.00%)
Episodes with Negative Reward: 3/10 (30.00%)
================================================================================

Results Array: [15.2345, 0.6, 0.3, 98.33, 45.23]
```

