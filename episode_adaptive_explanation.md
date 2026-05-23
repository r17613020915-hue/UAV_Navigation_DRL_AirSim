# Episode内自适应阈值调节机制

## 🎯 核心理念

**每个Episode都是独立的**，不应依赖历史Episode的表现进行调节。相反，应该在每个Episode内部，根据当前Episode的实时表现进行动态阈值调整。

## 🏗️ 系统架构

### EpisodeAdaptiveTuner 类

```python
class EpisodeAdaptiveTuner:
    def __init__(self, initial_entropy_scale=7.0, ...):
        # 基准阈值 (Episode间保持)
        self.base_entropy_scale = initial_entropy_scale
        self.base_variance_scale = initial_variance_scale

        # Episode内动态阈值 (实时调节)
        self.current_entropy_scale = initial_entropy_scale
        self.current_variance_scale = initial_variance_scale

        # Episode内统计
        self.episode_attack_count = 0
        self.episode_step_count = 0
        self.episode_reward_sum = 0.0

        # 调节参数
        self.learning_rate = 0.3          # Episode内学习率
        self.target_attack_rate = 0.20    # Episode内目标攻击率
        self.adaptation_interval = 50     # 每50步调节一次
```

## 🔄 工作流程

### 1. Episode开始时 (静默)
```python
def start_episode(self):
    # 重置Episode统计
    self.episode_attack_count = 0
    self.episode_step_count = 0
    self.episode_reward_sum = 0.0

    # 重置为基准阈值 (不打印)
    self.current_entropy_scale = self.base_entropy_scale
    self.current_variance_scale = self.base_variance_scale
```

### 2. 每一步实时更新
```python
def update_step(self, did_attack, reward):
    self.episode_step_count += 1
    if did_attack:
        self.episode_attack_count += 1
    self.episode_reward_sum += reward

    # 每50步检查并调节一次
    if self.episode_step_count % 50 == 0:
        self._adapt_during_episode()
```

### 3. Episode内动态调节 (静默执行)
```python
def _adapt_during_episode(self):
    current_attack_rate = self.episode_attack_count / self.episode_step_count
    attack_rate_error = current_attack_rate - self.target_attack_rate

    if attack_rate_error > 0.05:  # 攻击过多
        self.current_entropy_scale *= (1 + self.learning_rate * attack_rate_error)
        self.current_variance_scale *= (1 + self.learning_rate * attack_rate_error * 0.7)
    elif attack_rate_error < -0.05:  # 攻击过少
        self.current_entropy_scale *= (1 + self.learning_rate * attack_rate_error)
        self.current_variance_scale *= (1 + self.learning_rate * attack_rate_error * 0.7)

    # 不打印每步调节信息，保持输出简洁
```

### 4. Episode结束时 (打印总结)
```python
def end_episode(self, final_reward, success):
    final_attack_rate = self.episode_attack_count / self.episode_step_count

    # 根据Episode整体表现微调基准阈值
    if final_attack_rate > 0.3:
        self.base_entropy_scale *= 1.02  # 轻微提高基准
    elif final_attack_rate < 0.1:
        self.base_entropy_scale *= 0.98  # 轻微降低基准

    # 打印Episode调节总结
    threshold_change_e = self.base_entropy_scale - self.episode_start_thresholds['entropy_scale']
    threshold_change_v = self.base_variance_scale - self.episode_start_thresholds['variance_scale']

    print(f"[Episode Adaptive] Final: Attack rate: {final_attack_rate:.3f}, "
          f"Reward: {final_reward:.1f}, Success: {success}, "
          f"Threshold changes: E:{threshold_change_e:+.2f}, V:{threshold_change_v:+.2f}")
```

## 📊 调节逻辑详解

### 攻击率调节
- **目标**: 维持20%的Episode内攻击率
- **过高时** (>25%): 提高阈值 → 更难触发 → 攻击率降低
- **过低时** (<15%): 降低阈值 → 更容易触发 → 攻击率升高

### 奖励表现调节
- **奖励太差** (< -2.0/步): 提高阈值减少攻击
- **奖励不错** (>1.0/步): 轻微降低阈值增加攻击

### 调节频率
- **检查间隔**: 每50步检查一次
- **调节时机**: Episode内实时调节
- **基准更新**: Episode结束时微调基准值

## 🎪 与原系统的区别

| 方面 | 原AdaptiveThresholdTuner | 新EpisodeAdaptiveTuner |
|------|-------------------------|----------------------|
| **调节范围** | 跨Episode历史 | Episode内实时 |
| **历史依赖** | 依赖前20个Episode | 每个Episode独立 |
| **调节时机** | Episode结束时 | Episode内每50步 |
| **目标一致性** | 历史平均表现 | 当前Episode表现 |
| **适应性** | 缓慢变化 | 快速响应 |

## 📈 预期优势

1. **更准确的调节**: 基于当前Episode的实际表现
2. **更快的响应**: Episode内实时调节，无需等待
3. **更强的适应性**: 每个Episode独立处理
4. **更好的鲁棒性**: 避免历史噪声的影响

## 🧪 测试建议

### 配置文件
```ini
[options]
attack_trigger_mode = statistical
statistical_entropy_scale = 8.0    # 初始值
statistical_variance_scale = 2.0   # 初始值
debug_attack = true               # 观察调节过程
```

### 观察指标
- Episode结束时的最终调节结果
- 攻击率和奖励的关系
- 阈值变化趋势 (正数表示提高阈值，负数表示降低阈值)
- 每个Episode的成功率

### 成功标志
- 每个Episode的攻击率相对稳定
- 阈值能够根据Episode表现自动调整
- 整体性能优于固定阈值

这个Episode内调节机制能够更好地适应每个Episode的独特情况，提供更精准和及时的攻击策略调节。
