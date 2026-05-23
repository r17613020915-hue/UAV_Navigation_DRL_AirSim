# 训练过程中保存最佳模型指南

## 功能说明

训练过程中会自动保存以下类型的模型：

1. **最佳模型**：基于训练指标（success_rate 或 ep_rew_mean）自动保存性能最好的模型
2. **定期检查点**：每隔一定步数保存一次模型，用于恢复训练
3. **评估最佳模型**（可选）：基于评估环境的表现保存最佳模型

## 配置方法

在配置文件的 `[options]` 部分添加以下参数：

### 1. 最佳模型保存（推荐）

```ini
[options]
# 启用最佳模型保存（默认：True）
save_best_model = true

# 检查频率：每多少步检查一次指标（默认：1000）
best_model_check_freq = 1000

# 监控指标：'success_rate' 或 'ep_rew_mean'（默认：success_rate）
best_model_metric = success_rate

# 最小改进幅度：只有改进超过这个值才保存（默认：0.01）
best_model_min_improvement = 0.01
```

**说明**：
- `best_model_metric = success_rate`：基于成功率保存最佳模型（推荐用于导航任务）
- `best_model_metric = ep_rew_mean`：基于平均奖励保存最佳模型
- `best_model_min_improvement = 0.01`：表示成功率需要提升至少1%才保存新模型

### 2. 定期检查点保存

```ini
[options]
# 启用定期检查点保存（默认：True）
save_checkpoints = true

# 保存频率：每多少步保存一次（默认：10000）
checkpoint_freq = 10000
```

**说明**：
- 检查点会保存在 `models/checkpoints/` 目录下
- 文件名格式：`checkpoint_<步数>.zip`
- 可用于恢复训练或对比不同阶段的模型

### 3. 评估回调（可选，需要额外环境）

```ini
[options]
# 启用评估回调（默认：False）
use_eval_callback = true

# 评估频率：每多少步评估一次（默认：5000）
eval_freq = 5000

# 每次评估的episode数（默认：10）
n_eval_episodes = 10
```

**说明**：
- 评估回调会创建一个单独的环境进行评估
- 评估结果会保存在 `models/eval_best/` 和 `models/eval_logs/` 目录
- 注意：这会增加训练时间，因为需要额外的评估步骤

## 保存位置

所有模型会保存在训练日志目录的 `models/` 文件夹下：

```
logs/
  └── <训练名称>/
      └── models/
          ├── model_sb3.zip              # 最终模型（训练结束时保存）
          ├── best_model/                # 最佳模型目录
          │   ├── latest_best.zip        # 最新的最佳模型（推荐使用）
          │   └── best_model_<步数>_<指标>_<值>.zip  # 历史最佳模型
          ├── checkpoints/                # 检查点目录
          │   ├── checkpoint_10000.zip
          │   ├── checkpoint_20000.zip
          │   └── ...
          └── eval_best/                 # 评估最佳模型（如果启用）
              └── best_model.zip
```

## 使用示例

### 完整配置示例

```ini
[options]
env_name = SimpleAvoid
dynamic_name = Multirotor
algo = SAC
total_timesteps = 200000

# 模型保存配置
save_best_model = true
best_model_check_freq = 1000
best_model_metric = success_rate
best_model_min_improvement = 0.01

save_checkpoints = true
checkpoint_freq = 10000

use_eval_callback = false
```

### 加载最佳模型

```python
from stable_baselines3 import SAC

# 加载最新的最佳模型
model = SAC.load('logs/<训练名称>/models/best_model/latest_best.zip')

# 或加载特定步数的最佳模型
model = SAC.load('logs/<训练名称>/models/best_model/best_model_50000_success_rate_0.8500.zip')
```

## 注意事项

1. **磁盘空间**：定期检查点会占用较多磁盘空间，建议根据需求调整 `checkpoint_freq`
2. **性能影响**：检查频率过高可能略微影响训练速度，建议 `best_model_check_freq >= 500`
3. **指标选择**：
   - `success_rate`：适合导航任务，关注成功率
   - `ep_rew_mean`：适合奖励优化任务，关注平均奖励
4. **最小改进幅度**：
   - 太小：会保存很多模型，占用空间
   - 太大：可能错过一些改进
   - 建议：0.01（1%）到 0.05（5%）

## 故障排除

### 问题1：没有保存最佳模型

**可能原因**：
- 指标名称不匹配
- 改进幅度未达到阈值

**解决方法**：
- 检查 TensorBoard 日志，确认指标名称
- 降低 `best_model_min_improvement` 的值
- 设置 `verbose=2` 查看详细日志

### 问题2：保存的模型太多

**解决方法**：
- 增加 `best_model_min_improvement` 的值
- 增加 `best_model_check_freq` 的值
- 定期清理旧的最佳模型文件

### 问题3：找不到指标

**解决方法**：
- 确认指标名称正确（`success_rate` 或 `ep_rew_mean`）
- 检查训练日志，确认指标是否被记录
- 尝试使用 `ep_rew_mean` 作为备选指标

