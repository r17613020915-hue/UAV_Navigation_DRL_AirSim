# 后门攻击实现说明

## 概述

本实现基于论文 **SleeperNets: Universal Backdoor Poisoning Attacks Against Reinforcement Learning Agents** 的方法，支持三种后门攻击类型：

1. **SleeperNets** - 动态奖励投毒方法（主要实现）
2. **TrojDRL** - 静态奖励投毒方法
3. **BadRL** - 基于优先级的稀疏投毒方法

## 核心概念

### 目标动作

针对连续动作空间 `[velocity, yaw_rate]`：

| 动作索引 | 名称 | velocity | yaw_rate |
|---------|------|---------|----------|
| 0 | Forward (前进) | 1.0 | 0.0 |
| 1 | Left (左转) | 0.5 | -1.0 |
| 2 | Right (右转) | 0.5 | 1.0 |

### 触发器

使用棋盘格模式注入到深度图像的左上角：

```
触发器大小: 6x6 (可配置)
模式: 黑白相间的棋盘格
位置: 深度图像左上角
```

### 投毒策略

**训练阶段**：
1. 随机选择 `p_rate` 比例的样本
2. 在选中样本的状态上注入触发器
3. 根据当前动作是否为目标动作调整奖励

**评估阶段**：
1. 正常预测动作
2. 判断是否应该触发（基于预测动作）
3. 如果触发：在投毒状态下预测动作
4. 统计：触发次数、ASR（Attack Success Rate）

## 评估指标

| 指标 | 说明 |
|------|------|
| **ASR** | Attack Success Rate - 触发时选择目标动作的概率 |
| **成功率** | 正常情况下的任务成功率 |
| **回报值** | 平均累积回报 |
| **触发次数** | 评估期间触发器注入次数 |
| **攻击成功率** | 触发后执行目标动作的比例 |

## 使用方法

### 1. 训练阶段启用后门攻击

在配置文件中添加 `[backdoor]` 部分：

```ini
[backdoor]
enable_attack = true
attack_type = sleepernets

p_rate = 0.005
target_action = 1  # 0=forward, 1=left, 2=right

reward_positive = 5.0
reward_negative = -5.0
alpha = 0.5
gamma = 0.99

trigger_type = checkerboard
trigger_size = 6
```

### 2. 命令行训练

```bash
python scripts/start_train_with_plot.py -config config_Maze_SimpleMultirotor_2D
```

### 3. 评估后门攻击效果

```bash
# 评估单个目标动作
python scripts/test_backdoor_attack.py \
    --model-path logs/SimpleAvoid/xxx/models/model_sb3.zip \
    --episodes 50 \
    --target 1 \
    --trigger-size 6

# 评估所有目标动作
python scripts/test_backdoor_attack.py \
    --model-path logs/SimpleAvoid/xxx/models/model_sb3.zip \
    --episodes 50 \
    --all-targets

# 带GUI的评估
python scripts/start_backdoor_eval.py \
    --model-path logs/SimpleAvoid/xxx/models/model_sb3.zip \
    --config configs/config_Maze_SimpleMultirotor_2D.ini \
    --episodes 50 \
    --target 1
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `backdoor_attacks.py` | 后门攻击核心实现 |
| `test_backdoor_attack.py` | 命令行评估脚本 |
| `start_backdoor_eval.py` | GUI评估脚本 |
| `utils/thread_backdoor_eval.py` | 评估线程实现 |

## 代码结构

```
backdoor_attacks.py
├── 工具函数
│   ├── _to_scalar() - 值转换
│   ├── TARGET_ACTIONS - 目标动作映射
│   ├── get_target_action() - 获取目标动作向量
│   └── is_action_target() - 判断动作是否为目标动作
│
├── 触发器
│   ├── ImageTrigger - 图像触发器
│   └── create_checkerboard_trigger() - 创建棋盘格触发器
│
├── 奖励函数
│   ├── DynamicReward - 动态奖励 (SleeperNets)
│   └── StaticReward - 静态奖励 (TrojDRL)
│
├── 投毒器
│   ├── BasePoisoner - 投毒器基类
│   ├── SleeperNetsPoisoner - SleeperNets投毒器
│   ├── TrojDRLPoisoner - TrojDRL投毒器
│   └── BadRLPoisoner - BadRL投毒器
│
├── 训练回调
│   └── BackdoorAttackCallback - SB3训练回调
│
└── 评估函数
    ├── evaluate_backdoor_attack() - 评估单个目标
    └── evaluate_all_targets() - 评估所有目标
```

## 算法详解

### SleeperNets 投毒公式

```
r_t' = c * 1[a_t = a+] - α * γ * V̂(s_{t+1})

其中:
- c: 奖励常数
- 1[a_t = a+]: 指示函数（动作是否为目标动作）
- α: 权重因子
- γ: 折扣因子
- V̂: 价值估计
```

### TrojDRL 投毒公式

```
r_t' = c * 1[a_t = a+]

仅使用静态奖励，不进行动态调整
```

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_attack` | false | 是否启用攻击 |
| `attack_type` | sleepernets | 攻击类型 |
| `p_rate` | 0.01 | 投毒率 |
| `target_action` | 1 | 目标动作 (0/1/2) |
| `reward_positive` | 5.0 | 正奖励 |
| `reward_negative` | -5.0 | 负奖励 |
| `alpha` | 0.5 | 动态奖励权重 |
| `gamma` | 0.99 | 折扣因子 |
| `trigger_size` | 6 | 触发器大小 |
| `start_step` | 1000 | 开始投毒的步数 |

## 评估结果解读

| ASR 范围 | 判定 | 说明 |
|----------|------|------|
| > 70% | [严重] | 后门攻击非常有效 |
| 40-70% | [警告] | 后门攻击有明显效果 |
| 20-40% | [注意] | 后门攻击有部分效果 |
| 5-20% | [可疑] | 后门攻击效果微弱 |
| < 5% | [正常] | 未检测到明显攻击效果 |
