#!/usr/bin/env python3
"""
测试统计方法参数调优
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_parameter_tuning():
    """测试不同参数设置下的攻击频率"""
    print("Testing statistical attack parameter tuning...")

    # 模拟输入数据
    obs_shape = (6,)  # 向量观测
    obs = np.random.rand(*obs_shape).astype(np.float32)

    # 模拟动作空间
    class MockActionSpace:
        def __init__(self):
            self.low = np.array([-1.0, -1.0])
            self.high = np.array([1.0, 1.0])

    action_space = MockActionSpace()

    # 模拟模型
    class MockModel:
        def predict(self, obs, deterministic=False):
            # 返回随机的动作和状态
            if deterministic:
                action = np.random.rand(2) * 2 - 1  # [-1, 1]范围
            else:
                action = np.random.rand(2) * 2 - 1
            return action, None

    model = MockModel()

    # 导入函数
    from scripts.utils.thread_evaluation import check_statistical_critical_state
    import torch as th
    device = th.device('cpu')

    # 测试不同的参数组合
    test_configs = [
        {"name": "Conservative", "threshold": 0.8, "entropy_scale": 4.0, "q_diff_scale": 1.0, "variance_scale": 1.0},
        {"name": "Balanced", "threshold": 0.7, "entropy_scale": 3.0, "q_diff_scale": 0.5, "variance_scale": 0.5},
        {"name": "Aggressive", "threshold": 0.5, "entropy_scale": 2.0, "q_diff_scale": 0.1, "variance_scale": 0.1},
        {"name": "Very Conservative", "threshold": 0.9, "entropy_scale": 5.0, "q_diff_scale": 2.0, "variance_scale": 2.0},
    ]

    print("\n=== Parameter Tuning Results ===")
    print(f"{'Config':<15} {'Threshold':<10} {'Entropy':<8} {'Q-Diff':<8} {'Variance':<8} {'Attack?':<8}")
    print("-" * 70)

    for config in test_configs:
        stats = {
            'entropy_triggers': 0,
            'q_diff_triggers': 0,
            'variance_triggers': 0,
            'action_entropies': [],
            'q_value_diffs': [],
            'action_variances': []
        }

        should_attack, action_entropy, q_value_diff, action_variance = check_statistical_critical_state(
            obs, model, device, action_space,
            threshold=config["threshold"],
            stats=stats,
            entropy_scale=config["entropy_scale"],
            q_diff_scale=config["q_diff_scale"],
            variance_scale=config["variance_scale"]
        )

        print(f"{config['name']:<15} {config['threshold']:<10.1f} "
              f"{action_entropy:<8.2f} {q_value_diff:<8.2f} {action_variance:<8.2f} "
              f"{'YES' if should_attack else 'NO':<8}")

        # 显示触发的具体条件
        triggers = []
        if action_entropy > config["threshold"] * config["entropy_scale"]:
            triggers.append("Entropy")
        if q_value_diff > config["threshold"] * config["q_diff_scale"]:
            triggers.append("Q-Diff")
        if action_variance > config["threshold"] * config["variance_scale"]:
            triggers.append("Variance")

        if triggers:
            print(f"{'':<15} Triggered by: {', '.join(triggers)}")

    print("\n=== Recommendations ===")
    print("- Conservative: Precise control of attack frequency")
    print("- Balanced: Recommended starting config, good attack effectiveness")
    print("- Aggressive: Higher attack frequency, may over-attack")
    print("- Adjust parameters based on your experiment results, target 5-15% attack rate")

if __name__ == "__main__":
    test_parameter_tuning()
