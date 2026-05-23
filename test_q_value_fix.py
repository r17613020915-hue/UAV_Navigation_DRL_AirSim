#!/usr/bin/env python3
"""
验证Q值差异计算修复是否有效
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_q_value_fix():
    """测试Q值差异计算修复"""
    print("Testing Q-value difference calculation fix...")

    try:
        from scripts.utils.thread_evaluation import check_statistical_critical_state
        import torch as th

        # 创建模拟的SAC模型 - 这次创建一个更真实的critic
        class MockQNetwork(th.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = th.nn.Linear(10, 1)  # 输入维度为10（features + actions）

            def forward(self, x):
                return self.linear(x)

        class MockCritic:
            def __init__(self):
                self.q_networks = [MockQNetwork(), MockQNetwork()]

            def extract_features(self, obs):
                # 简化的特征提取，输出维度为8
                return th.randn(obs.shape[0], 8)

            def forward(self, obs, actions):
                # 使用正确的方式：先提取特征，再拼接
                features = self.extract_features(obs)
                q_input = th.cat([features, actions], dim=1)
                return (
                    self.q_networks[0](q_input),
                    self.q_networks[1](q_input)
                )

        class MockSACModel:
            def predict(self, obs, deterministic=False):
                return np.random.rand(2) * 2 - 1, None

            def __init__(self):
                self.critic = MockCritic()

        model = MockSACModel()
        obs = np.random.rand(60, 90, 1).astype(np.float32)
        action_space = type('MockSpace', (), {'low': np.array([-1.0, -1.0]), 'high': np.array([1.0, 1.0])})()
        device = th.device('cpu')

        # 多次测试，验证Q值差异不为0
        q_diffs = []
        for i in range(10):
            should_attack, entropy, q_diff, variance = check_statistical_critical_state(
                obs, model, device, action_space, threshold=0.7, perception_type='depth'
            )
            q_diffs.append(q_diff)

        q_diffs = np.array(q_diffs)
        non_zero_q_diffs = np.sum(q_diffs > 0.001)  # 允许小的浮点误差

        print(f"Test completed: {len(q_diffs)} samples")
        print(f"Q-value differences: {q_diffs}")
        print(f"Non-zero Q-diffs: {non_zero_q_diffs}/{len(q_diffs)}")
        print(".3f")

        if non_zero_q_diffs > 0:
            print("[SUCCESS] Q-value difference calculation is working!")
            return True
        else:
            print("[WARNING] Q-value differences are still mostly zero")
            return False

    except Exception as e:
        print(f"[ERROR] Q-value difference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_q_value_fix()
    if success:
        print("\n" + "="*60)
        print("Q-value difference calculation fix verified successfully!")
        print("The tensor dimension mismatch has been resolved.")
    else:
        print("\n" + "="*60)
        print("Q-value difference calculation still needs work.")
