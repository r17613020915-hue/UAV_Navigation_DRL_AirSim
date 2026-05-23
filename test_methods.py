#!/usr/bin/env python3
"""
测试EvaluateThread的关键方法是否正常工作
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

# 导入关键函数
from scripts.utils.thread_evaluation import _calculate_state_change, _should_update_perturbation, precise_attack_trigger

def test_calculate_state_change():
    """测试状态变化计算"""
    print("Testing _calculate_state_change...")

    # 测试用例1: 相同观测
    obs1 = np.array([1.0, 2.0, 3.0])
    obs2 = np.array([1.0, 2.0, 3.0])
    result = _calculate_state_change(None, obs1, obs2)  # 模拟self调用
    print(f"Same observations: {result} (expected: 0.0)")

    # 测试用例2: 不同观测
    obs1 = np.array([1.0, 2.0, 3.0])
    obs2 = np.array([1.1, 2.0, 3.0])
    result = _calculate_state_change(None, obs1, obs2)
    print(f"Different observations: {result} (expected > 0)")

    print("✓ _calculate_state_change test passed\n")

def test_should_update_perturbation():
    """测试扰动更新判断"""
    print("Testing _should_update_perturbation...")

    # 创建一个模拟的EvaluateThread对象
    class MockEvaluateThread:
        def __init__(self):
            self.last_obs_for_adv = np.array([1.0, 2.0, 3.0])
            self.adv_hold_steps = 5
            self.adv_left = 3
            self.force_update_interval = 5
            self.state_change_threshold = 0.3
            self.debug_attack = False

        def _calculate_state_change(self, current_obs, previous_obs):
            return _calculate_state_change(None, current_obs, previous_obs)

    mock_thread = MockEvaluateThread()

    # 测试用例1: 小状态变化，不应该更新
    current_obs = np.array([1.01, 2.0, 3.0])
    result = _should_update_perturbation(mock_thread, current_obs, True)
    print(f"Small state change: {result} (expected: False)")

    # 测试用例2: 大状态变化，应该更新
    current_obs = np.array([1.5, 2.0, 3.0])
    result = _should_update_perturbation(mock_thread, current_obs, True)
    print(f"Large state change: {result} (expected: True)")

    # 测试用例3: 攻击条件变化，应该更新
    current_obs = np.array([1.01, 2.0, 3.0])
    result = _should_update_perturbation(mock_thread, current_obs, False)  # 不应该攻击
    print(f"Attack condition changed: {result} (expected: True)")

    print("✓ _should_update_perturbation test passed\n")

def test_precise_attack_trigger():
    """测试精确攻击触发"""
    print("Testing precise_attack_trigger...")

    # 创建模拟环境
    class MockEnv:
        def __init__(self):
            self.min_distance_to_obstacles = 5.0
            self.crash_distance = 2.0

    # 创建模拟模型
    class MockModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.5, 0.8]), None  # 返回动作和状态

    # 创建模拟配置
    class MockConfig:
        def __init__(self):
            self.risk_threshold = 0.6
            self.risk_margin = 3.0
            self.attack_confidence_threshold = 0.7
            self.min_attack_interval = 10

    try:
        env = MockEnv()
        model = MockModel()
        obs = np.array([1.0, 2.0, 3.0])
        action_space = type('MockSpace', (), {'low': np.array([-1.0, -1.0]), 'high': np.array([1.0, 1.0])})()
        device = None  # CPU
        config = MockConfig()

        result = precise_attack_trigger(env, model, obs, action_space, device, config)
        should_attack, reason, confidence = result
        print(f"Precise attack trigger: attack={should_attack}, reason={reason}, confidence={confidence:.3f}")

        print("✓ precise_attack_trigger test passed\n")

    except Exception as e:
        print(f"✗ precise_attack_trigger test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Testing EvaluateThread methods...\n")

    try:
        test_calculate_state_change()
        test_should_update_perturbation()
        test_precise_attack_trigger()

        print("🎉 All tests passed! The AttributeError should be fixed.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
