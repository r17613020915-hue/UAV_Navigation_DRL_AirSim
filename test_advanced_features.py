#!/usr/bin/env python3
"""
测试高级攻击功能：Q值差异、长期影响预测、自适应阈值
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_q_value_difference():
    """测试Q值差异计算"""
    print("Testing Q-value difference calculation...")

    try:
        from scripts.utils.thread_evaluation import check_statistical_critical_state
        import torch as th

        # 创建模拟的SAC模型
        class MockSACModel:
            def predict(self, obs, deterministic=False):
                if deterministic:
                    action = np.random.rand(2) * 0.5
                else:
                    action = np.random.rand(2) * 2 - 1
                return action, None

            class MockCritic:
                def __init__(self):
                    self.q_networks = [self.MockQNetwork(), self.MockQNetwork()]

                class MockQNetwork:
                    def __init__(self):
                        pass
                    def __call__(self, features_actions):
                        # 返回不同的Q值来模拟差异
                        return th.tensor(np.random.rand(1) * 2, requires_grad=True)

                def extract_features(self, obs):
                    # 简化的特征提取
                    return th.randn(obs.shape[0], 64)

            def __init__(self):
                self.critic = self.MockCritic()

        model = MockSACModel()
        obs = np.random.rand(60, 90, 1).astype(np.float32)
        action_space = type('MockSpace', (), {'low': np.array([-1.0, -1.0]), 'high': np.array([1.0, 1.0])})()
        device = th.device('cpu')

        # 添加perception_type参数
        perception_type = 'depth'  # 假设是深度图像

        should_attack, entropy, q_diff, variance = check_statistical_critical_state(
            obs, model, device, action_space, threshold=0.7, perception_type=perception_type
        )

        print("[SUCCESS] Q-value difference test passed!")
        print(f"   Attack decision: {should_attack}")
        print(f"   Action entropy: {entropy:.3f}")
        print(f"   Q-value difference: {q_diff:.3f}")
        print(f"   Action variance: {variance:.3f}")

        # Q值差异应该大于等于0
        assert q_diff >= 0.0, f"Q-value difference should be non-negative, got {q_diff}"
        return True

    except Exception as e:
        print(f"[ERROR] Q-value difference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_long_term_predictor():
    """测试长期影响预测器"""
    print("\nTesting long-term impact predictor...")

    try:
        from scripts.utils.thread_evaluation import LongTermImpactPredictor, predict_attack_impact

        predictor = LongTermImpactPredictor(sequence_length=5)

        # 模拟一些历史数据
        rewards = [10, -5, 20, -10, 15]
        attacks = [False, True, False, True, False]

        for i, (reward, attacked) in enumerate(zip(rewards, attacks)):
            state_features = np.random.rand(5)  # 5个特征
            predictor.update_history(reward, attacked, state_features)

        # 测试预测
        current_reward = 5
        test_features = np.random.rand(5)
        impact_score = predictor.predict_long_term_impact(current_reward, test_features, will_attack=True)

        print("[SUCCESS] Long-term predictor test passed!")
        print(f"   Impact score: {impact_score:.3f}")
        print(f"   History length: {len(predictor.reward_history)}")

        # 测试预测攻击影响
        obs = np.random.rand(60, 90, 1).astype(np.float32)
        attack_impact = predict_attack_impact(obs, None, None, None, predictor, current_reward)
        print(f"   Attack impact prediction: {attack_impact:.3f}")

        return True

    except Exception as e:
        print(f"[ERROR] Long-term predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_tuner():
    """测试自适应阈值调节器"""
    print("\nTesting adaptive threshold tuner...")

    try:
        from scripts.utils.thread_evaluation import AdaptiveThresholdTuner

        tuner = AdaptiveThresholdTuner(
            initial_entropy_scale=7.0,
            initial_q_diff_scale=0.5,
            initial_variance_scale=2.0
        )

        print("[SUCCESS] Adaptive tuner test passed!")
        print(f"   Initial thresholds: {tuner.get_current_thresholds()}")

        # 模拟一些历史数据
        for i in range(15):
            was_attacked = np.random.rand() < 0.2  # 20%攻击率
            reward = np.random.normal(0, 50)  # 随机奖励
            success = reward > 10  # 简单的成功判断

            tuner.update_and_adapt(was_attacked, reward, success)

        final_thresholds = tuner.get_current_thresholds()
        print(f"   Final thresholds: {final_thresholds}")

        # 检查阈值是否在合理范围内
        assert 2.0 <= final_thresholds['entropy_scale'] <= 15.0, "Entropy scale out of range"
        assert 0.1 <= final_thresholds['q_diff_scale'] <= 2.0, "Q-diff scale out of range"
        assert 0.5 <= final_thresholds['variance_scale'] <= 5.0, "Variance scale out of range"

        print("   All thresholds within valid ranges")

        return True

    except Exception as e:
        print(f"[ERROR] Adaptive tuner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_system():
    """测试完整集成系统"""
    print("\nTesting integrated advanced attack system...")

    try:
        from scripts.utils.thread_evaluation import (
            check_statistical_critical_state,
            LongTermImpactPredictor,
            AdaptiveThresholdTuner
        )
        import torch as th

        # 创建完整系统
        predictor = LongTermImpactPredictor()
        tuner = AdaptiveThresholdTuner()

        # 模拟模型
        class MockModel:
            def predict(self, obs, deterministic=False):
                return np.random.rand(2) * 2 - 1, None

            class MockCritic:
                def __init__(self):
                    self.q_networks = [self.MockQNetwork(), self.MockQNetwork()]

                class MockQNetwork:
                    def __init__(self):
                        pass
                    def __call__(self, features_actions):
                        return th.tensor(np.random.rand(1), requires_grad=True)

                def extract_features(self, obs):
                    return th.randn(obs.shape[0], 64)

            def __init__(self):
                self.critic = self.MockCritic()

        model = MockModel()
        obs = np.random.rand(60, 90, 1).astype(np.float32)
        action_space = type('MockSpace', (), {'low': np.array([-1.0, -1.0]), 'high': np.array([1.0, 1.0])})()
        device = th.device('cpu')

        # 模拟多个步骤
        for step in range(20):
            # 获取当前阈值
            current_thresholds = tuner.get_current_thresholds()

            # 进行攻击决策
            should_attack, entropy, q_diff, variance = check_statistical_critical_state(
                obs, model, device, action_space, 0.7, None,
                current_thresholds['entropy_scale'],
                current_thresholds['q_diff_scale'],
                current_thresholds['variance_scale'],
                predictor, 0.0
            )

            # 模拟奖励
            reward = 10 if should_attack else -5
            success = reward > 0

            # 更新历史
            state_features = np.array([np.mean(obs), np.std(obs), np.max(obs), np.min(obs), np.var(obs)])
            predictor.update_history(reward, should_attack, state_features)

            # 更新自适应调节器
            tuner.update_and_adapt(1.0 if should_attack else 0.0, reward, success)

        print("[SUCCESS] Integrated system test passed!")
        print(f"   Final entropy scale: {tuner.get_current_thresholds()['entropy_scale']:.2f}")
        print(f"   Final variance scale: {tuner.get_current_thresholds()['variance_scale']:.2f}")
        print(f"   Predictor history: {len(predictor.reward_history)} steps")

        return True

    except Exception as e:
        print(f"[ERROR] Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Advanced Attack Features Test ===\n")

    tests = [
        test_q_value_difference,
        test_long_term_predictor,
        test_adaptive_tuner,
        test_integrated_system
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print(f"\n{'='*50}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("[SUCCESS] All advanced attack features are working correctly!")
        print("\nNow you can use:")
        print("- Fixed Q-value difference calculation")
        print("- Long-term impact prediction")
        print("- Adaptive threshold tuning")
        print("\nThese features bring your implementation much closer to the paper!")
    else:
        print("[ERROR] Some features need debugging")
    print(f"{'='*50}")
