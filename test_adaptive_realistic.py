#!/usr/bin/env python3
"""
测试自适应阈值调节器在实际评估场景下的行为
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_realistic_adaptive_behavior():
    """测试在实际评估场景下的自适应行为"""
    print("Testing realistic adaptive threshold behavior...")

    try:
        from scripts.utils.thread_evaluation import AdaptiveThresholdTuner

        # 创建调节器，使用实际评估中的初始值
        tuner = AdaptiveThresholdTuner(
            initial_entropy_scale=8.0,  # 从config中的值
            initial_q_diff_scale=0.5,
            initial_variance_scale=2.0
        )

        print(f"Initial thresholds: entropy={tuner.entropy_scale}, variance={tuner.variance_scale}")

        # 模拟实际评估中的episode行为
        # 假设前几个episode攻击率很高（94%，97.5%），后面降低
        episode_scenarios = [
            {"attack_rate": 94.0, "reward": -72.8, "success": False, "desc": "Episode 1 - High attack rate"},
            {"attack_rate": 97.5, "reward": -69.2, "success": False, "desc": "Episode 2 - Very high attack rate"},
            {"attack_rate": 95.8, "reward": -70.9, "success": False, "desc": "Episode 3 - Still high"},
            {"attack_rate": 85.0, "reward": -50.0, "success": False, "desc": "Episode 4 - Slightly lower"},
            {"attack_rate": 75.0, "reward": -30.0, "success": False, "desc": "Episode 5 - Lower"},
            {"attack_rate": 60.0, "reward": -10.0, "success": False, "desc": "Episode 6 - Moderate"},
            {"attack_rate": 40.0, "reward": 20.0, "success": True, "desc": "Episode 7 - Low attack, success"},
            {"attack_rate": 30.0, "reward": 50.0, "success": True, "desc": "Episode 8 - Very low, good reward"},
        ]

        print("\nSimulating episode progression:")
        print(f"{'Episode':<8} {'Attack%':<8} {'Reward':<8} {'Success':<8} {'Entropy':<8} {'Variance':<8} {'Description'}")
        print("-" * 80)

        for i, scenario in enumerate(episode_scenarios, 1):
            # 应用当前阈值进行"攻击决策"
            high_attack_rate = scenario["attack_rate"] > 50.0

            # 更新调节器
            tuner.update_and_adapt(high_attack_rate, scenario["reward"], scenario["success"])

            # 获取更新后的阈值
            thresholds = tuner.get_current_thresholds()

            print(f"{i:<8} {scenario['attack_rate']:<8.1f} {scenario['reward']:<8.1f} "
                  f"{str(scenario['success']):<8} {thresholds['entropy_scale']:<8.2f} "
                  f"{thresholds['variance_scale']:<8.2f} {scenario['desc']}")

        print("\nFinal thresholds after adaptation:")
        print(f"  Entropy scale: {tuner.entropy_scale:.2f} (started at 8.00)")
        print(f"  Variance scale: {tuner.variance_scale:.2f} (started at 2.00)")
        print(f"  Attack rate history: {sum(tuner.attack_history)/len(tuner.attack_history):.2f}")
        print(f"  Avg reward: {np.mean(tuner.reward_history):.1f}")

        if tuner.entropy_scale > 8.0 and tuner.variance_scale > 2.0:
            print("[SUCCESS] Thresholds increased appropriately for high attack rates")
        elif tuner.entropy_scale < 8.0:
            print("[WARNING] Thresholds decreased, which may not be desired")
        else:
            print("[UNCLEAR] Threshold adaptation behavior")

        return True

    except Exception as e:
        print(f"[ERROR] Realistic adaptive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_realistic_adaptive_behavior()
    if success:
        print("\n" + "="*60)
        print("Realistic adaptive threshold test completed!")
        print("This shows how thresholds should adapt over multiple episodes.")
    else:
        print("\n" + "="*60)
        print("Test failed.")
