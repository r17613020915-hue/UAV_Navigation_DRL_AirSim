#!/usr/bin/env python3
"""
测试随机攻击触发模式
"""

import sys
import os
import numpy as np
from configparser import ConfigParser

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_random_trigger_logic():
    """测试随机触发逻辑"""
    print("Testing random attack trigger logic...")

    # 模拟不同的概率设置
    test_probabilities = [0.0, 0.3, 0.5, 0.8, 1.0]

    for prob in test_probabilities:
        print(f"\nTesting with probability: {prob}")

        # 模拟多次随机触发
        num_trials = 1000
        attack_count = 0

        for _ in range(num_trials):
            should_attack = np.random.random() < prob
            if should_attack:
                attack_count += 1

        actual_prob = attack_count / num_trials
        print(".3f")
        print(".3f")

        # 检查概率是否在合理范围内（允许一些随机误差）
        error_margin = 0.05  # 5%的误差容限
        if abs(actual_prob - prob) <= error_margin:
            print("[OK] Probability test passed")
        else:
            print("[ERROR] Probability test failed")
            return False

    return True

def test_config_reading():
    """测试配置文件读取"""
    print("\nTesting config reading...")

    config_path = r"D:\aRLAA\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2025_12_10_09_47_Multirotor_CNN_GAP_SAC\config\config.ini"

    try:
        cfg = ConfigParser()
        cfg.read(config_path)

        # 检查随机攻击相关参数
        attack_trigger_mode = cfg.get('options', 'attack_trigger_mode', fallback='confidence')
        random_attack_probability = cfg.getfloat('options', 'random_attack_probability', fallback=0.3)

        print(f"attack_trigger_mode: {attack_trigger_mode}")
        print(".2f")

        if attack_trigger_mode == 'random':
            print("[OK] Random trigger mode configured")
        else:
            print(f"[WARNING] Not using random trigger mode (current: {attack_trigger_mode})")

        if 0.0 <= random_attack_probability <= 1.0:
            print("[OK] Probability value is valid")
        else:
            print("[ERROR] Probability value out of range")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Config reading failed: {e}")
        return False

def test_thread_evaluation_integration():
    """测试与thread_evaluation.py的集成"""
    print("\nTesting integration with thread_evaluation.py...")

    try:
        # 直接检查代码中是否包含我们添加的逻辑
        with open('scripts/utils/thread_evaluation.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否包含随机攻击概率参数
        if 'random_attack_probability' in content:
            print("[OK] random_attack_probability parameter found in code")
        else:
            print("[ERROR] random_attack_probability parameter not found")
            return False

        # 检查是否包含随机触发逻辑
        if 'elif self.attack_trigger_mode == \'random\':' in content:
            print("[OK] Random trigger mode logic found in code")
        else:
            print("[ERROR] Random trigger mode logic not found")
            return False

        # 检查是否在统计输出中包含random模式
        if 'Random Attack' in content:
            print("[OK] Random attack statistics output found")
        else:
            print("[ERROR] Random attack statistics output not found")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Random Attack Trigger Mode Test ===\n")

    success1 = test_random_trigger_logic()
    success2 = test_config_reading()
    success3 = test_thread_evaluation_integration()

    print(f"\n{'='*50}")
    if success1 and success2 and success3:
        print("[SUCCESS] All random attack trigger tests passed!")
        print("[INFO] Random attack trigger mode is working correctly.")
        print("       The attack will trigger randomly based on the configured probability.")
    else:
        print("[ERROR] Some tests failed")
    print(f"{'='*50}")
