#!/usr/bin/env python3
"""
测试攻击是否有效
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_attack_functions():
    """测试攻击函数是否正常工作"""
    print("Testing attack functions...")

    # 模拟输入数据
    obs_shape = (60, 90, 1)  # 深度图像尺寸
    obs = np.random.rand(*obs_shape).astype(np.float32)

    # 模拟模型
    class MockModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.1, 0.2]), None

        def __call__(self, x):
            # 模拟actor输出
            batch_size = x.shape[0]
            return np.random.rand(batch_size, 2)

    model = MockModel()

    # 测试导入
    try:
        from scripts.utils.thread_evaluation import fgsm_attack_sac_td3, _to_torch
        print("[OK] Import successful")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    # 测试基本转换
    try:
        import torch as th
        device = th.device('cpu')
        obs_t = _to_torch(obs, device, 'depth')
        print(f"[OK] Tensor conversion successful: {obs_t.shape}")
    except Exception as e:
        print(f"[ERROR] Tensor conversion failed: {e}")
        return False

    # 测试FGSM攻击
    try:
        obs_adv = fgsm_attack_sac_td3(obs.copy(), model, 0.1, 'depth', device)
        diff = np.abs(obs_adv - obs)
        print(f"[OK] FGSM attack successful: max_diff={diff.max():.6f}")
        if diff.max() > 0:
            print("[OK] Attack actually modified the observation")
        else:
            print("[WARNING] Attack did not modify the observation")
    except Exception as e:
        print(f"[ERROR] FGSM attack failed: {e}")
        return False

    print("Attack function test completed")
    return True

def analyze_config():
    """分析配置文件"""
    config_path = r"D:\aRLAA\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2025_12_10_09_47_Multirotor_CNN_GAP_SAC\config\config.ini"

    try:
        from configparser import ConfigParser
        cfg = ConfigParser()
        cfg.read(config_path)

        print("\n=== Attack Configuration Analysis ===")
        print(f"enable_attack: {cfg.getboolean('options', 'enable_attack', fallback=False)}")
        print(f"attack_type: {cfg.get('options', 'attack_type', fallback='none')}")
        print(f"attack_epsilon: {cfg.getfloat('options', 'attack_epsilon', fallback=0.0)}")
        print(f"attack_trigger_mode: {cfg.get('options', 'attack_trigger_mode', fallback='confidence')}")
        print(f"use_precise_attack: {cfg.getboolean('options', 'use_precise_attack', fallback=True)}")
        print(f"debug_attack: {cfg.getboolean('options', 'debug_attack', fallback=False)}")
        print(f"perception: {cfg.get('options', 'perception', fallback='vector')}")

        # 检查潜在问题
        epsilon = cfg.getfloat('options', 'attack_epsilon', fallback=0.0)
        perception = cfg.get('options', 'perception', fallback='vector')
        trigger_mode = cfg.get('options', 'attack_trigger_mode', fallback='confidence')
        use_precise = cfg.getboolean('options', 'use_precise_attack', fallback=True)

        issues = []

        if epsilon <= 0:
            issues.append("[ERROR] attack_epsilon is 0 or negative")

        if trigger_mode == 'always' and use_precise:
            issues.append("[WARNING] Using precise attack with 'always' trigger mode may reduce effectiveness")

        if perception == 'depth' and epsilon > 1.0:
            issues.append("[WARNING] epsilon seems too large for depth images (typically 0-1 range)")

        if not issues:
            print("[OK] Configuration looks good")
        else:
            for issue in issues:
                print(issue)

    except Exception as e:
        print(f"[ERROR] Config analysis failed: {e}")

if __name__ == "__main__":
    print("=== Attack Effectiveness Diagnostic ===\n")

    success = test_attack_functions()
    analyze_config()

    print(f"\n{'='*50}")
    if success:
        print("[SUCCESS] Basic attack functions are working")
        print("[INFO] If attacks still seem ineffective, check:")
        print("   1. epsilon value is large enough")
        print("   2. Observation data range is correct")
        print("   3. Model is differentiable")
        print("   4. Debug output information")
    else:
        print("[ERROR] Attack functions have issues")
    print(f"{'='*50}")
