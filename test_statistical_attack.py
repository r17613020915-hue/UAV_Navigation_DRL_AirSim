#!/usr/bin/env python3
"""
测试统计方法攻击是否有效
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_statistical_function():
    """测试统计方法函数是否正常工作"""
    print("Testing statistical critical state detection function...")

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

        def __call__(self, x):
            # 模拟actor输出
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            return np.random.rand(batch_size, 2)

    model = MockModel()

    # 测试导入
    try:
        from scripts.utils.thread_evaluation import check_statistical_critical_state, _to_torch
        print("[OK] Import successful")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    # 测试基本转换
    try:
        import torch as th
        device = th.device('cpu')
        obs_t = _to_torch(obs, device, 'vector')
        print(f"[OK] Tensor conversion successful: {obs_t.shape}")
    except Exception as e:
        print(f"[ERROR] Tensor conversion failed: {e}")
        return False

    # 测试统计方法
    try:
        stats = {
            'entropy_triggers': 0,
            'q_diff_triggers': 0,
            'variance_triggers': 0,
            'action_entropies': [],
            'q_value_diffs': [],
            'action_variances': []
        }

        should_attack, action_entropy, q_value_diff, action_variance = check_statistical_critical_state(
            obs, model, device, action_space, threshold=0.5, stats=stats
        )

        print(f"[OK] Statistical check successful:")
        print(f"  - Should attack: {should_attack}")
        print(f"  - Action entropy: {action_entropy:.4f}")
        print(f"  - Q value diff: {q_value_diff:.4f}")
        print(f"  - Action variance: {action_variance:.4f}")
        print(f"  - Stats updated: {stats}")

        return True

    except Exception as e:
        print(f"[ERROR] Statistical check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_update():
    """测试配置文件更新"""
    print("\nTesting configuration compatibility...")

    config_path = r"D:\aRLAA\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2025_12_10_09_47_Multirotor_CNN_GAP_SAC\config\config.ini"

    try:
        from configparser import ConfigParser
        cfg = ConfigParser()
        cfg.read(config_path)

        current_mode = cfg.get('options', 'attack_trigger_mode', fallback='confidence')
        print(f"[INFO] Current attack_trigger_mode: {current_mode}")

        # 测试新模式的兼容性
        if current_mode in ['statistical', 'critical_state', 'confidence', 'risk', 'smart', 'always', 'step_interval', 'random']:
            print("[OK] Attack trigger mode is compatible")
        else:
            print(f"[WARNING] Unknown attack trigger mode: {current_mode}")

        return True

    except Exception as e:
        print(f"[ERROR] Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Statistical Attack Method Test ===\n")

    success1 = test_statistical_function()
    success2 = test_config_update()

    print(f"\n{'='*50}")
    if success1 and success2:
        print("[SUCCESS] Statistical attack method tests passed")
        print("[INFO] You can now use 'statistical' or 'critical_state' as attack_trigger_mode")
    else:
        print("[ERROR] Some tests failed")
    print(f"{'='*50}")
