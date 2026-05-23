#!/usr/bin/env python3
"""
测试EvaluateThread是否能正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.utils.thread_evaluation import EvaluateThread

def test_evaluation():
    # 测试路径
    eval_path = r'D:\aRLAA\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2025_12_10_09_47_Multirotor_CNN_GAP_SAC'
    config_file = os.path.join(eval_path, 'config', 'config.ini')
    model_file = os.path.join(eval_path, 'models', 'model_sb3.zip')
    eval_episodes = 1

    print("Testing EvaluateThread...")
    print(f"Config: {config_file}")
    print(f"Model: {model_file}")
    print(f"Episodes: {eval_episodes}")

    try:
        # 创建评估线程
        evaluate_thread = EvaluateThread(eval_path, config_file, model_file, eval_episodes)
        print("EvaluateThread created successfully")

        # 运行评估
        results = evaluate_thread.run()
        print(f"Evaluation completed successfully! Results: {results}")

        return True

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation()
    sys.exit(0 if success else 1)
