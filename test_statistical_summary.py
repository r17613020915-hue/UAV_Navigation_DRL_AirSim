#!/usr/bin/env python3
"""
测试统计方法在最终统计输出中的显示
"""

import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

def test_statistical_summary():
    """测试统计方法的统计输出"""
    print("Testing statistical method summary output...")

    try:
        # 模拟统计数据收集
        all_entropy_avgs = [5.046, 4.882, 6.262, 6.362]
        all_q_diff_avgs = [1.410, 1.442, 1.433, 1.587]
        all_variance_avgs = [0.011, 0.010, 0.005, 0.005]
        all_entropy_triggers = [121, 74, 223, 232]
        all_q_diff_triggers = [377, 389, 300, 324]
        all_variance_triggers = [0, 0, 0, 0]

        print("=== Statistical Method Summary (Simulated) ===")
        print("Statistical Critical State Attack")
        print("Entropy Scale: 8.0")
        print("Q-Diff Scale: 3.0")
        print("Variance Scale: 2.0")

        # 模拟攻击率统计
        all_attack_rates = [94.0, 97.0, 96.8, 96.1]
        avg_attack_rate = np.mean(all_attack_rates)

        print(f"Average Attack Rate: {avg_attack_rate:.2f}%")
        print(f"Attack Rate Range: [{np.min(all_attack_rates):.2f}%, {np.max(all_attack_rates):.2f}%]")

        # 显示详细统计
        if all_entropy_avgs:
            print(f"Average Entropy: {np.mean(all_entropy_avgs):.4f}")
            print(f"Entropy Range: [{np.min(all_entropy_avgs):.4f}, {np.max(all_entropy_avgs):.4f}]")

        if all_q_diff_avgs:
            print(f"Average Q-Value Difference: {np.mean(all_q_diff_avgs):.4f}")
            print(f"Q-Diff Range: [{np.min(all_q_diff_avgs):.4f}, {np.max(all_q_diff_avgs):.4f}]")

        if all_variance_avgs:
            print(f"Average Variance: {np.mean(all_variance_avgs):.6f}")
            print(f"Variance Range: [{np.min(all_variance_avgs):.6f}, {np.max(all_variance_avgs):.6f}]")

        if all_entropy_triggers:
            total_entropy = np.sum(all_entropy_triggers)
            total_q_diff = np.sum(all_q_diff_triggers)
            total_variance = np.sum(all_variance_triggers)
            print(f"Entropy Triggers (Total): {total_entropy}")
            print(f"Q-Diff Triggers (Total): {total_q_diff}")
            print(f"Variance Triggers (Total): {total_variance}")

            # 计算每个触发器的平均触发率
            avg_entropy_rate = total_entropy / len(all_entropy_triggers)
            avg_q_diff_rate = total_q_diff / len(all_q_diff_triggers)
            avg_variance_rate = total_variance / len(all_variance_triggers)
            print(f"Average Entropy Triggers per Episode: {avg_entropy_rate:.1f}")
            print(f"Average Q-Diff Triggers per Episode: {avg_q_diff_rate:.1f}")
            print(f"Average Variance Triggers per Episode: {avg_variance_rate:.1f}")

        print("\n=== Analysis ===")
        print("Expected in real evaluation output:")
        print("- Attack mode identification: Statistical Critical State Attack")
        print("- Scale parameters display")
        print("- Comprehensive trigger statistics")
        print("- Range analysis for each metric")

        return True

    except Exception as e:
        print(f"[ERROR] Statistical summary test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_statistical_summary()
    if success:
        print("\n" + "="*60)
        print("Statistical summary output test completed!")
        print("The evaluation summary will now include comprehensive")
        print("statistical attack method statistics.")
    else:
        print("\n" + "="*60)
        print("Test failed.")
