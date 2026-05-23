#!/usr/bin/env python3
"""
分析当前阈值设置和攻击率的关系
"""

import numpy as np

def analyze_current_thresholds():
    """分析当前阈值设置"""
    print("=== Threshold Analysis ===")

    # 当前配置
    threshold = 0.7  # attack_confidence_threshold
    entropy_scale = 8.0
    q_diff_scale = 3.0  # 新设置
    variance_scale = 2.0

    # 计算实际阈值
    entropy_threshold = threshold * entropy_scale
    q_diff_threshold = threshold * q_diff_scale
    variance_threshold = threshold * variance_scale

    print("Current Configuration:")
    print(f"  attack_confidence_threshold: {threshold}")
    print(f"  entropy_scale: {entropy_scale}")
    print(f"  q_diff_scale: {q_diff_scale}")
    print(f"  variance_scale: {variance_scale}")
    print()
    print("Calculated Thresholds:")
    print(f"  entropy_threshold: {entropy_threshold:.2f}")
    print(f"  q_diff_threshold: {q_diff_threshold:.2f}")
    print(f"  variance_threshold: {variance_threshold:.2f}")
    print()

    # 实际观测值（从最新评估结果）
    actual_values = [
        {"episode": 1, "entropy_avg": 5.046, "q_diff_avg": 1.410, "var_avg": 0.011, "triggers": "E:121 Q:377 V:0"},
        {"episode": 2, "entropy_avg": 4.882, "q_diff_avg": 1.442, "var_avg": 0.010, "triggers": "E:74 Q:389 V:0"},
        {"episode": 3, "entropy_avg": 6.262, "q_diff_avg": 1.433, "var_avg": 0.005, "triggers": "E:223 Q:300 V:0"},
        {"episode": 4, "entropy_avg": 6.362, "q_diff_avg": 1.587, "var_avg": 0.005, "triggers": "E:232 Q:324 V:0"},
    ]

    print("Actual Values vs Thresholds:")
    print("Episode | Entropy | Q-Diff | Variance | Triggers | Analysis")
    print("-" * 70)

    for ep in actual_values:
        entropy_trigger = ep["entropy_avg"] > entropy_threshold
        q_diff_trigger = ep["q_diff_avg"] > q_diff_threshold
        var_trigger = ep["var_avg"] > variance_threshold

        analysis = []
        if entropy_trigger:
            analysis.append("Entropy")
        if q_diff_trigger:
            analysis.append("Q-Diff")
        if var_trigger:
            analysis.append("Variance")

        trigger_count = len(analysis)
        analysis_str = ", ".join(analysis) if analysis else "None"

        print("5d")

    print()
    print("Expected Results with New Thresholds:")
    print("- Q-Diff triggers should be significantly reduced (from ~350 to ~0)")
    print("- Entropy triggers may still occur when entropy > 5.6")
    print("- Variance triggers should be rare (var_avg ~0.01 < 1.4)")
    print("- Overall attack rate should drop from ~95% to ~10-20%")

if __name__ == "__main__":
    analyze_current_thresholds()
