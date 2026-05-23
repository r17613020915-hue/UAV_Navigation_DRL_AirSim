#!/usr/bin/env python3
"""
测试SmartC Continuous (Burst) 攻击模式
验证连续攻击机制是否正常工作
"""

import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'gym_env'))

def test_burst_attack_config():
    """测试连续攻击配置"""

    config_path = "logs/SimpleAvoid/2025_12_10_09_47_Multirotor_CNN_GAP_SAC/config/config_burst.ini"

    if not os.path.exists(config_path):
        print("❌ 配置文件不存在:", config_path)
        print("请先运行代码创建配置文件")
        return

    print("🎯 SmartC Continuous (Burst) 攻击模式测试")
    print("=" * 60)

    print("📋 配置说明：")
    print(f"  - 模式: smartc_continuous")
    print(f"  - 连续攻击步数: 5步")
    print(f"  - 攻击强度: 0.08")
    print(f"  - 风险阈值: 0.2")

    print("\n🎯 预期行为：")
    print("  ✅ 触发时连续攻击5步，无需重新判断")
    print("  ✅ 触发频率较低，但破坏力放大")
    print("  ✅ Debug输出显示连击开始和结束")

    print("\n🚀 测试命令：")
    print(f"python scripts/start_evaluate_with_plot.py --config {config_path} --episodes 10 --no-plot")

    print("\n📊 预期输出示例：")
    print("  [Attack Burst] SmartC Continuous mode: attack triggered at step 45, starting 5 continuous attacks")
    print("  [Attack Burst] Continuous attack ended at step 50")

    print("\n🔍 验证要点：")
    print("  1. 攻击是否按连击模式执行")
    print("  2. 连击结束后是否正确停止")
    print("  3. 触发频率是否合理")
    print("  4. 整体性能下降是否显著")

if __name__ == "__main__":
    test_burst_attack_config()























