#!/usr/bin/env python3
"""
演示随机攻击触发模式的统计输出
"""

def demo_attack_statistics():
    """演示攻击统计输出"""
    print("=" * 80)
    print("ATTACK STATISTICS")
    print("=" * 80)

    # 模拟不同模式的输出
    modes = ['confidence', 'risk', 'smart', 'step_interval', 'random']

    for mode in modes:
        print(f"\nMode: {mode}")
        print("-" * 40)

        if mode == 'always':
            print(' Always Attack')
        elif mode == 'step_interval':
            print('Step Interval Attack')
            expected_rate = (4 / (1 + 4) * 100)  # n=1, m=4
            print('.1f')
        elif mode == 'confidence':
            print('Confidence-based Attack')
        elif mode == 'risk':
            print('Risk-based Attack')
        elif mode == 'smart':
            print('Smart Attack (Risk + Confidence)')
        elif mode == 'random':
            print('Random Attack (Probability: 0.30)')

        # 模拟统计信息显示
        if mode in ('confidence', 'risk', 'smart', 'step_interval', 'random'):
            print('Average Attack Rate: 25.3%')
            print('Attack Rate Range: [15.2%, 35.6%]')

            # confidence模式的额外统计
            if mode == 'confidence':
                print('Average Max Distance (Method 1): 0.2345')
                print('Max Distance Range: [0.1234, 0.3456]')
                print('Average Policy Confidence (Method 2): 0.6789')
                print('Policy Confidence Range: [0.5678, 0.7890]')
                print('Method 1 Triggers (Total): 1234')
                print('Method 2 Triggers (Total): 567')
                print('Method 2 Failures (Total): 89')

        # 显示攻击参数
        print('Attack Type: fgsm')
        print('Attack Epsilon: 0.07')
        if mode == 'confidence':
            print('Confidence Threshold: 0.5')
        elif mode == 'risk':
            print('Risk Threshold: 0.6, Risk Margin: 3.0')
        elif mode == 'step_interval':
            print('Step Interval Parameters: n=1, m=4')
        elif mode == 'random':
            print('Random Attack Probability: 0.30')

if __name__ == "__main__":
    print("随机攻击触发模式统计输出演示")
    print("=" * 50)
    demo_attack_statistics()
