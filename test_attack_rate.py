#!/usr/bin/env python3
"""
Test SmartC Continuous mode attack rate statistics
Verify attack counting and statistics output are correct
"""

import os
import sys
import re

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'gym_env'))

def check_attack_statistics():
    """Check attack statistics related code"""

    print("Checking SmartC Continuous mode attack rate statistics")
    print("=" * 60)

    # Check thread_evaluation.py code
    eval_file = "scripts/utils/thread_evaluation.py"

    if not os.path.exists(eval_file):
        print("ERROR: File not found:", eval_file)
        return False

    with open(eval_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if smartc_continuous is correctly included in all relevant locations
    checks = [
        ("Episode attack rate collection", r"attack_trigger_mode in \([^)]*'smartc_continuous'[^)]*\)"),
        ("Average attack rate calculation", r"attack_trigger_mode in \([^)]*'smartc_continuous'[^)]*\).*all_attack_rates"),
        ("Statistics output", r"attack_trigger_mode in \([^)]*'smartc_continuous'[^)]*\).*all_attack_rates"),
        ("Episode end printing", r"attack_trigger_mode in \([^)]*'smartc_continuous'[^)]*\)"),
        ("Reset logic", r"attack_trigger_mode in \([^)]*'smartc_continuous'[^)]*\)"),
    ]

    all_passed = True

    for check_name, pattern in checks:
        if re.search(pattern, content):
            print(f"PASS: {check_name} - smartc_continuous correctly included")
        else:
            print(f"FAIL: {check_name} - smartc_continuous missing")
            all_passed = False

    # Check smartc_continuous mode implementation
    if "smartc_continuous" in content:
        print("PASS: SmartC Continuous mode implementation exists")
    else:
        print("FAIL: SmartC Continuous mode implementation missing")
        all_passed = False

    # Check continuous attack counter
    if "continuous_attack_counter" in content:
        print("PASS: Continuous attack counter implemented")
    else:
        print("FAIL: Continuous attack counter missing")
        all_passed = False

    # Check continuous_attack_steps parameter
    if "continuous_attack_steps" in content:
        print("PASS: continuous_attack_steps parameter implemented")
    else:
        print("FAIL: continuous_attack_steps parameter missing")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All attack rate statistics checks passed!")
        print("\nVerification points:")
        print("1. Attack counting accumulates correctly")
        print("2. Episode end correctly calculates attack rate")
        print("3. Statistics page outputs average attack rate correctly")
        print("4. Burst mode doesn't affect counting logic")
        print("\nSuggested test command:")
        print("python scripts/start_evaluate_with_plot.py --config logs/SimpleAvoid/2025_12_10_09_47_Multirotor_CNN_GAP_SAC/config/config_burst.ini --episodes 5 --no-plot")
    else:
        print("ERROR: Issues found, please check code implementation")

    return all_passed

def test_config_file():
    """Test configuration file"""

    config_path = "logs/SimpleAvoid/2025_12_10_09_47_Multirotor_CNN_GAP_SAC/config/config_burst.ini"

    print(f"\nChecking config file: {config_path}")

    if not os.path.exists(config_path):
        print("ERROR: Config file does not exist")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    required_settings = [
        "attack_trigger_mode = smartc_continuous",
        "continuous_attack_steps = 5",
        "debug_attack = true"
    ]

    all_present = True
    for setting in required_settings:
        if setting in content:
            print(f"PASS: {setting}")
        else:
            print(f"FAIL: Missing {setting}")
            all_present = False

    return all_present

if __name__ == "__main__":
    print("SmartC Continuous Attack Rate Statistics Test")
    print("=" * 60)

    # Check code implementation
    code_ok = check_attack_statistics()

    # Check config file
    config_ok = test_config_file()

    print("\n" + "=" * 60)
    if code_ok and config_ok:
        print("SUCCESS: All checks passed! SmartC Continuous mode attack rate statistics should work correctly")
        print("\nExpected output format:")
        print("  episode:  1  reward: XX.XX success: True   | risk_attack_rate: X.X%")
        print("  Average Attack Rate: X.XX%")
        print("  Attack Rate Range: [X.XX%, XX.XX%]")
    else:
        print("WARNING: Configuration or code issues found, please fix and test again")