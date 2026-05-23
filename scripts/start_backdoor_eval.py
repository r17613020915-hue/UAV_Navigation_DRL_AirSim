"""
后门攻击评估GUI

可视化评估后门攻击效果

使用方法:
    python scripts/start_backdoor_eval.py --model-path <path> --config <config>
"""

import sys
import argparse
import os
import numpy as np
from configparser import ConfigParser

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPalette, QColor

from utils.ui_train import TrainingUi
from utils.thread_backdoor_eval import BackdoorEvalThread
from backdoor_attacks import TARGET_ACTIONS


class BackdoorEvalGui(TrainingUi):
    """后门攻击评估GUI"""
    
    def __init__(self, config: str = None, model_path: str = None,
                 num_episodes: int = 50, target_action: int = None,
                 trigger_size: int = None, trigger_type: str = None,
                 attack_on_trigger: bool = None):
        super().__init__(config)

        # 从配置文件读取后门参数
        cfg = ConfigParser()
        if config and os.path.exists(config):
            cfg.read(config)

        cfg_target = cfg.getint('backdoor', 'target_action', fallback=0) if cfg.has_section('backdoor') else 0
        cfg_trigger_size = cfg.getint('backdoor', 'trigger_size', fallback=6) if cfg.has_section('backdoor') else 6
        cfg_trigger_type = cfg.get('backdoor', 'trigger_type', fallback='checkerboard') if cfg.has_section('backdoor') else 'checkerboard'
        cfg_trigger_position = cfg.get('backdoor', 'trigger_position', fallback='top_left') if cfg.has_section('backdoor') else 'top_left'
        cfg_attack_on_trigger = cfg.getboolean('backdoor', 'attack_on_trigger', fallback=False) if cfg.has_section('backdoor') else False

        self.model_path = model_path
        self.num_episodes = num_episodes
        self.target_action = cfg_target if target_action is None else target_action
        self.trigger_size = cfg_trigger_size if trigger_size is None else trigger_size
        self.trigger_type = cfg_trigger_type if trigger_type is None else trigger_type
        self.trigger_position = cfg_trigger_position
        self.attack_on_trigger = cfg_attack_on_trigger if attack_on_trigger is None else attack_on_trigger

        self.eval_thread = None

        # 添加后门攻击评估控件
        self.setup_backdoor_controls()

        # 初始化显示
        self.update_backdoor_display()
        
    def setup_backdoor_controls(self):
        """设置后门攻击评估控件"""
        # 创建后门评估组
        self.backdoor_group = QtWidgets.QGroupBox("后门攻击评估")
        backdoor_layout = QtWidgets.QVBoxLayout()
        
        # 目标动作选择
        target_layout = QtWidgets.QHBoxLayout()
        target_layout.addWidget(QtWidgets.QLabel("目标动作:"))
        
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("前进 (Forward)", 0)
        self.target_combo.addItem("左转 (Left)", 1)
        self.target_combo.addItem("右转 (Right)", 2)
        self.target_combo.setCurrentIndex(self.target_action)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        target_layout.addWidget(self.target_combo)
        target_layout.addStretch()
        
        backdoor_layout.addLayout(target_layout)
        
        # 评估参数
        params_layout = QtWidgets.QHBoxLayout()
        
        params_layout.addWidget(QtWidgets.QLabel("评估回合:"))
        self.episodes_spin = QtWidgets.QSpinBox()
        self.episodes_spin.setRange(1, 1000)
        self.episodes_spin.setValue(self.num_episodes)
        params_layout.addWidget(self.episodes_spin)
        
        params_layout.addWidget(QtWidgets.QLabel("触发器大小:"))
        self.trigger_size_spin = QtWidgets.QSpinBox()
        self.trigger_size_spin.setRange(2, 40)
        self.trigger_size_spin.setValue(self.trigger_size)
        params_layout.addWidget(self.trigger_size_spin)

        params_layout.addWidget(QtWidgets.QLabel("触发器类型:"))
        self.trigger_type_combo = QtWidgets.QComboBox()
        self.trigger_type_combo.addItems(["checkerboard", "solid", "border"])
        self.trigger_type_combo.setCurrentText(self.trigger_type)
        params_layout.addWidget(self.trigger_type_combo)

        self.force_target_checkbox = QtWidgets.QCheckBox("强制执行目标动作")
        self.force_target_checkbox.setChecked(self.attack_on_trigger)
        params_layout.addWidget(self.force_target_checkbox)
        
        params_layout.addStretch()
        
        backdoor_layout.addLayout(params_layout)
        
        # 按钮
        button_layout = QtWidgets.QHBoxLayout()
        
        self.start_backdoor_btn = QtWidgets.QPushButton("开始后门评估")
        self.start_backdoor_btn.clicked.connect(self.start_backdoor_eval)
        button_layout.addWidget(self.start_backdoor_btn)
        
        self.stop_backdoor_btn = QtWidgets.QPushButton("停止")
        self.stop_backdoor_btn.clicked.connect(self.stop_backdoor_eval)
        self.stop_backdoor_btn.setEnabled(False)
        button_layout.addWidget(self.stop_backdoor_btn)
        
        backdoor_layout.addLayout(button_layout)
        
        # 统计显示
        stats_layout = QtWidgets.QGridLayout()
        
        # ASR
        stats_layout.addWidget(QtWidgets.QLabel("攻击成功率 (ASR):"), 0, 0)
        self.asr_label = QtWidgets.QLabel("0.0%")
        self.asr_label.setStyleSheet("font-weight: bold; color: gray;")
        stats_layout.addWidget(self.asr_label, 0, 1)
        
        # 触发次数
        stats_layout.addWidget(QtWidgets.QLabel("触发次数:"), 0, 2)
        self.trigger_label = QtWidgets.QLabel("0")
        stats_layout.addWidget(self.trigger_label, 0, 3)
        
        # 成功率
        stats_layout.addWidget(QtWidgets.QLabel("任务成功率:"), 1, 0)
        self.success_label = QtWidgets.QLabel("0.0%")
        stats_layout.addWidget(self.success_label, 1, 1)
        
        # 平均回报
        stats_layout.addWidget(QtWidgets.QLabel("平均回报:"), 1, 2)
        self.reward_label = QtWidgets.QLabel("0.0")
        stats_layout.addWidget(self.reward_label, 1, 3)
        
        backdoor_layout.addLayout(stats_layout)
        
        # 进度条
        self.backdoor_progress = QtWidgets.QProgressBar()
        backdoor_layout.addWidget(self.backdoor_progress)
        
        # 判定
        self.verdict_label = QtWidgets.QLabel("")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet("padding: 5px; font-weight: bold;")
        backdoor_layout.addWidget(self.verdict_label)
        
        self.backdoor_group.setLayout(backdoor_layout)
        
        # 添加到主布局
        self.layout().addWidget(self.backdoor_group)
        
    def on_target_changed(self, index):
        """目标动作改变"""
        self.target_action = self.target_combo.itemData(index)
        self.update_backdoor_display()
        
    def update_backdoor_display(self):
        """更新显示"""
        target_name = TARGET_ACTIONS.get(self.target_action, {}).get('name', 'unknown')
        self.setWindowTitle(f"后门攻击评估 - 目标: {target_name}")
        
    def start_backdoor_eval(self):
        """开始后门评估"""
        if self.eval_thread is not None and self.eval_thread.isRunning():
            print("[GUI] 评估已在运行中")
            return
        
        self.num_episodes = self.episodes_spin.value()
        self.target_action = self.target_combo.itemData(self.target_combo.currentIndex())
        self.trigger_size = self.trigger_size_spin.value()
        self.trigger_type = self.trigger_type_combo.currentText()
        self.attack_on_trigger = self.force_target_checkbox.isChecked()
        
        print(f"\n[GUI] 开始后门评估")
        print(f"  模型: {self.model_path}")
        print(f"  目标动作: {self.target_action} ({TARGET_ACTIONS.get(self.target_action, {}).get('name', 'unknown')})")
        print(f"  评估回合: {self.num_episodes}")
        print(f"  触发器: {self.trigger_type}, {self.trigger_size}x{self.trigger_size}, {self.trigger_position}")
        print(f"  强制执行目标动作: {self.attack_on_trigger}")
        
        # 创建评估线程
        self.eval_thread = BackdoorEvalThread(
            model_path=self.model_path,
            config_file=self.config_file,
            num_episodes=self.num_episodes,
            target_action=self.target_action,
            trigger_size=self.trigger_size,
            trigger_type=self.trigger_type,
            trigger_position=self.trigger_position,
            device='cuda',
            deterministic=True,
            attack_on_trigger=self.attack_on_trigger
        )
        
        # 连接信号
        self.eval_thread.eval_started.connect(self.on_eval_started)
        self.eval_thread.eval_progress.connect(self.on_eval_progress)
        self.eval_thread.episode_finished.connect(self.on_episode_finished)
        self.eval_thread.eval_finished.connect(self.on_eval_finished)

        # 连接可视化信号
        self.eval_thread.pose_updated.connect(self.traj_plot_cb)
        
        # 更新UI状态
        self.start_backdoor_btn.setEnabled(False)
        self.stop_backdoor_btn.setEnabled(True)
        
        # 开始评估
        self.eval_thread.start()
        
    def stop_backdoor_eval(self):
        """停止后门评估"""
        if self.eval_thread is not None:
            print("[GUI] 停止后门评估...")
            self.eval_thread.stop()
            
    def on_eval_started(self):
        """评估开始"""
        print("[GUI] 评估已开始")
        self.backdoor_progress.setValue(0)
        
    def on_eval_progress(self, current: int, total: int, asr: float):
        """评估进度更新"""
        self.backdoor_progress.setValue(int(current / total * 100))
        self.asr_label.setText(f"{asr*100:.1f}%")
        
        # 根据ASR更新颜色
        if asr > 0.7:
            self.asr_label.setStyleSheet("font-weight: bold; color: red;")
        elif asr > 0.4:
            self.asr_label.setStyleSheet("font-weight: bold; color: orange;")
        elif asr > 0.2:
            self.asr_label.setStyleSheet("font-weight: bold; color: yellow; background: black;")
        else:
            self.asr_label.setStyleSheet("font-weight: bold; color: green;")
        
    def on_episode_finished(self, ep: int, reward: float, success: bool, length: int):
        """Episode完成"""
        self.success_label.setText(f"{'是' if success else '否'} ({int(success)*100:.0f}%)")
        self.reward_label.setText(f"{reward:.1f}")
        
    def on_backdoor_triggered(self, step: int, triggered: bool, target_taken: bool):
        """后门触发"""
        if triggered:
            self.trigger_label.setText(f"触发 (目标: {'是' if target_taken else '否'})")
        
    def on_eval_finished(self, results: dict):
        """评估完成"""
        print("[GUI] 评估已完成")

        self.start_backdoor_btn.setEnabled(True)
        self.stop_backdoor_btn.setEnabled(False)

        if 'error' in results:
            self.verdict_label.setText(f"错误: {results['error']}")
            self.verdict_label.setStyleSheet("color: red; padding: 5px; font-weight: bold;")
            return

        clean = results.get('clean', {})
        attack = results.get('attack', {})
        delta = results.get('delta', {})

        self.asr_label.setText(f"{attack.get('global_asr', 0)*100:.1f}%")
        self.trigger_label.setText(str(attack.get('total_triggers', 0)))
        self.success_label.setText(f"{clean.get('success_rate', 0)*100:.1f}%")
        self.reward_label.setText(f"{clean.get('mean_reward', 0):.2f}")

        verdict = results.get('verdict', '评估完成')
        self.verdict_label.setText(verdict)

        if '严重' in verdict:
            color = 'red'
        elif '警告' in verdict:
            color = 'orange'
        elif '注意' in verdict:
            color = 'yellow'
        elif '可疑' in verdict:
            color = 'blue'
        else:
            color = 'green'

        self.verdict_label.setStyleSheet(f"color: {color}; padding: 5px; font-weight: bold;")

        print(f"\n{'='*60}")
        print("后门攻击评估结果")
        print(f"{'='*60}")
        print(f"目标动作: {results['target_action']} ({results['target_action_name']})")

        print(f"\n[Pass 1 - 干净基线]")
        print(f"  成功率:   {clean.get('success_rate', 0)*100:.1f}%")
        print(f"  平均回报: {clean.get('mean_reward', 0):.2f} ± {clean.get('std_reward', 0):.2f}")
        print(f"  碰撞率:   {clean.get('crash_rate', 0)*100:.1f}%")

        print(f"\n[Pass 1 - 干净基线]")
        print(f"  成功率:   {clean.get('success_rate', 0)*100:.1f}%")
        print(f"  平均回报: {clean.get('mean_reward', 0):.2f}")
        print(f"  碰撞率:   {clean.get('crash_rate', 0)*100:.1f}%")

        print(f"\n[Pass 2 - 攻击模式]")
        print(f"  成功率:   {attack.get('success_rate', 0)*100:.1f}%")
        print(f"  平均回报: {attack.get('mean_reward', 0):.2f}")
        print(f"  碰撞率:   {attack.get('crash_rate', 0)*100:.1f}%")
        print(f"  ASR:      {attack.get('global_asr', 0)*100:.1f}%")
        print(f"  触发次数: {attack.get('total_triggers', 0)}")
        print(f"  触发后碰撞率: {attack.get('collision_rate_on_trigger', 0)*100:.1f}%")

        print(f"\n[性能对比] (干净基线 vs 攻击模式)")
        print(f"  回报变化:    {delta.get('reward', 0):+.2f}")
        print(f"  成功率变化:  {delta.get('success_rate', 0):+.1f}%")
        print(f"  碰撞率变化:  {delta.get('crash_rate', 0):+.1f}%")

        print(f"\n判定: {verdict}")
        
    def closeEvent(self, event):
        """关闭窗口"""
        if self.eval_thread is not None and self.eval_thread.isRunning():
            print("[GUI] 正在停止评估...")
            self.eval_thread.stop()
            self.eval_thread.wait()
        event.accept()


def get_parser():
    parser = argparse.ArgumentParser(description="后门攻击评估GUI")
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--config', type=str, 
                       default='configs/config_Maze_SimpleMultirotor_2D.ini',
                       help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=50,
                       help='评估回合数')
    parser.add_argument('--target', type=int, default=1, choices=[0, 1, 2],
                       help='目标动作: 0=forward, 1=left, 2=right')
    parser.add_argument('--trigger-size', type=int, default=6,
                       help='触发器大小')
    parser.add_argument('--trigger-type', type=str, default=None,
                       choices=['checkerboard', 'solid', 'border'],
                       help='触发器类型')
    parser.add_argument('--attack-on-trigger', action='store_true',
                       help='触发后强制执行目标动作')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    
    # 设置样式
    app.setStyle('Fusion')
    
    # 创建GUI
    gui = BackdoorEvalGui(
        config=args.config,
        model_path=args.model_path,
        num_episodes=args.episodes,
        target_action=args.target,
        trigger_size=args.trigger_size,
        trigger_type=args.trigger_type,
        attack_on_trigger=args.attack_on_trigger if args.attack_on_trigger else None
    )
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
