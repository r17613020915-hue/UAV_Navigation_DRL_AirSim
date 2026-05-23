import sys
import argparse

from PyQt5 import QtWidgets

# from evaluate_td3 import evaluate
from utils.thread_train import TrainingThread
# from utils.thread_train_fixedwing import TrainingThread
from utils.ui_train import TrainingUi
from configparser import ConfigParser


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training navigation model using TD3")
    parser.add_argument('-config', required=True,
                        help='config file name, such as config0925.ini', default='config_default.ini')
    parser.add_argument('-objective', required=True, help='training objective')

    return parser


def main():
    # select your config file here
    # config_file = 'configs/config_SimpleAvoid_SimpleMultirotor.ini'
    #config_file = 'configs/config_Trees_SimpleMultirotor.ini'
    #config_file = 'configs/config_NH_center_SimpleMultirotor_3D.ini'
    config_file = 'configs/config_Maze_SimpleMultirotor_2D.ini'

    # 1. Create the qt thread
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()

    # 2. Start training thread
    training_thread = TrainingThread(config_file)

    training_thread.env.action_signal.connect(gui.action_cb)
    training_thread.env.state_signal.connect(gui.state_cb)
    training_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)

    # 打印训练信息到终端
    print("=" * 60)
    print("训练已启动！")
    print(f"配置文件: {config_file}")
    print(f"模型保存路径: logs/{cfg.get('options', 'env_name')}/")
    print("请查看 GUI 窗口查看实时训练曲线")
    print("=" * 60)
    sys.stdout.flush()

    training_thread.start()

    # 保持终端活跃
    print("\n[提示] 训练进行中... 你可以:")
    print("  - 查看 GUI 窗口中的实时训练曲线")
    print("  - 按 Ctrl+C 停止训练")
    print("-" * 60)
    sys.stdout.flush()

    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
