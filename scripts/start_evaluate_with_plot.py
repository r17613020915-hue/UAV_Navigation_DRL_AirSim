import sys
import argparse
from PyQt5 import QtWidgets

from utils.thread_evaluation import EvaluateThread
from utils.ui_train import TrainingUi
from configparser import ConfigParser
from stable_baselines3 import TD3, PPO


def get_parser():
    parser = argparse.ArgumentParser(
        description="trained model evaluation with plot")
    parser.add_argument('-model_path', required=True, help='model path to be evaluated, \
                                            just copy the relative path of the log')
    parser.add_argument('-eval_eps', required=True, type=int,
                        help='evaluation episode number')

    return parser


def main():

    # set evaluation model path
    #eval_path = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2022_09_06_15_05_SimpleMultirotor_No_CNN_SAC'
    eval_path = r'D:\aRLAA\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2026_05_22_13_31_Multirotor_No_CNN_SAC_attack-sleepernets'

    # select config file and model name
    config_file = r'D:\aRLAA\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2026_05_22_13_31_Multirotor_No_CNN_SAC_attack-sleepernets\config\config.ini'
    # config_file = r"D:\OneDrive - mail.nwpu.edu.cn\Github\PhD-thesis-plot\CH3\3_1_simple_training_and_anaylysis\3_1_4_training_analysis_multi\data\2022_08_30_10_43_SimpleMultirotor_No_CNN_SAC\config\config.ini"
    model_file = r'D:\aRLAA\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2026_05_22_13_31_Multirotor_No_CNN_SAC_attack-sleepernets\models\model_sb3.zip'
    # config_file = r"C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\configs\config_new.ini"
    # model_file = eval_path + '/models/model_200000.zip'
    total_eval_episodes = 100

    # 1. Create the qt thread (is MainThread in fact)
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=config_file)
    gui.show()

    # 2. Start training thread
    evaluate_thread = EvaluateThread(
        eval_path, config_file, model_file, total_eval_episodes)
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option('options', 'perception'):
        if cfg.get('options', 'perception') == 'lgmd':
            evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()

    # program will not terminate until you closed the GUI
    sys.exit(app.exec_())
    print('Exiting program')


if __name__ == "__main__":
    main()


#  新增攻击相关配置  在模型的config文件里加
# enable_attack = False  ; 是否启用攻击，true/false
# attack_type = fgsm    ; 攻击类型: fgsm, pgd, random, none, deepfool, cw, bim, mim，crash（增强版fgsm，离得近加大力度）
# attack_epsilon = 0.08  ; 扰动强度 epsilon
# attack_trigger_mode = risk     ; risk / confidence / always /smart /step_interval/random/statistical/smart_q
#设 risk_distance = crash_distance + risk_margin
#风险值 risk = (risk_distance - min_dist) / risk_distance（裁剪到 0~1）  min_dist 越小，risk 越接近 1（越危险）
# attack_confidence_threshold = 0.7  ; 动作置信度阈值 (0-1)，超过此阈值才触发攻击，默认0.7