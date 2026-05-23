# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import time
import tyro

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.utils import Args, make_env, load_dict_from_yaml
from utils.models import Agent, QNetwork
from adversary.Adversary import ImagePoison, Discrete, SingleValuePoison
from adversary.OuterLoop import SleeperNets, Q_Incept
from adversary.InnerLoop import BadRL, TrojDRL
from adversary import patterns

def initialize_attacks(args, envs):
    device = args.device
    if args.sn_outer:
        run_name = f"SN_{args.p_rate}_{args.rew_p}_{args.alpha}"
    elif args.inception:
        run_name = f"QIn_{args.p_rate}"
    elif args.trojdrl:
        run_name = f"TrojDRL_{args.p_rate}_{args.rew_p}"
    elif args.badrl:
        run_name = f"BadRL_{args.p_rate}_{args.rew_p}"
    else:
        run_name = f"Benign"
    run_name += f"_{args.exp_name}"
    attacker = None
    poison = None
    poison_batch = None

    # --- Set up Attacks --- #
    if args.sn_outer or args.inception or args.trojdrl or args.badrl:
        if args.safety or args.cage or args.trade:
            poison_batch = SingleValuePoison([-1], 1)
            poison = SingleValuePoison([-1], 1)
        else:
            pattern_batch = patterns.Stacked_Img_Pattern((1,4, 84, 84), (8,8)).to(device)
            poison_batch = ImagePoison(pattern_batch, 0, 255)

            pattern = patterns.Single_Stacked_Img_Pattern((4, 84, 84), (8,8))
            pattern = pattern.to(device)# if args.sn_outer or args.inception else pattern.numpy()
            poison = ImagePoison(pattern, 0, 255)#, numpy = args.trojdrl or args.badrl)
        if args.inception:
            Q = lambda : QNetwork(envs, not (args.safety or args.trade or args.cage), args.safety, args.trade, args.cage)
            attacker = Q_Incept(poison, Q, args, envs)
        elif args.sn_outer:
            attacker = SleeperNets(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.gamma, p_rate = args.p_rate, alpha = args.alpha, simple = args.simple_select, clip = args.clip)
        elif args.trojdrl:
            attacker = TrojDRL(envs.single_action_space.n, poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.total_timesteps, args.total_timesteps*args.p_rate, args.strong, args.clip)
        elif args.badrl:
            q_net_adv = QNetwork(envs, not (args.safety or args.trade or args.cage), args.safety, args.trade, args.cage)
            q_net_adv.load_state_dict(torch.load(f"dqn_models/{args.env_id}__dqn/dqn.cleanrl_model", map_location = "cpu"))
            q_net_adv.to(device)
            attacker = BadRL(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.p_rate, q_net_adv, args.strong, args.clip)
        

    return run_name, attacker, poison, poison_batch


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    os.makedirs("runs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("wandb", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if len(args.attack_config) > 0:
        if len(args.attack_name) == 0: 
            args.attack_name = "benign"
        attack_config = load_dict_from_yaml(args.attack_config)[args.attack_name]
        args.__dict__.update(attack_config)
    if len(args.env_config) > 0:
        env_config = load_dict_from_yaml(args.env_config)[args.env_id]
        args.__dict__.update(env_config)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.track:
        import wandb

    args.unique = int(time.time()) #unique id to identify runs if 