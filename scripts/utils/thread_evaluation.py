# 修改后的 thread_evaluation.py 文件内容

from PyQt5 import QtCore
from configparser import ConfigParser
from stable_baselines3 import TD3, SAC, PPO
import numpy as np
import gym_env
import gym
import gym.spaces
import math
import os
import sys
import cv2
from tqdm import tqdm

import torch as th
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(
    r"C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\scripts")


def rule_based_policy(obs):
    '''
    custom linear policy
    used for LGMD compare
    '''
    action = 0
    # 将obs从1~-1转换成0~1
    obs = np.squeeze(obs, axis=0)

    for i in range(5):
        obs[i] = obs[i] / 2 + 0.5

    # obs_weight_depth = np.array([1.0, 3.0, 5.0, -3.0, -1.0, 3.0])
    obs_weight = np.array([1.0, 3.0, 3.0, -3.0, -1.0, 3.0])
    action = obs * obs_weight

    action_sum = np.sum(action)

    if action_sum > math.radians(40):
        action_sum = math.radians(40)
    elif action_sum < -math.radians(40):
        action_sum = -math.radians(40)

    return np.array([action_sum])


# ==================== 攻击 =====================
# 将 numpy obs 转为 torch tensor（batch 维度及类型处理）
def _to_torch(obs_np, device, perception_type):
    """
    obs_np: 原始 numpy obs
    perception_type: 'vector' 或 'depth'/'image' 等
    返回: torch tensor, 已送到 device, shape 以 policy 所期待为准
    """
    if perception_type == 'vector' or perception_type == 'lgmd':
        t = th.from_numpy(obs_np).float().to(device)
        # Ensure batch dim
        if t.ndim == 1:
            t = t[None, ...]
        return t
    else:
        # image-like obs: handle both 2-channel (depth+state) and 3-channel images
        obs_f = obs_np.astype('float32') / 255.0
        # For 2D images (H, W, C), reorder to C, H, W and add batch
        if obs_f.ndim == 3:
            # Standard image format: H, W, C -> C, H, W
            t = th.from_numpy(obs_f.transpose(2, 0, 1)[None, ...]).float().to(device)
        elif obs_f.ndim == 2:
            # Single channel image: H, W -> 1, H, W
            t = th.from_numpy(obs_f[None, None, ...]).float().to(device)
        else:
            raise ValueError(f"Unsupported observation shape: {obs_f.shape}")
        return t


def fgsm_attack_sac_td3(obs_np, model, epsilon, perception_type, device):
    """
    对基于 SB3 的 SAC/TD3 policy 做单步 FGSM（白盒）。
    返回与 obs_np 相同 shape/dtype 的对抗观测 obs_adv_np。
    """
    model.policy.eval()

    # 1) clean action using model.predict (keeps same preprocessing as SB3)
    with th.no_grad():
        try:
            action_clean_np, _ = model.predict(obs_np, deterministic=True)
        except Exception:
            # fallback: some custom models might require tuple input
            action_clean_np = model.predict(obs_np)
    action_clean_t = th.from_numpy(np.asarray(action_clean_np)).float().to(device)

    # 2) prepare obs tensor (requires_grad)
    obs_t = _to_torch(obs_np, device, perception_type)
    obs_t.requires_grad_(True)

    # 3) forward through actor to get actions as tensor
    # Try common attribute names. adapt if your model uses different names.
    action_adv_t = None
    # SB3 typical: actor is model.actor or model.policy.actor
    for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
        if candidate is None:
            continue
        try:
            # some actor signature expects flat inputs; others expect dicts - we try simple call
            action_adv_t = candidate(obs_t)
            break
        except Exception:
            # ignore and try next
            action_adv_t = None

    # fallback: try policy._predict or policy.forward producing actions
    if action_adv_t is None:
        try:
            # _predict often expects numpy, so call it and convert
            act_np_fallback = model.policy._predict(obs_t.detach().cpu().numpy())
            action_adv_t = th.from_numpy(np.asarray(act_np_fallback)).float().to(device)
        except Exception:
            # last resort: use model.predict but that does not give grad; raise error
            raise RuntimeError("无法从模型获得可导的 action 张量；请检查 model.policy/actor 接口。")

    # Make sure shapes align: action_adv_t shape -> (batch, action_dim)
    if action_adv_t.ndim == 1:
        action_adv_t = action_adv_t[None, ...]

    # 4) 计算攻击损失函数
    # ---------------- 原始逻辑：最大化动作差异 ----------------
    # if action_clean_t.ndim == 1:
    #     action_clean_t = action_clean_t[None, ...]
    # loss = F.mse_loss(action_adv_t, action_clean_t)

        # ---------------- 改进逻辑：最大化 critic Q 值 ----------------
    # 目的：让 agent 采取在 Q 网络看来"最好"的动作，但这可能不是我们想要的
    # 更好的策略：最大化 |action_adv - action_clean|，迫使模型做出极端动作
    try:
        # 确保形状一致
        if action_clean_t.ndim == 1:
            action_clean_t = action_clean_t[None, ...]

        # 主要策略：最大化动作差异，迫使模型做出与安全动作差异最大的动作
        loss = -F.mse_loss(action_adv_t, action_clean_t)  # 负号使梯度方向正确

        # 可选：也可以结合Q值作为辅助，但不是主要目标
        # if hasattr(model, 'critic'):
        #     try:
        #         q = model.critic(obs_t, action_adv_t)
        #         if isinstance(q, (tuple, list)):
        #             q_value = (q[0] + q[1]).mean()
        #         else:
        #             q_value = q.mean()
        #         # 结合动作差异和Q值：既要差异大，也要Q值高（更危险）
        #         loss = -F.mse_loss(action_adv_t, action_clean_t) + 0.1 * q_value
        #     except:
        #         pass  # Q值计算失败，继续使用动作差异

    except Exception as e:
        # 出错则回退到原损失，避免中断
        if action_clean_t.ndim == 1:
            action_clean_t = action_clean_t[None, ...]
        loss = -F.mse_loss(action_adv_t, action_clean_t)

    # zero grads and backprop
    model.policy.zero_grad()
    if obs_t.grad is not None:
        obs_t.grad.zero_()
    loss.backward(retain_graph=False)

    if obs_t.grad is None:
        raise RuntimeError("obs gradient is None after backward. actor may not depend on obs in a differentiable way.")

    # 5) FGSM step: adv = obs + eps * sign(grad)
    grad_sign = th.sign(obs_t.grad.data)
    adv_t = obs_t + epsilon * grad_sign

    # 6) convert adv_t back to numpy with original scale/dtype
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_t.detach().cpu().numpy()
        # remove batch dim if original had none
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
        adv_np = np.clip(adv_np, 0.0, 1.0)  # vector obs assumed in [0,1]
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]  # (C,H,W)
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))  # (H,W,C)
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np


# 新增 PGD 攻击函数（迭代版本的 FGSM）
def pgd_attack_sac_td3(obs_np, model, epsilon, perception_type, device, num_iter, alpha=None):
    """
    对基于 SB3 的 SAC/TD3 policy 做 PGD 攻击（白盒，多步迭代）。
    返回与 obs_np 相同 shape/dtype 的对抗观测 obs_adv_np。
    alpha 默认为 epsilon / num_iter。
    """
    if alpha is None:
        alpha = epsilon / num_iter

    model.policy.eval()

    # 1) clean action
    with th.no_grad():
        try:
            action_clean_np, _ = model.predict(obs_np, deterministic=True)
        except Exception:
            action_clean_np = model.predict(obs_np)
    action_clean_t = th.from_numpy(np.asarray(action_clean_np)).float().to(device)
    if action_clean_t.ndim == 1:
        action_clean_t = action_clean_t[None, ...]

    # 2) prepare obs tensor
    obs_t = _to_torch(obs_np, device, perception_type)
    adv_t = obs_t.clone().detach().requires_grad_(True)

    for _ in range(num_iter):
        # 3) forward to get action_adv_t
        action_adv_t = None
        for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
            if candidate is None:
                continue
            try:
                action_adv_t = candidate(adv_t)
                break
            except Exception:
                action_adv_t = None

        if action_adv_t is None:
            try:
                act_np_fallback = model.policy._predict(adv_t.detach().cpu().numpy())
                action_adv_t = th.from_numpy(np.asarray(act_np_fallback)).float().to(device)
            except Exception:
                raise RuntimeError("无法从模型获得可导的 action 张量。")

        if action_adv_t.ndim == 1:
            action_adv_t = action_adv_t[None, ...]

        # 4) loss
        loss = F.mse_loss(action_adv_t, action_clean_t)

        # backprop
        model.policy.zero_grad()
        if adv_t.grad is not None:
            adv_t.grad.zero_()
        loss.backward(retain_graph=False)

        if adv_t.grad is None:
            raise RuntimeError("adv gradient is None.")

        # 5) PGD step: adv = adv + alpha * sign(grad), then project
        grad_sign = th.sign(adv_t.grad.data)
        adv_t = adv_t + alpha * grad_sign
        # project to [obs - eps, obs + eps]
        adv_t = th.min(th.max(adv_t, obs_t - epsilon), obs_t + epsilon)
        # clip to valid range
        if perception_type == 'vector' or perception_type == 'lgmd':
            adv_t = th.clamp(adv_t, 0.0, 1.0)
        else:
            adv_t = th.clamp(adv_t, 0.0, 1.0)

        adv_t = adv_t.detach().requires_grad_(True)

    # 6) convert back
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_t.detach().cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]  # (C,H,W)
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))  # (H,W,C)
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np


# 新增随机噪声攻击函数（用于对比，非梯度基于）
def random_attack(obs_np, epsilon, perception_type):
    """
    添加均匀随机噪声作为对抗扰动（黑盒对比）。
    返回与 obs_np 相同 shape/dtype 的对抗观测 obs_adv_np。
    """
    if perception_type == 'vector' or perception_type == 'lgmd':
        noise = np.random.uniform(-epsilon, epsilon, obs_np.shape)
        adv_np = np.clip(obs_np + noise, 0.0, 1.0)
    else:
        # 对于图像，噪声在 [0,255] 空间
        noise = np.random.uniform(-epsilon * 255, epsilon * 255, obs_np.shape)
        adv_np = np.clip(obs_np + noise, 0, 255).astype(np.uint8)

    return adv_np


# 新增：专门设计来让模型撞墙的攻击函数
def crash_inducing_attack_sac_td3(obs_np, model, epsilon, perception_type, device, env=None):
    """
    专门设计来让模型撞墙的攻击函数。
    策略：最大化动作与安全动作的差异，让模型做出"危险"的动作。
    如果提供了env，可以在接近障碍物时使用更强的攻击。
    """
    model.policy.eval()

    # 1) 获取干净动作（安全动作）
    with th.no_grad():
        try:
            action_clean_np, _ = model.predict(obs_np, deterministic=True)
        except Exception:
            action_clean_np = model.predict(obs_np)
    action_clean_t = th.from_numpy(np.asarray(action_clean_np)).float().to(device)
    if action_clean_t.ndim == 1:
        action_clean_t = action_clean_t[None, ...]

    # 策略：如果接近障碍物，使用更强的攻击
    attack_epsilon = epsilon
    if env is not None:
        min_dist = float(getattr(env, "min_distance_to_obstacles", 1e9))
        crash_dist = float(getattr(env, "crash_distance", 2.0))
        if min_dist < crash_dist + 5.0:  # 接近障碍物时
            # 使用更大的epsilon倍数
            attack_epsilon = epsilon * 1.5

    # 2) 准备观测张量
    obs_t = _to_torch(obs_np, device, perception_type)
    obs_t.requires_grad_(True)

    # 3) 获取对抗动作
    action_adv_t = None
    for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
        if candidate is None:
            continue
        try:
            action_adv_t = candidate(obs_t)
            break
        except Exception:
            action_adv_t = None

    if action_adv_t is None:
        try:
            act_np_fallback = model.policy._predict(obs_t.detach().cpu().numpy())
            action_adv_t = th.from_numpy(np.asarray(act_np_fallback)).float().to(device)
        except Exception:
            raise RuntimeError("无法从模型获得可导的 action 张量。")

    if action_adv_t.ndim == 1:
        action_adv_t = action_adv_t[None, ...]

    # 4) 关键：使用"反向"loss - 最大化动作差异
    # 使用负MSE loss，这样梯度会让动作偏离安全动作
    loss = -F.mse_loss(action_adv_t, action_clean_t)

    # 5) 反向传播
    model.policy.zero_grad()
    if obs_t.grad is not None:
        obs_t.grad.zero_()
    loss.backward(retain_graph=False)

    if obs_t.grad is None:
        raise RuntimeError("obs gradient is None after backward.")

    # 6) FGSM step: 注意这里使用负梯度（因为我们要最大化loss）
    grad_sign = th.sign(obs_t.grad.data)
    adv_t = obs_t - attack_epsilon * grad_sign  # 注意是减号，因为我们最大化loss

    # 7) 转换回numpy
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_t.detach().cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
        adv_np = np.clip(adv_np, 0.0, 1.0)
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np


# 新增：专门为持续扰动设计的增强攻击函数
def targeted_attack_sac_td3(obs_np, model, epsilon, perception_type, device, target_action=None):
    """
    目标导向攻击：尝试让模型输出特定的目标动作。

    参数:
        obs_np: 原始观测
        model: 目标模型
        epsilon: 攻击强度
        perception_type: 观测类型
        device: 计算设备
        target_action: 目标动作，如果为None则自动选择危险动作

    返回:
        adv_obs: 对抗观测
    """
    model.policy.eval()

    # 获取当前动作
    with th.no_grad():
        try:
            action_current, _ = model.predict(obs_np, deterministic=True)
        except:
            action_current = model.predict(obs_np)

    # 如果没有指定目标动作，选择一个"危险"的动作
    if target_action is None:
        # 基于当前动作选择相反或极端的动作
        action_space = getattr(model, 'action_space', None)
        if action_space is not None and isinstance(action_space, gym.spaces.Box):
            # 对于连续动作空间，选择远离当前动作的方向
            action_range = action_space.high - action_space.low
            center = (action_space.high + action_space.low) / 2

            # 计算远离中心的方向
            direction = np.sign(action_current - center)
            # 稍微放大以确保效果
            target_action = center + direction * action_range * 0.8
            # 确保在边界内
            target_action = np.clip(target_action, action_space.low, action_space.high)
        else:
            # 默认策略：反转当前动作
            target_action = -action_current

    target_action_t = th.from_numpy(np.asarray(target_action)).float().to(device)
    if target_action_t.ndim == 1:
        target_action_t = target_action_t[None, ...]

    # 准备观测张量
    obs_t = _to_torch(obs_np, device, perception_type)
    obs_t.requires_grad_(True)

    # 获取对抗动作
    action_adv_t = None
    for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
        if candidate is None:
            continue
        try:
            action_adv_t = candidate(obs_t)
            break
        except Exception:
            action_adv_t = None

    if action_adv_t is None:
        try:
            act_np_fallback = model.policy._predict(obs_t.detach().cpu().numpy())
            action_adv_t = th.from_numpy(np.asarray(act_np_fallback)).float().to(device)
        except Exception:
            raise RuntimeError("无法从模型获得可导的 action 张量。")

    if action_adv_t.ndim == 1:
        action_adv_t = action_adv_t[None, ...]

    # 目标导向损失：最小化与目标动作的差异
    loss = F.mse_loss(action_adv_t, target_action_t)

    # 反向传播
    model.policy.zero_grad()
    if obs_t.grad is not None:
        obs_t.grad.zero_()
    loss.backward(retain_graph=False)

    if obs_t.grad is None:
        raise RuntimeError("obs gradient is None after backward.")

    # FGSM step
    grad_sign = th.sign(obs_t.grad.data)
    adv_t = obs_t + epsilon * grad_sign

    # 转换回numpy
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_t.detach().cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
        adv_np = np.clip(adv_np, 0.0, 1.0)
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np


def deepfool_improved(obs_np, model, max_iter, perception_type, device,tau=0.002, overshoot=0.1, debug=False):
    """
    DeepFool adapted: compute per-action gradients (Jacobian rows) and pick minimal
    perturbation that would increase any action component's difference to >= tau_component.
    """
    model.policy.eval()

    with th.no_grad():
        try:
            action0_np, _ = model.predict(obs_np, deterministic=True)
        except Exception:
            action0_np = model.predict(obs_np)
    action0_t = th.from_numpy(np.asarray(action0_np)).float().to(device)
    if action0_t.ndim == 1:
        action0_t = action0_t[None, ...]

    obs_t = _to_torch(obs_np, device, perception_type)
    adv = obs_t.clone().detach().to(device).requires_grad_(True)

    for itr in range(max_iter):
        if adv.grad is not None:
            adv.grad.zero_()

        # forward
        # get differentiable action
        action_adv_t = None
        for cand in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
            if cand is None:
                continue
            try:
                action_adv_t = cand(adv)
                break
            except Exception:
                action_adv_t = None
        if action_adv_t is None:
            raise RuntimeError("No differentiable actor found for DeepFool improved.")

        if action_adv_t.ndim == 1:
            action_adv_t = action_adv_t[None, ...]

        # compute difference per action dim
        diff = (action_adv_t.detach() - action0_t)  # no grad
        abs_diff = th.abs(diff)
        max_diff_val, max_idx = torch_max = th.max(abs_diff.view(-1)).item(), None

        cur_norm = th.norm(diff).item()
        if debug:
            print(f"[DF itr {itr}] cur_action_norm={cur_norm:.6f}")

        if cur_norm >= tau:
            if debug:
                print("Reached tau; stop.")
            break

        # For each action dimension compute gradient of that dim wrt input
        act_dim = action_adv_t.shape[-1]
        best_alpha = None
        best_delta = None
        for j in range(act_dim):
            # zero grads
            if adv.grad is not None:
                adv.grad.zero_()

            scalar = action_adv_t[0, j]  # single output
            scalar.backward(retain_graph=True)
            grad_j = adv.grad.detach().clone()
            grad_norm_sq = float(th.sum(grad_j.view(-1) ** 2).item())

            # f_j = current value - original
            f_j = float((action_adv_t.detach()[0, j] - action0_t[0, j]).item())
            # we want |f_j + grad_j · r| >= tau_component. take tau_component = tau / sqrt(act_dim) (or tune)
            tau_comp = tau / (act_dim ** 0.5)

            # solve alpha for r = alpha * grad_j: |f_j + alpha * (grad_j dot grad_j_flat)| >= tau_comp
            # approximate grad_j dot grad_j_flat = ||grad_j||^2
            denom = grad_norm_sq + 1e-12
            # compute required alpha magnitude
            need = (tau_comp - abs(f_j))
            if need <= 0:
                # this dimension already enough
                best_alpha = 0.0
                best_delta = th.zeros_like(adv)
                break
            alpha = need / denom
            if best_alpha is None or alpha < best_alpha:
                best_alpha = alpha
                best_delta = (alpha * grad_j)

        if best_delta is None:
            if debug:
                print("No usable gradient found; abort.")
            break

        adv = (adv.detach() + (1.0 + overshoot) * best_delta).clamp(0.0, 1.0).requires_grad_(True)

    # convert adv
    adv_final = adv.detach()
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_final.cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
    else:
        adv_cpu = adv_final.detach().cpu().numpy()[0]
        adv_cpu = np.transpose(adv_cpu, (1,2,0))
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np


def cw_l2_attack_sac_td3(obs_np, model, perception_type, device, c=1e-2, steps=100, lr=1e-2):
    """
    Carlini-Wagner L2 风格（优化最小 L2 扰动以改变 actor 输出）。
    c: 损失权重，越大越注重 misclassification（此处为 action 差异）
    steps: 内部优化步数
    lr: 学习率
    注意：计算量较大，慎用在大量评估循环里。
    """
    model.policy.eval()
    # baseline action
    with th.no_grad():
        try:
            action0_np, _ = model.predict(obs_np, deterministic=True)
        except Exception:
            action0_np = model.predict(obs_np)
    action0_t = th.from_numpy(np.asarray(action0_np)).float().to(device)
    if action0_t.ndim == 1:
        action0_t = action0_t[None, ...]

    obs_t = _to_torch(obs_np, device, perception_type)
    # paramize perturbation via tanh-space (optional); for simplicity直接优化 delta
    delta = th.zeros_like(obs_t, requires_grad=True, device=device)
    opt = th.optim.Adam([delta], lr=lr)

    for step in range(steps):
        perturbed = obs_t + delta
        # clamp to valid range
        if perception_type == 'vector' or perception_type == 'lgmd':
            perturbed_c = th.clamp(perturbed, 0.0, 1.0)
        else:
            perturbed_c = th.clamp(perturbed, 0.0, 1.0)

        # forward
        action_adv_t = None
        for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
            if candidate is None:
                continue
            try:
                action_adv_t = candidate(perturbed_c)
                break
            except Exception:
                action_adv_t = None
        if action_adv_t is None:
            act_np_fallback = model.policy._predict(perturbed_c.detach().cpu().numpy())
            action_adv_t = th.from_numpy(np.asarray(act_np_fallback)).float().to(device)
        if action_adv_t.ndim == 1:
            action_adv_t = action_adv_t[None, ...]

        # loss = ||delta||_2^2 + c * MSE(action_adv, action0)
        loss_l2 = th.sum(delta ** 2)
        loss_action = F.mse_loss(action_adv_t, action0_t)
        loss = loss_l2 - c * loss_action

        opt.zero_grad()
        loss.backward()
        opt.step()

    adv_t = th.clamp(obs_t + delta.detach(), 0.0, 1.0)

    # convert back
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_t.detach().cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)
    return adv_np



def bim_attack_sac_td3(obs_np, model, epsilon, perception_type, device, num_iter=20):
    """
    Basic Iterative Method (BIM) = iterative FGSM (like PGD but using sign step and projection).
    接口与 pgd_attack_sac_td3 保持相似，alpha = epsilon / num_iter
    """
    alpha = epsilon / num_iter
    model.policy.eval()

    with th.no_grad():
        try:
            action_clean_np, _ = model.predict(obs_np, deterministic=True)
        except Exception:
            action_clean_np = model.predict(obs_np)
    action_clean_t = th.from_numpy(np.asarray(action_clean_np)).float().to(device)
    if action_clean_t.ndim == 1:
        action_clean_t = action_clean_t[None, ...]

    obs_t = _to_torch(obs_np, device, perception_type)
    adv_t = obs_t.clone().detach().requires_grad_(True)

    for _ in range(num_iter):
        # forward to get differentiable action
        action_adv_t = None
        for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
            if candidate is None:
                continue
            try:
                action_adv_t = candidate(adv_t)
                break
            except Exception:
                action_adv_t = None
        if action_adv_t is None:
            act_np_fallback = model.policy._predict(adv_t.detach().cpu().numpy())
            action_adv_t = th.from_numpy(np.asarray(act_np_fallback)).float().to(device)

        if action_adv_t.ndim == 1:
            action_adv_t = action_adv_t[None, ...]

        loss = F.mse_loss(action_adv_t, action_clean_t)
        model.policy.zero_grad()
        if adv_t.grad is not None:
            adv_t.grad.zero_()
        loss.backward(retain_graph=False)
        if adv_t.grad is None:
            raise RuntimeError("adv gradient is None in BIM.")

        adv_t = adv_t + alpha * th.sign(adv_t.grad.data)
        adv_t = th.min(th.max(adv_t, obs_t - epsilon), obs_t + epsilon)
        adv_t = th.clamp(adv_t, 0.0, 1.0).detach().requires_grad_(True)

    # convert back
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_t.detach().cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np



import torch as th
import torch.nn.functional as F
import numpy as np

def check_action_confidence(action, action_space, threshold=0.5, model=None, obs=None, device=None):
    """
    改进的置信度检测：检查智能体是否强烈偏向某一动作。
    
    参数:
        action: numpy array, 当前动作
        action_space: gym.spaces.Box, 动作空间
        threshold: float, 触发攻击的阈值 (0-1)，默认0.5表示动作幅度超过50%边界时触发
        model: 可选，如果提供，会检查policy的确定性（多次采样的方差）
        obs: 可选，当前观测，用于policy采样
        device: 可选，计算设备
    
    返回:
        bool: True表示应该触发攻击（动作强烈偏向），False表示不攻击
    """
    if not isinstance(action_space, gym.spaces.Box):
        # 如果不是连续动作空间，默认不触发（或可以扩展支持离散动作）
        return False
    
    action = np.asarray(action)
    action_low = action_space.low
    action_high = action_space.high
    
    # 方法1: 检查动作值是否接近边界（改进版）
    # 归一化动作到 [0, 1] 范围
    action_normalized = (action - action_low) / (action_high - action_low)
    
    # 计算每个动作维度距离中心(0.5)的距离
    # 如果动作接近边界（接近0或1），说明智能体强烈偏向
    distances_from_center = np.abs(action_normalized - 0.5) * 2  # 归一化到[0,1]
    max_distance = np.max(distances_from_center)
    
    # 方法2: 如果提供了model和obs，检查policy的确定性（多次采样的方差）
    policy_variance = None
    if model is not None and obs is not None and device is not None:
        try:
            # 采样多次动作，计算方差
            num_samples = 10
            actions_samples = []
            for _ in range(num_samples):
                # 使用非确定性采样
                sampled_action, _ = model.predict(obs, deterministic=False)
                actions_samples.append(sampled_action)
            
            actions_array = np.array(actions_samples)
            # 计算每个动作维度的方差
            action_variances = np.var(actions_array, axis=0)
            # 计算每个动作维度的范围
            action_ranges = action_high - action_low
            # 归一化每个维度的方差（相对于该维度的范围）
            normalized_variances = action_variances / (action_ranges ** 2 + 1e-8)
            # 取平均归一化方差
            normalized_variance = np.mean(normalized_variances)
            
            # 如果方差小（确定性高），说明policy很确定，应该攻击
            # 方差越小，确定性越高，置信度越高
            # 使用更合理的归一化：如果方差小于范围的1%，认为确定性很高
            # policy_confidence范围应该在[0, 1]，当normalized_variance很小时，confidence接近1
            policy_confidence = 1.0 / (1.0 + normalized_variance * 100)  # 使用sigmoid-like函数
            
            # 结合两种方法：动作接近边界 OR policy确定性高
            method1_trigger = max_distance >= threshold
            method2_trigger = policy_confidence >= threshold
            should_attack = method1_trigger or method2_trigger
            
            # 记录统计信息（如果提供了stats字典）
            if hasattr(model, '_confidence_stats') or (hasattr(model, 'policy') and hasattr(model.policy, '_confidence_stats')):
                stats = getattr(model, '_confidence_stats', None) or getattr(model.policy, '_confidence_stats', None)
                if stats is not None:
                    if method1_trigger:
                        stats['method1_triggers'] += 1
                    if method2_trigger:
                        stats['method2_triggers'] += 1
                    stats['max_distances'].append(max_distance)
                    stats['policy_confidences'].append(policy_confidence)
        except Exception as e:
            # 如果采样失败，回退到方法1
            should_attack = max_distance >= threshold
            # 记录方法2失败
            if hasattr(model, '_confidence_stats') or (hasattr(model.policy, '_confidence_stats')):
                stats = getattr(model, '_confidence_stats', None) or getattr(model.policy, '_confidence_stats', None)
                if stats is not None:
                    stats['method2_failed'] += 1
    else:
        # 只使用方法1
        should_attack = max_distance >= threshold
    
    return should_attack



def check_risk_trigger(env, risk_threshold, risk_margin):
    """
    env.min_distance_to_obstacles 越小越危险
    risk = (risk_distance - min_dist) / risk_distance, 其中 risk_distance = crash_distance + risk_margin
    触发条件：risk >= risk_threshold
    """
    # 这两个值在 airsim_env.py 中已有
    min_dist = float(getattr(env, "min_distance_to_obstacles", 1e9))
    crash_dist = float(getattr(env, "crash_distance", 0.0))

    risk_distance = max(1e-6, crash_dist + float(risk_margin))

    # 归一化风险到 [0,1]
    risk = (risk_distance - min_dist) / risk_distance
    risk = float(np.clip(risk, 0.0, 1.0))

    should_attack = (risk >= float(risk_threshold))
    return should_attack, risk, min_dist, risk_distance


def check_statistical_critical_state(obs, model, device, action_space, threshold=0.7, stats=None, entropy_scale=8.0, q_diff_scale=0.5, variance_scale=2.0, impact_predictor=None, current_reward=0.0, perception_type='depth', env=None, risk_threshold=0.6, risk_margin=1.0):
    """
    基于论文统计方法的临界状态检测
    检测动作分布的统计特征来识别关键状态

    参数:
        obs: 当前观测
        model: SAC/TD3模型
        device: 计算设备
        action_space: 动作空间
        threshold: 触发阈值
        stats: 可选的统计信息字典，用于记录诊断信息

    返回:
        should_attack: 是否应该攻击
        action_entropy: 动作熵值
        q_value_diff: Q值差异
        action_variance: 动作方差
    """
    try:
        # 1. 计算动作方差（通过多次采样）
        num_samples = 20  # 增加采样次数以获得更稳定的统计
        actions_samples = []

        for _ in range(num_samples):
            sampled_action, _ = model.predict(obs, deterministic=False)
            actions_samples.append(sampled_action)

        actions_array = np.array(actions_samples)

        # 计算动作方差
        action_variances = np.var(actions_array, axis=0)
        mean_action_variance = np.mean(action_variances)

        # 归一化方差（相对于动作范围）
        action_range = action_space.high - action_space.low
        normalized_variance = mean_action_variance / (np.mean(action_range ** 2) + 1e-8)

        # 2. 计算动作分布熵（近似）
        # 对于连续动作，使用更合理的熵计算
        # 熵应该在[0, +∞)范围内，高熵表示不确定性高
        if normalized_variance < 1e-6:
            action_entropy = 0.0  # 方差为0时，熵为0（完全确定）
        else:
            action_entropy = -np.log(np.clip(normalized_variance, 1e-6, 1.0))  # 限制在合理范围内

        # 3. 计算Q值差异（如果模型有双Q网络，如SAC）
        q_value_diff = 0.0
        try:
            # 尝试访问SAC/Twin Critic的Q网络
            if hasattr(model, 'critic') and model.critic is not None:
                critic = model.critic

                # 方法1: 直接访问q_networks (SAC的标准结构)
                if hasattr(critic, 'q_networks') and len(critic.q_networks) >= 2:
                    # 准备输入数据 - 确保有batch维度
                    obs_t = _to_torch(obs, device, perception_type)  # 使用正确的perception_type
                    action_t = th.from_numpy(np.mean(actions_array, axis=0)).float().to(device)

                    # 确保action_t有正确的形状 (batch_size, action_dim)
                    if action_t.dim() == 1:
                        action_t = action_t.unsqueeze(0)  # 添加batch维度

                    with th.no_grad():
                        # 使用critic的forward方法，这样更安全
                        try:
                            q_outputs = critic(obs_t, action_t)
                            if len(q_outputs) >= 2:
                                q_value_diff = th.abs(q_outputs[0] - q_outputs[1]).mean().item()
                            else:
                                q_value_diff = 0.0
                        except Exception as e:
                            # 如果forward方法失败，回退到手动计算
                            try:
                                features = critic.extract_features(obs_t)
                                q_input = th.cat([features, action_t], dim=1)
                                q1_output = critic.q_networks[0](q_input)
                                q2_output = critic.q_networks[1](q_input)
                                q_value_diff = th.abs(q1_output - q2_output).mean().item()
                            except Exception as e2:
                                q_value_diff = 0.0

                # 方法2: 访问policy的critic (TD3/SAC的另一种结构)
                elif hasattr(model, 'policy') and hasattr(model.policy, 'critic'):
                    policy_critic = model.policy.critic
                    if hasattr(policy_critic, 'q_networks') and len(policy_critic.q_networks) >= 2:
                        obs_t = _to_torch(obs, device, perception_type)
                        action_t = th.from_numpy(np.mean(actions_array, axis=0)).float().to(device)

                        # 确保action_t有正确的形状
                        if action_t.dim() == 1:
                            action_t = action_t.unsqueeze(0)

                        with th.no_grad():
                            try:
                                q_outputs = policy_critic(obs_t, action_t)
                                if len(q_outputs) >= 2:
                                    q_value_diff = th.abs(q_outputs[0] - q_outputs[1]).mean().item()
                                else:
                                    q_value_diff = 0.0
                            except Exception as e:
                                try:
                                    features = policy_critic.extract_features(obs_t)
                                    q_input = th.cat([features, action_t], dim=1)
                                    q1_output = policy_critic.q_networks[0](q_input)
                                    q2_output = policy_critic.q_networks[1](q_input)
                                    q_value_diff = th.abs(q1_output - q2_output).mean().item()
                                except Exception as e2:
                                    q_value_diff = 0.0

                # 方法3: 尝试q1_forward等方法 (TD3风格)
                elif hasattr(critic, 'q1_forward'):
                    obs_t = _to_torch(obs, device, perception_type)
                    action_t = th.from_numpy(np.mean(actions_array, axis=0)).float().to(device)

                    with th.no_grad():
                        # 对于只有一个critic的情况，差异设为0
                        q_value_diff = 0.0

        except Exception as e:
            # 如果无法计算Q值差异，记录错误但不中断程序
            print(f"[Q-Value Debug] Failed to compute Q-value difference: {e}")
            q_value_diff = 0.0

        # 4. 综合判断是否为关键状态
        # 使用传入的缩放参数计算阈值
        entropy_threshold = threshold * entropy_scale
        q_diff_threshold = threshold * q_diff_scale
        variance_threshold = threshold * variance_scale

        is_entropy_critical = action_entropy > entropy_threshold
        is_q_diff_critical = q_value_diff > q_diff_threshold
        is_variance_critical = normalized_variance > variance_threshold

        # 5. 长期影响预测 (如果提供预测器)
        impact_score = 0.0
        if impact_predictor is not None:
            impact_score = predict_attack_impact(obs, model, device, action_space, impact_predictor, current_reward)
            # 长期影响阈值：如果预测攻击会带来显著负面影响，则降低攻击倾向
            impact_threshold = -0.2  # 如果预测负面影响超过0.2，则不攻击
            is_impact_negative = impact_score < impact_threshold
        else:
            is_impact_negative = False

        # 混合智能判断：结合Statistical和环境风险
        risk_boost = 1.0  # 默认风险权重

        # 如果提供了环境信息，计算风险权重
        if env is not None:
            try:
                min_dist = float(getattr(env, "min_distance_to_obstacles", 1e9))
                crash_dist = float(getattr(env, "crash_distance", 2.0))
                risk_distance = max(1e-6, crash_dist + float(risk_margin))
                risk = (risk_distance - min_dist) / risk_distance
                risk = float(np.clip(risk, 0.0, 1.0))

                # 高风险区域降低阈值，提高攻击率
                if risk > risk_threshold:
                    risk_boost = 0.7  # 降低阈值30%，更容易触发攻击
                elif risk > risk_threshold * 0.7:
                    risk_boost = 0.85  # 降低阈值15%
                else:
                    risk_boost = 1.0   # 正常阈值
            except:
                risk_boost = 1.0

        # 应用风险权重调整阈值
        effective_entropy_threshold = entropy_threshold * risk_boost
        effective_q_diff_threshold = q_diff_threshold * risk_boost
        effective_variance_threshold = variance_threshold * risk_boost

        # 重新评估关键状态（使用风险调整后的阈值）
        is_entropy_critical_boosted = action_entropy > effective_entropy_threshold
        is_q_diff_critical_boosted = q_value_diff > effective_q_diff_threshold
        is_variance_critical_boosted = normalized_variance > effective_variance_threshold

        # 质量优先的逻辑：优先考虑最重要的指标
        if is_q_diff_critical_boosted:
            # Q值差异是最高质量指标
            statistical_attack = True
        elif is_entropy_critical_boosted and q_value_diff > q_diff_threshold * 0.8:
            # 熵触发且Q值差异较高
            statistical_attack = True
        elif risk_boost < 1.0 and (is_entropy_critical_boosted or is_variance_critical_boosted):
            # 在高风险区域，熵或方差触发也认为是有效的
            statistical_attack = True
        else:
            statistical_attack = False

        should_attack = statistical_attack and not is_impact_negative

        # 记录统计信息
        if stats is not None:
            if is_entropy_critical:
                stats['entropy_triggers'] += 1
            if is_q_diff_critical:
                stats['q_diff_triggers'] += 1
            if is_variance_critical:
                stats['variance_triggers'] += 1

            stats['action_entropies'].append(action_entropy)
            stats['q_value_diffs'].append(q_value_diff)
            stats['action_variances'].append(normalized_variance)

        return should_attack, action_entropy, q_value_diff, normalized_variance

    except Exception as e:
        # 如果计算失败，默认不攻击
        print(f"[Statistical Check Error] {e}")
        return False, 0.0, 0.0, 0.0





def mim_attack_sac_td3(obs_np, model, epsilon, perception_type, device,
                       num_iter=20, decay=1.0, debug=False):
    """
    修复版 MIM（Momentum Iterative Method）。
    - 使用 .reshape 代替 .view，避免非连续张量报错。
    - 若模型没有可导 actor，则抛错而不是悄然 fallback（避免返回 clean obs）。
    - 接口与之前保持一致，返回 adv_obs numpy（同 obs_np 格式）。
    """
    alpha = float(epsilon) / float(max(1, num_iter))
    model.policy.eval()

    # 先获取 clean action（用于构造 loss）
    try:
        maybe = model.predict(obs_np, deterministic=True)
        if isinstance(maybe, tuple) or isinstance(maybe, list):
            action_clean_np = maybe[0]
        else:
            action_clean_np = maybe
    except Exception:
        action_clean_np = model.predict(obs_np)

    action_clean_t = th.from_numpy(np.asarray(action_clean_np)).float().to(device)
    if action_clean_t.ndim == 1:
        action_clean_t = action_clean_t[None, ...]

    # 把 obs 转为 torch tensor（调用你原来的 _to_torch）
    obs_t = _to_torch(obs_np, device, perception_type)   # 保留你原有转换函数
    adv_t = obs_t.clone().detach().requires_grad_(True)

    # 找到一个可导 actor：优先 model.actor -> model.policy.actor
    actor = None
    for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
        if candidate is None:
            continue
        try:
            test_out = candidate(obs_t.clone().detach())
            if isinstance(test_out, th.Tensor):
                actor = candidate
                break
        except Exception:
            continue

    if actor is None:
        raise RuntimeError("MIM attack requires a torch actor (model.actor or model.policy.actor). "
                           "Found none or actor.forward is not torch-differentiable.")

    # momentum buffer
    g = th.zeros_like(adv_t).to(device)

    for itr in range(num_iter):
        if adv_t.grad is not None:
            adv_t.grad.detach_()
            adv_t.grad.zero_()

        action_adv_t = actor(adv_t)
        if action_adv_t.ndim == 1:
            action_adv_t = action_adv_t[None, ...]

        loss = F.mse_loss(action_adv_t, action_clean_t)
        loss.backward(retain_graph=False)

        if adv_t.grad is None:
            raise RuntimeError(f"adv_t.grad is None at iteration {itr} — "
                               "check that actor->adv_t path is differentiable.")

        grad = adv_t.grad.data
        grad_flat = grad.detach().reshape(grad.shape[0], -1) if grad.dim() > 1 else grad.detach().reshape(1, -1)
        grad_norm = th.norm(grad_flat, p=1, dim=1).view(-1, *([1] * (grad.dim() - 1)))
        grad_normalized = grad / (grad_norm + 1e-12)

        g = decay * g + grad_normalized
        adv_t = adv_t + alpha * th.sign(g)

        adv_t = th.min(th.max(adv_t, obs_t - epsilon), obs_t + epsilon)
        adv_t = th.clamp(adv_t, 0.0, 1.0).detach().requires_grad_(True)

        if debug:
            with th.no_grad():
                try:
                    cur_action = actor(adv_t)
                    diff_norm = th.norm(cur_action.detach().reshape(cur_action.shape[0], -1) - action_clean_t.detach().reshape(action_clean_t.shape[0], -1), dim=1)
                except Exception:
                    diff_norm = th.tensor([-1.0])
            print(f"[MIM] itr={itr} loss={loss.item():.6f} action_diff_norm={diff_norm.cpu().numpy()}")

    # 转回 numpy
    if perception_type in ('vector', 'lgmd'):
        adv_np = adv_t.detach().cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np



# ==================== end =====================


class EvaluateThread(QtCore.QThread):
    # signals
    def __init__(self, eval_path, config, model_file, eval_ep_num, eval_env=None, eval_dynamics=None):
        super(EvaluateThread, self).__init__()
        print("init training thread")

        # config
        self.cfg = ConfigParser()
        self.cfg.read(config)

        # change eval_env and eval_dynamics if is not None
        if eval_env is not None:
            self.cfg.set('options', 'env_name', eval_env)

        if eval_env == 'NH_center':
            self.cfg.set('environment', 'accept_radius', str(1))

        if eval_dynamics is not None:
            self.cfg.set('options', 'dynamic_name', eval_dynamics)

        # 从 config 读取攻击相关参数
        self.enable_attack = self.cfg.getboolean('options', 'enable_attack', fallback=False)
        self.attack_type = self.cfg.get('options', 'attack_type', fallback='none')  # fgsm, pgd, random, targeted, none
        self.attack_epsilon = self.cfg.getfloat('options', 'attack_epsilon', fallback=0.0)
        # 动作置信度阈值（0-1），超过此阈值才触发攻击。默认0.5
        self.attack_confidence_threshold = self.cfg.getfloat('options', 'attack_confidence_threshold', fallback=0.5)
        # 统计方法阈值缩放参数
        self.statistical_entropy_scale = self.cfg.getfloat('options', 'statistical_entropy_scale', fallback=3.0)
        self.statistical_q_diff_scale = self.cfg.getfloat('options', 'statistical_q_diff_scale', fallback=0.5)
        self.statistical_variance_scale = self.cfg.getfloat('options', 'statistical_variance_scale', fallback=0.5)
        # 触发模式（兼容旧配置），缺省为基于置信度
        self.attack_trigger_mode = self.cfg.get('options', 'attack_trigger_mode', fallback='confidence')
        # attack_on_confidence 不再从 config 读取，统一由 trigger_mode 推断
        self.attack_on_confidence = (self.attack_trigger_mode == 'confidence')
        # 其他风险相关参数
        self.risk_margin = self.cfg.getfloat('options', 'risk_margin', fallback=3.0)
        self.risk_threshold = self.cfg.getfloat('options', 'risk_threshold', fallback=0.6)
        # 步数间隔攻击相关参数
        self.step_interval_n = self.cfg.getint('options', 'step_interval_n', fallback=5)
        self.step_attack_m = self.cfg.getint('options', 'step_attack_m', fallback=2)
        # 随机攻击概率参数
        self.random_attack_probability = self.cfg.getfloat('options', 'random_attack_probability', fallback=0.3)
        # 连续攻击相关参数
        self.continuous_attack_steps = self.cfg.getint('options', 'continuous_attack_steps', fallback=3)

        # Smart Q-Entropy融合攻击参数
        self.smart_q_entropy_threshold = self.cfg.getfloat('options', 'smart_q_entropy_threshold', fallback=0.5)
        self.smart_q_entropy_risk_weight = self.cfg.getfloat('options', 'smart_q_entropy_risk_weight', fallback=0.3)
        self.smart_q_entropy_entropy_weight = self.cfg.getfloat('options', 'smart_q_entropy_entropy_weight', fallback=0.35)
        self.smart_q_entropy_q_weight = self.cfg.getfloat('options', 'smart_q_entropy_q_weight', fallback=0.35)

        # 调试与计数
        self.debug_attack = self.cfg.getboolean('options', 'debug_attack', fallback=False)
        self.attack_count = 0  # 只计数实际生成对抗观测的次数
        self.total_steps = 0
        # 连续攻击状态跟踪
        self.continuous_attack_counter = 0  # 剩余连续攻击步数

        
        # 用于诊断置信度检测问题
        self.confidence_stats = {
            'method1_triggers': 0,  # 方法1触发的次数
            'method2_triggers': 0,  # 方法2触发的次数
            'method2_failed': 0,    # 方法2失败的次数
            'max_distances': [],    # 记录max_distance值
            'policy_confidences': []  # 记录policy_confidence值
        }

        # 用于统计方法的诊断
        self.statistical_stats = {
            'entropy_triggers': 0,     # 熵触发次数
            'q_diff_triggers': 0,      # Q值差异触发次数
            'variance_triggers': 0,    # 方差触发次数
            'action_entropies': [],    # 记录动作熵值
            'q_value_diffs': [],       # 记录Q值差异
            'action_variances': []     # 记录动作方差
        }

        # 用于Smart Q-Entropy融合攻击的诊断
        self.smart_q_entropy_stats = {
            'fusion_scores': [],       # 融合分数
            'risks': [],               # 风险值
            'entropies': [],           # 熵值
            'q_diffs': [],             # Q值差异
            'q_stds': [],              # Q值标准差
            'triggers': 0              # 触发次数
        }

        # 长期影响预测器 - 设为None以禁用，只使用Q值和动作熵
        self.impact_predictor = None
        # 如果需要启用预测器，取消下面这行的注释：
        # self.impact_predictor = LongTermImpactPredictor(sequence_length=15)

        # 自适应阈值调节器 - 已禁用，使用静态阈值
        # self.episode_tuner = EpisodeAdaptiveTuner(
        #     initial_entropy_scale=self.statistical_entropy_scale,
        #     initial_q_diff_scale=self.statistical_q_diff_scale,
        #     initial_variance_scale=self.statistical_variance_scale
        # )


        self.env = gym.make('airsim-env-v0')
        self.env.set_config(self.cfg)

        self.eval_path = eval_path
        self.model_file = model_file
        self.eval_ep_num = eval_ep_num
        self.eval_env = self.cfg.get('options', 'env_name')
        self.eval_dynamics = self.cfg.get('options', 'dynamic_name')

    def terminate(self):
        print('Evaluation terminated')

    def run(self):
        # self.run_rule_policy()
        return self.run_drl_model()

    def run_drl_model(self):
        print('start evaluation')
        algo = self.cfg.get('options', 'algo')
        if algo == 'TD3':
            model = TD3.load(self.model_file, env=self.env)
        elif algo == 'SAC':
            model = SAC.load(self.model_file, env=self.env)
        elif algo == 'PPO':
            model = PPO.load(self.model_file, env=self.env)
        else:
            raise Exception('algo set error {}'.format(algo))
        self.env.model = model
        
        # 将置信度统计字典附加到model上，方便check_action_confidence访问
        model._confidence_stats = self.confidence_stats

        obs = self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        episode_successes = []
        episode_crashes = []
        traj_list_all = []
        action_list_all = []
        state_list_all = []
        obs_list_all = []

        traj_list = []
        action_list = []
        state_raw_list = []
        step_num_list = []
        obs_list = []
        
        # 用于收集所有episode的综合统计信息
        all_episode_rewards = []
        all_episode_steps = []
        all_attack_rates = []  # 每个episode的攻击率
        all_max_distances = []  # 所有episode的max_distance
        all_policy_confidences = []  # 所有episode的policy_confidence
        all_method1_triggers = []
        all_method2_triggers = []
        all_method2_failures = []

        # 统计方法的统计数据
        all_entropy_avgs = []  # 所有episode的熵平均值
        all_q_diff_avgs = []   # 所有episode的Q值差异平均值
        all_variance_avgs = [] # 所有episode的方差平均值
        all_entropy_triggers = []  # 熵触发次数
        all_q_diff_triggers = []   # Q值差异触发次数
        all_variance_triggers = [] # 方差触发次数
        cv2.waitKey()

        while episode_num < self.eval_ep_num:

            # Episode开始时初始化调节器 - 已禁用自动调节
            # if self.attack_trigger_mode in ('statistical', 'critical_state'):
            #     self.episode_tuner.start_episode()

            # ==================== 攻击 =====================
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')

            # 先获取当前动作，用于判断是否需要攻击
            should_attack_this_step = False
            self.total_steps += 1
            
            if self.enable_attack and self.attack_type != 'none':
                # 如果启用了"只在关键时刻攻击"模式
                '''
                if self.attack_trigger_mode == 'confidence':
                    # 先获取当前观测下的动作（不攻击时）
                    temp_action, _ = model.predict(obs, deterministic=True)
                    # 检查动作置信度（传入model和obs以检查policy确定性）
                    should_attack_this_step = check_action_confidence(
                        temp_action, 
                        self.env.action_space, 
                        self.attack_confidence_threshold,
                        model=model,
                        obs=obs,
                        device=device
                    )
                    
                    # 调试信息（已移除每步打印，只在episode结束时统计）
                    # 如果需要详细的每步调试信息，可以设置 debug_attack=True 并取消下面的注释
                    # if self.debug_attack and should_attack_this_step:
                    #     action_normalized = (temp_action - self.env.action_space.low) / \
                    #                       (self.env.action_space.high - self.env.action_space.low)
                    #     max_dist = np.max(np.abs(action_normalized - 0.5) * 2)
                    #     print(f"[Attack Debug] Step {self.total_steps}: Attack triggered! "
                    #           f"Max action distance: {max_dist:.3f}, Threshold: {self.attack_confidence_threshold}")
                else:
                    # 传统模式：每轮都攻击
                    should_attack_this_step = True
                '''
                if self.enable_attack and self.attack_type != 'none':

                    if self.attack_trigger_mode == 'always':
                        should_attack_this_step = True
                        if self.debug_attack and should_attack_this_step:
                            print(f"[Attack Trigger] Always mode: attack triggered at step {self.total_steps}")
                    elif self.attack_trigger_mode == 'confidence':
                        temp_action, _ = model.predict(obs, deterministic=True)
                        should_attack_this_step = check_action_confidence(
                            temp_action,
                            self.env.action_space,
                            self.attack_confidence_threshold,
                            model=model,
                            obs=obs,
                            device=device
                        )
                    elif self.attack_trigger_mode == 'risk':
                        should_attack_this_step, risk, min_dist, risk_dist = check_risk_trigger(
                            self.env,
                            self.risk_threshold,
                            self.risk_margin
                        )
                    elif self.attack_trigger_mode == 'smart':
                        # 智能攻击模式：只在真正危险且模型非常确定时攻击
                        # 1) 先用风险触发做预筛选
                        should_attack_risk, risk, min_dist, risk_dist = check_risk_trigger(
                            self.env,
                            self.risk_threshold,
                            self.risk_margin
                        )
                        # 只有在风险已经超过基础阈值时，才再看更严格条件
                        if should_attack_risk:
                            # 2) 使用更严格的风险阈值（在基础阈值和0.9之间）
                            strict_risk_threshold = min(0.9, self.risk_threshold)
                            is_high_risk = risk >= strict_risk_threshold
                            
                            if is_high_risk:
                                # 3) 使用更高的置信度阈值（比配置的再严格一些）
                                temp_action, _ = model.predict(obs, deterministic=True)
                                strict_conf_threshold = max(self.attack_confidence_threshold, 0.7)
                                is_confident = check_action_confidence(
                                    temp_action,
                                    self.env.action_space,
                                    strict_conf_threshold,
                                    model=model,
                                    obs=obs,
                                    device=device
                                )
                                # 必须同时满足：高风险 AND 高置信度
                                should_attack_this_step = is_confident
                            else:
                                should_attack_this_step = False
                        else:
                            should_attack_this_step = False
                    elif self.attack_trigger_mode == 'smart_q':
                        # SmartQ模式：先用风险触发预筛选（同Smart），再使用Q值差异作为置信度指标
                        should_attack_risk, risk, min_dist, risk_dist = check_risk_trigger(
                            self.env,
                            self.risk_threshold,
                            self.risk_margin
                        )
                        if should_attack_risk:
                            strict_risk_threshold = min(0.9, self.risk_threshold)
                            is_high_risk = risk >= strict_risk_threshold

                            if is_high_risk:
                                # 使用SAC双Q结构的Q值差异作为触发器
                                # 采用严格的 q_diff scale 以降低误触
                                strict_q_diff_scale = self.statistical_q_diff_scale

                                # 调用统计函数以获得 q_value_diff（该函数会对critic做多种fallback访问）
                                # 将 stats 传入以便记录到统计方法的诊断信息里
                                _, _, q_value_diff, _ = check_statistical_critical_state(
                                    obs, model, device, self.env.action_space, self.attack_confidence_threshold,
                                    self.statistical_stats,  # 记录统计信息
                                    self.statistical_entropy_scale,
                                    self.statistical_q_diff_scale,
                                    self.statistical_variance_scale,
                                    None, 0.0,  # 不使用长期影响预测器
                                    'depth',  # perception_type（与训练一致）
                                    None, 0.6, 1.0  # 不使用环境风险感知（这里我们已在外面处理risk）
                                )

                                q_threshold = self.attack_confidence_threshold * strict_q_diff_scale
                                should_attack_this_step = (q_value_diff > q_threshold)

                                if self.debug_attack and should_attack_this_step:
                                    print(f"[Attack Trigger] SmartQ mode: attack triggered at step {self.total_steps} "
                                          f"(risk: {risk:.3f}, q_diff: {q_value_diff:.3f}, q_threshold: {q_threshold:.3f})")
                            else:
                                should_attack_this_step = False
                        else:
                            should_attack_this_step = False
                    elif self.attack_trigger_mode == 'smart_q_entropy':
                        # Smart Q-Entropy融合攻击模式：融合风险、动作熵和Q值特征
                        should_attack_this_step, fusion_score, components = check_smart_q_entropy_trigger(
                            env=self.env,
                            model=model,
                            obs=obs,
                            device=device,
                            action_space=self.env.action_space,
                            perception_type='depth',
                            fusion_threshold=self.smart_q_entropy_threshold,
                            risk_threshold=self.risk_threshold,
                            risk_margin=self.risk_margin,
                            risk_weight=self.smart_q_entropy_risk_weight,
                            entropy_weight=self.smart_q_entropy_entropy_weight,
                            q_weight=self.smart_q_entropy_q_weight,
                            stats=self.smart_q_entropy_stats,
                            debug=self.debug_attack
                        )

                        # 保存融合分数和风险值供攻击函数使用
                        if not hasattr(self, '_current_fusion_score'):
                            self._current_fusion_score = 0.0
                            self._current_risk_value = 0.0
                        self._current_fusion_score = fusion_score
                        self._current_risk_value = components.get('risk', 0.0)

                        if self.debug_attack and should_attack_this_step:
                            print(f"[Attack Trigger] Smart Q-Entropy mode: attack triggered at step {self.total_steps} "
                                  f"(fusion_score: {fusion_score:.3f}, risk: {components.get('risk', 0.0):.3f}, "
                                  f"entropy: {components.get('action_entropy', 0.0):.3f}, "
                                  f"q_diff: {components.get('q_diff', 0.0):.3f})")
                    elif self.attack_trigger_mode == 'smartc':
                        # SmartC模式：结合Smart的风险感知和Statistical的统计特征评估
                        # 1) 先用风险触发做预筛选（和Smart模式相同）
                        should_attack_risk, risk, min_dist, risk_dist = check_risk_trigger(
                            self.env,
                            self.risk_threshold,
                            self.risk_margin
                        )
                        # 只有在风险已经超过基础阈值时，才再看更严格条件
                        if should_attack_risk:
                            # 2) 使用适中的风险阈值（在基础阈值和0.8之间）
                            strict_risk_threshold = min(0.8, self.risk_threshold * 1.5)
                            is_high_risk = risk >= strict_risk_threshold

                            if is_high_risk:
                                # 3) 使用Statistical模式的Q值差异和动作熵评估（设置严格的阈值）
                                # 使用配置中的statistical参数，设置严格阈值以降低攻击频率
                                strict_entropy_scale = max(self.statistical_entropy_scale, 8.0)  # 严格的熵阈值
                                strict_q_diff_scale = max(self.statistical_q_diff_scale, 3.0)    # 严格的Q差异阈值
                                strict_variance_scale = max(self.statistical_variance_scale, 3.0) # 严格的方差阈值

                                should_attack_statistical, action_entropy, q_value_diff, action_variance = check_statistical_critical_state(
                                    obs, model, device, self.env.action_space, self.attack_confidence_threshold,
                                    None,  # 不记录统计信息
                                    strict_entropy_scale, strict_q_diff_scale, strict_variance_scale,
                                    None, 0.0,  # 不使用长期影响预测器
                                    'depth',  # perception_type
                                    None, 0.6, 1.0  # 不使用环境风险感知
                                )
                                # 必须同时满足：高风险 AND 统计特征触发
                                should_attack_this_step = should_attack_statistical

                                if self.debug_attack and should_attack_this_step:
                                    print(f"[Attack Trigger] SmartC mode: attack triggered at step {self.total_steps} "
                                          f"(risk: {risk:.3f}, entropy: {action_entropy:.3f}, q_diff: {q_value_diff:.3f}, variance: {action_variance:.3f})")
                            else:
                                should_attack_this_step = False
                        else:
                            should_attack_this_step = False
                    elif self.attack_trigger_mode == 'smartc_continuous':
                        # SmartC Continuous模式：在SmartC基础上添加连续攻击机制
                        # 一旦触发，就连续攻击N步，无需再判断触发条件

                        # 首先检查是否处于连续攻击状态
                        if self.continuous_attack_counter > 0:
                            # 连续攻击状态：直接攻击，无需判断
                            should_attack_this_step = True
                            self.continuous_attack_counter -= 1

                            if self.debug_attack and self.continuous_attack_counter == 0:
                                print(f"[Attack Burst] Continuous attack ended at step {self.total_steps}")
                        else:
                            # 非连续攻击状态：使用SmartC逻辑判断是否触发
                            should_attack_risk, risk, min_dist, risk_dist = check_risk_trigger(
                                self.env,
                                self.risk_threshold,
                                self.risk_margin
                            )
                            # 只有在风险已经超过基础阈值时，才再看更严格条件
                            if should_attack_risk:
                                # 2) 使用适中的风险阈值（在基础阈值和0.8之间）
                                strict_risk_threshold = min(0.8, self.risk_threshold * 1.5)
                                is_high_risk = risk >= strict_risk_threshold

                                if is_high_risk:
                                    # 3) 使用Statistical模式的Q值差异和动作熵评估（设置严格的阈值）
                                    # 使用配置中的statistical参数，设置严格阈值以降低攻击频率
                                    strict_entropy_scale = max(self.statistical_entropy_scale, 8.0)  # 严格的熵阈值
                                    strict_q_diff_scale = max(self.statistical_q_diff_scale, 3.0)    # 严格的Q差异阈值
                                    strict_variance_scale = max(self.statistical_variance_scale, 3.0) # 严格的方差阈值

                                    should_attack_statistical, action_entropy, q_value_diff, action_variance = check_statistical_critical_state(
                                        obs, model, device, self.env.action_space, self.attack_confidence_threshold,
                                        None,  # 不记录统计信息
                                        strict_entropy_scale, strict_q_diff_scale, strict_variance_scale,
                                        None, 0.0,  # 不使用长期影响预测器
                                        'depth',  # perception_type
                                        None, 0.6, 1.0  # 不使用环境风险感知
                                    )
                                    # 必须同时满足：高风险 AND 统计特征触发
                                    should_attack_this_step = should_attack_statistical

                                    # 如果触发了，开始连续攻击
                                    if should_attack_this_step:
                                        self.continuous_attack_counter = self.continuous_attack_steps

                                        if self.debug_attack:
                                            print(f"[Attack Burst] SmartC Continuous mode: attack triggered at step {self.total_steps}, "
                                                  f"starting {self.continuous_attack_steps} continuous attacks "
                                                  f"(risk: {risk:.3f}, entropy: {action_entropy:.3f}, q_diff: {q_value_diff:.3f}, variance: {action_variance:.3f})")
                                else:
                                    should_attack_this_step = False
                            else:
                                should_attack_this_step = False
                    elif self.attack_trigger_mode == 'step_interval':
                        # 步数间隔攻击模式：每间隔n步攻击m步
                        # 计算当前在完整周期中的位置
                        cycle_length = self.step_interval_n + self.step_attack_m
                        position_in_cycle = self.total_steps % cycle_length
                        # 如果在攻击阶段（前m步），则攻击
                        should_attack_this_step = position_in_cycle < self.step_attack_m
                    elif self.attack_trigger_mode == 'random':
                        # 随机攻击模式：根据配置的概率随机决定是否攻击
                        should_attack_this_step = np.random.random() < self.random_attack_probability
                        if self.debug_attack and should_attack_this_step:
                            print(f"[Attack Trigger] Random mode: attack triggered at step {self.total_steps} "
                                  f"(probability: {self.random_attack_probability:.2f})")
                    elif self.attack_trigger_mode == 'statistical':
                        # 统计方法：基于动作分布统计特征识别关键状态
                        # 使用静态阈值（从配置文件）
                        current_thresholds = {
                            'entropy_scale': self.statistical_entropy_scale,
                            'q_diff_scale': self.statistical_q_diff_scale,
                            'variance_scale': self.statistical_variance_scale
                        }
                        should_attack_this_step, action_entropy, q_value_diff, action_variance = check_statistical_critical_state(
                            obs, model, device, self.env.action_space, self.attack_confidence_threshold, self.statistical_stats,
                            current_thresholds['entropy_scale'], current_thresholds['q_diff_scale'], current_thresholds['variance_scale'],
                            self.impact_predictor, 0.0,  # 暂时使用0，长期影响预测需要episode级别的上下文
                            'depth',  # perception_type
                            self.env,  # env参数用于风险感知
                            self.risk_threshold if hasattr(self, 'risk_threshold') else 0.6,  # 风险阈值
                            self.risk_margin if hasattr(self, 'risk_margin') else 1.0   # 风险边际
                        )
                        if self.debug_attack and should_attack_this_step:
                            print(f"[Attack Trigger] Statistical mode: attack triggered at step {self.total_steps} "
                                  f"(entropy: {action_entropy:.3f}, q_diff: {q_value_diff:.3f}, variance: {action_variance:.3f})")
                    elif self.attack_trigger_mode == 'critical_state':
                        # 兼容旧配置：critical_state映射到statistical
                        current_thresholds = {
                            'entropy_scale': self.statistical_entropy_scale,
                            'q_diff_scale': self.statistical_q_diff_scale,
                            'variance_scale': self.statistical_variance_scale
                        }
                        should_attack_this_step, action_entropy, q_value_diff, action_variance = check_statistical_critical_state(
                            obs, model, device, self.env.action_space, self.attack_confidence_threshold, self.statistical_stats,
                            current_thresholds['entropy_scale'], current_thresholds['q_diff_scale'], current_thresholds['variance_scale'],
                            self.impact_predictor, 0.0  # 暂时使用0，长期影响预测需要episode级别的上下文
                        )
                        if self.debug_attack and should_attack_this_step:
                            print(f"[Attack Trigger] Critical State mode: attack triggered at step {self.total_steps} "
                                  f"(entropy: {action_entropy:.3f}, q_diff: {q_value_diff:.3f}, variance: {action_variance:.3f})")
                    else:
                        should_attack_this_step = True

    
            
            # --------------- 攻击逻辑 ----------------
            if should_attack_this_step:
                try:
                    perception = self.cfg.get('options', 'perception')
                except Exception:
                    perception = 'vector'

                # epsilon - 根据风险连续放大（risk 越大，eps 越大）
                eps = self.attack_epsilon
                if self.attack_trigger_mode in ('smart', 'risk', 'smartc', 'smartc_continuous'):
                    # 获取当前风险值
                    min_dist = float(getattr(self.env, "min_distance_to_obstacles", 1e9))
                    crash_dist = float(getattr(self.env, "crash_distance", 2.0))
                    risk_margin = self.risk_margin
                    risk_distance = max(1e-6, crash_dist + float(risk_margin))
                    risk = (risk_distance - min_dist) / risk_distance
                    risk = float(np.clip(risk, 0.0, 1.0))
                    
                    # 连续映射：risk 从 0→1 时，eps 从 1.0→(1+k)
                    # 例如 k=1.5 时，最大 eps = 2.5 * base_epsilon
                    k = 1.5
                    scale = 1.0 + k * risk
                    eps = self.attack_epsilon * scale

                # 根据 attack_type 选择攻击
                try:
                    if self.attack_type == 'fgsm':
                        obs_adv = fgsm_attack_sac_td3(obs.copy(), model, eps, perception, device)
                    elif self.attack_type == 'targeted':
                        obs_adv = targeted_attack_sac_td3(obs.copy(), model, eps, perception, device)
                    elif self.attack_type == 'pgd':
                        num_iter = self.cfg.getint('options', 'pgd_iter', fallback=20)
                        obs_adv = pgd_attack_sac_td3(obs.copy(), model, eps, perception, device, num_iter)
                    elif self.attack_type == 'random':
                        obs_adv = random_attack(obs.copy(), eps, perception)
                    elif self.attack_type == 'deepfool':
                        # max_iter 可通过 config 新增 deepfool_iter
                        df_iter = self.cfg.getint('options', 'deepfool_iter', fallback=20)
                        obs_adv = deepfool_improved(obs.copy(), model, df_iter, perception, device)
                    elif self.attack_type == 'cw':
                        # cw 参数通过 config 可配置 c, steps, lr
                        c = self.cfg.getfloat('options', 'cw_c', fallback=1e-2)
                        steps = self.cfg.getint('options', 'cw_steps', fallback=80)
                        lr = self.cfg.getfloat('options', 'cw_lr', fallback=1e-2)
                        obs_adv = cw_l2_attack_sac_td3(obs.copy(), model, perception, device, c=c, steps=steps, lr=lr)
                    elif self.attack_type == 'combo':
                        # 组合攻击：FGSM + PGD优化 (更强的组合)
                        # 先用FGSM生成初始扰动
                        obs_adv = fgsm_attack_sac_td3(obs.copy(), model, eps, perception, device)

                        # 再用PGD进行增强优化
                        try:
                            pgd_iter = self.cfg.getint('options', 'combo_pgd_iter', fallback=10)  # 减少迭代次数
                            obs_adv = pgd_attack_sac_td3(obs_adv, model, eps , perception, device, num_iter=pgd_iter)
                        except Exception as e:
                            # 如果PGD失败，使用纯FGSM结果
                            if self.debug_attack:
                                print(f"[Combo Debug] PGD optimization failed, using FGSM only: {e}")
                            pass
                    elif self.attack_type == 'strong_combo':
                        # 强力组合攻击：FGSM -> PGD -> 增强扰动
                        try:
                            # 第一阶段：FGSM生成基础扰动
                            obs_adv = fgsm_attack_sac_td3(obs.copy(), model, eps, perception, device)

                            # 第二阶段：PGD增强
                            pgd_iter = self.cfg.getint('options', 'strong_combo_pgd_iter', fallback=3)
                            obs_adv = pgd_attack_sac_td3(obs_adv, model, eps * 0.8, perception, device, num_iter=pgd_iter)

                            # 第三阶段：如果在危险区域，额外增强
                            if hasattr(self.env, 'min_distance_to_obstacles'):
                                min_dist = getattr(self.env, 'min_distance_to_obstacles', 1e9)
                                crash_dist = getattr(self.env, 'crash_distance', 2.0)
                                if min_dist < crash_dist * 2:
                                    # 在危险区域，进行最终增强
                                    obs_adv = fgsm_attack_sac_td3(obs_adv, model, eps * 0.3, perception, device)
                                    if self.debug_attack:
                                        print(f"[Strong Combo] Applied danger zone enhancement, distance: {min_dist:.2f}")

                        except Exception as e:
                            # 回退到标准FGSM
                            obs_adv = fgsm_attack_sac_td3(obs.copy(), model, eps, perception, device)
                            if self.debug_attack:
                                print(f"[Strong Combo] Failed, fallback to FGSM: {e}")
                    elif self.attack_type == 'bim':
                        num_iter = self.cfg.getint('options', 'bim_iter', fallback=20)
                        obs_adv = bim_attack_sac_td3(obs.copy(), model, eps, perception, device, num_iter=num_iter)
                    elif self.attack_type == 'mim':
                        num_iter = self.cfg.getint('options', 'mim_iter', fallback=20)
                        momentum = self.cfg.getfloat('options', 'mim_momentum', fallback=1.0)
                        obs_adv = mim_attack_sac_td3(obs.copy(), model, eps, perception, device, num_iter=num_iter,
                                                     decay=momentum)
                    elif self.attack_type == 'crash':
                        # 专门设计来让模型撞墙的攻击
                        obs_adv = crash_inducing_attack_sac_td3(obs.copy(), model, eps, perception, device, env=self.env)
                    elif self.attack_type == 'smart_q_entropy':
                        # Smart Q-Entropy融合攻击：动态调整攻击强度
                        fusion_score = getattr(self, '_current_fusion_score', 0.0)
                        risk_value = getattr(self, '_current_risk_value', 0.0)
                        obs_adv = smart_q_entropy_attack_sac_td3(
                            obs.copy(), model, eps, perception, device,
                            fusion_score=fusion_score, risk=risk_value, env=self.env
                        )
                    else:
                        raise ValueError(f"未知的攻击类型: {self.attack_type}")

                except Exception as e:
                    # 只在debug模式下打印攻击失败信息，避免每步都打印
                    if self.debug_attack:
                        print(f"{self.attack_type} attack failed (fallback to clean obs). Error:", e)
                    obs_adv = obs

                # 调试信息：检查攻击是否真的改变了观测
                if self.debug_attack and should_attack_this_step:
                    if np.array_equal(obs, obs_adv):
                        print(f"[Attack Debug] WARNING: Attack generated identical observation! Attack may not be working.")
                        print(f"  attack_type: {self.attack_type}, epsilon: {eps}, perception: {perception}")
                        print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}, range: [{obs.min():.6f}, {obs.max():.6f}]")
                    else:
                        obs_diff = np.abs(obs_adv - obs)
                        print(f"[Attack Debug] Attack applied - max_diff: {obs_diff.max():.6f}, mean_diff: {obs_diff.mean():.6f}")

                # 攻击成功执行，增加计数
                self.attack_count += 1
                obs_to_use = obs_adv
            else:
                obs_to_use = obs



            # 使用 obs_to_use 进行预测
            unscaled_action, _ = model.predict(obs_to_use, deterministic=True)
            # ==================== end =====================
            # unscaled_action, _ = model.predict(obs, deterministic=True)
            time_step += 1

            new_obs, reward, done, info, = self.env.step(unscaled_action)
            pose = self.env.dynamic_model.get_position()
            traj_list.append(pose)
            action_list.append(unscaled_action)
            state_raw_list.append(self.env.dynamic_model.state_raw)
            obs_list.append(obs)

            obs = new_obs
            reward_sum[-1] += reward

            # Episode内调节器更新 - 已禁用自动调节
            # if self.attack_trigger_mode in ('statistical', 'critical_state'):
            #     self.episode_tuner.update_step(should_attack_this_step, reward)

            # 更新长期影响预测器的历史记录
            if self.attack_trigger_mode in ('statistical', 'critical_state', 'smart_q', 'smart_q_entropy'):
                # 提取状态特征用于历史记录
                if obs.ndim == 3:  # 图像观测
                    state_features = np.array([
                        np.mean(obs), np.std(obs), np.max(obs), np.min(obs), np.var(obs)
                    ])
                else:  # 向量观测
                    state_features = obs.flatten()[:10]  # 取前10个特征

                # 标准化特征
                state_features = (state_features - np.mean(state_features)) / (np.std(state_features) + 1e-8)

                # 更新预测器历史
                if self.impact_predictor is not None:
                    self.impact_predictor.update_history(reward, should_attack_this_step, state_features)

            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                maybe_is_crash = info.get('is_crash')
                
                # 打印攻击统计信息
                attack_rate = (self.attack_count / self.total_steps * 100) if self.total_steps > 0 else 0

                # 收集当前episode的统计信息
                all_episode_rewards.append(reward_sum[-1])
                all_episode_steps.append(info.get('step_num', 0))
                if self.enable_attack and self.attack_trigger_mode in ('confidence', 'risk', 'smart', 'smartc', 'smartc_continuous', 'always', 'step_interval', 'random', 'statistical', 'critical_state', 'smart_q', 'smart_q_entropy'):
                    all_attack_rates.append(attack_rate)
                
                # 打印置信度检测诊断信息
                confidence_info = ''
                if self.enable_attack and self.attack_trigger_mode == 'confidence' and self.total_steps > 0:
                    stats = self.confidence_stats
                    avg_max_dist = np.mean(stats['max_distances']) if stats['max_distances'] else 0
                    avg_policy_conf = np.mean(stats['policy_confidences']) if stats['policy_confidences'] else 0
                    confidence_info = f' | max_dist_avg: {avg_max_dist:.3f}, policy_conf_avg: {avg_policy_conf:.3f}'
                    confidence_info += f' | M1:{stats["method1_triggers"]} M2:{stats["method2_triggers"]} M2_fail:{stats["method2_failed"]}'
                    
                    # 收集置信度统计信息
                    if stats['max_distances']:
                        all_max_distances.extend(stats['max_distances'])
                    if stats['policy_confidences']:
                        all_policy_confidences.extend(stats['policy_confidences'])
                    all_method1_triggers.append(stats['method1_triggers'])
                    all_method2_triggers.append(stats['method2_triggers'])
                    all_method2_failures.append(stats['method2_failed'])
                elif self.enable_attack and self.attack_trigger_mode in ('risk', 'smart', 'smartc', 'smartc_continuous'):
                    confidence_info = f' | risk_attack_rate: {attack_rate:.1f}%'
                elif self.enable_attack and self.attack_trigger_mode in ('statistical', 'critical_state', 'smart_q') and self.total_steps > 0:
                    stats = self.statistical_stats
                    avg_entropy = np.mean(stats['action_entropies']) if stats['action_entropies'] else 0
                    avg_q_diff = np.mean(stats['q_value_diffs']) if stats['q_value_diffs'] else 0
                    avg_variance = np.mean(stats['action_variances']) if stats['action_variances'] else 0
                    confidence_info = f' | entropy_avg: {avg_entropy:.3f}, q_diff_avg: {avg_q_diff:.3f}, var_avg: {avg_variance:.3f}'
                    confidence_info += f' | E:{stats["entropy_triggers"]} Q:{stats["q_diff_triggers"]} V:{stats["variance_triggers"]}'

                    # 收集统计方法的统计数据
                    all_entropy_avgs.append(avg_entropy)
                    all_q_diff_avgs.append(avg_q_diff)
                    all_variance_avgs.append(avg_variance)
                    all_entropy_triggers.append(stats['entropy_triggers'])
                    all_q_diff_triggers.append(stats['q_diff_triggers'])
                    all_variance_triggers.append(stats['variance_triggers'])
                elif self.enable_attack and self.attack_trigger_mode == 'smart_q_entropy' and self.total_steps > 0:
                    # Smart Q-Entropy融合攻击统计
                    sqe_stats = self.smart_q_entropy_stats
                    avg_fusion = np.mean(sqe_stats['fusion_scores']) if sqe_stats['fusion_scores'] else 0
                    avg_risk = np.mean(sqe_stats['risks']) if sqe_stats['risks'] else 0
                    avg_entropy = np.mean(sqe_stats['entropies']) if sqe_stats['entropies'] else 0
                    avg_q_diff = np.mean(sqe_stats['q_diffs']) if sqe_stats['q_diffs'] else 0
                    avg_q_std = np.mean(sqe_stats['q_stds']) if sqe_stats['q_stds'] else 0
                    confidence_info = f' | fusion_avg: {avg_fusion:.3f}, risk_avg: {avg_risk:.3f}'
                    confidence_info += f' | entropy_avg: {avg_entropy:.3f}, q_diff_avg: {avg_q_diff:.3f}, q_std_avg: {avg_q_std:.3f}'
                    confidence_info += f' | triggers: {sqe_stats["triggers"]}'

                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success,
                      f'attack_rate: {attack_rate:.1f}%' if self.enable_attack and self.attack_trigger_mode in ('confidence', 'risk', 'smart', 'always', 'random', 'statistical', 'critical_state', 'smart_q', 'smart_q_entropy') else '',
                      confidence_info)

                # 更新自适应阈值调节器 - 已禁用自动调节
                    # if self.attack_trigger_mode in ('statistical', 'critical_state'):
                    #     # Episode结束时进行最终调节
                    #     episode_reward = reward_sum[-1]
                    #     was_successful = maybe_is_success
                    #     self.episode_tuner.end_episode(episode_reward, was_successful)
                    #     print(f"[Adaptive Thresholds] entropy_scale: {current_thresholds['entropy_scale']:.2f}, "
                    #           f"q_diff_scale: {current_thresholds['q_diff_scale']:.2f}, "
                    #           f"variance_scale: {current_thresholds['variance_scale']:.2f}")

                episode_successes.append(float(maybe_is_success))
                episode_crashes.append(float(maybe_is_crash))
                reward_sum = np.append(reward_sum, .0)
                
                # 重置攻击统计（每个episode）
                if self.enable_attack and self.attack_trigger_mode in ('confidence', 'risk', 'smart', 'smartc', 'smartc_continuous', 'always', 'step_interval', 'random', 'statistical', 'critical_state', 'smart_q', 'smart_q_entropy'):
                    self.attack_count = 0
                    self.total_steps = 0
                    self.continuous_attack_counter = 0  # 重置连续攻击计数器
                    # 重置置信度统计
                    self.confidence_stats = {
                        'method1_triggers': 0,
                        'method2_triggers': 0,
                        'method2_failed': 0,
                        'max_distances': [],
                        'policy_confidences': []
                    }
                    # 重置统计方法统计
                    self.statistical_stats = {
                        'entropy_triggers': 0,
                        'q_diff_triggers': 0,
                        'variance_triggers': 0,
                        'action_entropies': [],
                        'q_value_diffs': [],
                        'action_variances': []
                    }
                    # 重置Smart Q-Entropy统计
                    self.smart_q_entropy_stats = {
                        'fusion_scores': [],
                        'risks': [],
                        'entropies': [],
                        'q_diffs': [],
                        'q_stds': [],
                        'triggers': 0
                    }
                    model._confidence_stats = self.confidence_stats
                
                obs = self.env.reset()
                if info.get('is_success'):
                    traj_list.append(1)
                    action_list.append(1)
                    step_num_list.append(info.get('step_num'))
                elif info.get('is_crash'):
                    traj_list.append(2)
                    action_list.append(2)
                else:
                    traj_list.append(3)
                    action_list.append(3)
                # traj_list.append(info)
                traj_list_all.append(traj_list)
                action_list_all.append(action_list)
                state_list_all.append(state_raw_list)
                obs_list_all.append(obs_list)
                traj_list = []
                action_list = []
                state_raw_list = []
                obs_list = []

        # save trajectory data in eval folder
        eval_folder = self.eval_path + '/eval_{}_{}_{}'.format(self.eval_ep_num, self.eval_env, self.eval_dynamics)
        os.makedirs(eval_folder, exist_ok=True)
        np.save(eval_folder + '/traj_eval',
                np.array(traj_list_all, dtype=object))
        np.save(eval_folder + '/action_eval',
                np.array(action_list_all, dtype=object))
        np.save(eval_folder + '/state_eval',
                np.array(state_list_all, dtype=object))
        np.save(eval_folder + '/obs_eval',
                np.array(obs_list_all, dtype=object))

        # 计算综合平均数据
        avg_reward = np.mean(all_episode_rewards) if all_episode_rewards else reward_sum[:self.eval_ep_num].mean()
        success_rate = np.mean(episode_successes) if episode_successes else 0.0
        crash_rate = np.mean(episode_crashes) if episode_crashes else 0.0
        avg_success_steps = np.mean(step_num_list) if step_num_list else 0.0
        avg_episode_steps = np.mean(all_episode_steps) if all_episode_steps else 0.0
        
        # 计算平均攻击率（如果启用了攻击）
        avg_attack_rate = 0.0
        if self.enable_attack:
            if self.attack_trigger_mode in ('confidence', 'risk', 'smart', 'smartc', 'smartc_continuous', 'step_interval', 'random', 'statistical', 'critical_state', 'smart_q', 'smart_q_entropy') and all_attack_rates:
                avg_attack_rate = np.mean(all_attack_rates)
            elif self.attack_trigger_mode == 'always':
                # always 模式视为持续攻击
                avg_attack_rate = 100.0
        
        # 打印基础统计信息
        print('\n' + '='*80)
        print('EVALUATION SUMMARY - All Episodes Statistics')
        print('='*80)
        print(f'Total Episodes: {self.eval_ep_num}')
        print(f'Average Episode Reward: {avg_reward:.4f}')
        print(f'Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)')
        print(f'Crash Rate: {crash_rate:.4f} ({crash_rate*100:.2f}%)')
        print(f'Average Episode Steps: {avg_episode_steps:.2f}')
        if step_num_list:
            print(f'Average Success Episode Steps: {avg_success_steps:.2f}')
            print(f'Success Episodes: {len(step_num_list)}/{self.eval_ep_num}')
        # 打印攻击相关统计（如果启用了攻击）
        if self.enable_attack:
            print(f'Average Attack Rate: {avg_attack_rate:.2f}%')
            print('\n' + '-'*80)
            print('ATTACK STATISTICS')
            print('-'*80)
            # 显示攻击模式名称和统计信息
            if self.attack_trigger_mode == 'always':
                print(' Always Attack')
            elif self.attack_trigger_mode == 'step_interval':
                print(f'Step Interval Attack')
                expected_rate = (self.step_attack_m / (self.step_interval_n + self.step_attack_m) * 100)
                print(f'Expected Attack Rate: {expected_rate:.1f}%')
            elif self.attack_trigger_mode == 'confidence':
                print('Confidence-based Attack')
            elif self.attack_trigger_mode == 'risk':
                print('Risk-based Attack')
            elif self.attack_trigger_mode == 'smart':
                print('Smart Attack (Risk + Confidence)')
            elif self.attack_trigger_mode == 'smartc':
                print('SmartC Attack (Risk + Statistical Features)')
                entropy_scale = max(self.statistical_entropy_scale, 8.0)
                q_diff_scale = max(self.statistical_q_diff_scale, 3.0)
                variance_scale = max(self.statistical_variance_scale, 3.0)
                print(f'Entropy Scale: {entropy_scale:.1f}')
                print(f'Q-Diff Scale: {q_diff_scale:.1f}')
                print(f'Variance Scale: {variance_scale:.1f}')
            elif self.attack_trigger_mode == 'smartc_continuous':
                print(f'SmartC Continuous Attack (Risk + Statistical Features + {self.continuous_attack_steps} Burst Steps)')
                entropy_scale = max(self.statistical_entropy_scale, 8.0)
                q_diff_scale = max(self.statistical_q_diff_scale, 3.0)
                variance_scale = max(self.statistical_variance_scale, 3.0)
                print(f'Entropy Scale: {entropy_scale:.1f}')
                print(f'Q-Diff Scale: {q_diff_scale:.1f}')
                print(f'Variance Scale: {variance_scale:.1f}')
                print(f'Continuous Steps: {self.continuous_attack_steps}')
            elif self.attack_trigger_mode == 'smart_q':
                print('SmartQ Attack (Risk + Q-Value Difference)')
                q_diff_scale = self.statistical_q_diff_scale
                print(f'Q-Diff Scale: {q_diff_scale:.1f}')
            elif self.attack_trigger_mode == 'smart_q_entropy':
                print('Smart Q-Entropy Fusion Attack (Risk + Action Entropy + Q-Value)')
                print(f'Fusion Threshold: {self.smart_q_entropy_threshold:.2f}')
                print(f'Risk Weight: {self.smart_q_entropy_risk_weight:.2f}')
                print(f'Entropy Weight: {self.smart_q_entropy_entropy_weight:.2f}')
                print(f'Q-Value Weight: {self.smart_q_entropy_q_weight:.2f}')
                # 打印Smart Q-Entropy的统计数据
                sqe_stats = self.smart_q_entropy_stats
                if sqe_stats['fusion_scores']:
                    print(f'Average Fusion Score: {np.mean(sqe_stats["fusion_scores"]):.4f}')
                    print(f'Average Risk: {np.mean(sqe_stats["risks"]):.4f}')
                    print(f'Average Entropy: {np.mean(sqe_stats["entropies"]):.4f}')
                    print(f'Average Q-Diff: {np.mean(sqe_stats["q_diffs"]):.4f}')
                    print(f'Average Q-Std: {np.mean(sqe_stats["q_stds"]):.4f}')
                    print(f'Total Triggers: {sqe_stats["triggers"]}')
            elif self.attack_trigger_mode == 'random':
                print(f'Random Attack (Probability: {self.random_attack_probability:.2f})')
            elif self.attack_trigger_mode in ('statistical', 'critical_state'):
                print('Statistical Critical State Attack')
                print(f'Entropy Scale: {self.statistical_entropy_scale:.1f}')
                print(f'Q-Diff Scale: {self.statistical_q_diff_scale:.1f}')
                print(f'Variance Scale: {self.statistical_variance_scale:.1f}')

            # 显示攻击统计信息（如果有数据）
            if self.attack_trigger_mode in ('confidence', 'risk', 'smart', 'smartc', 'smartc_continuous', 'step_interval', 'random', 'statistical', 'critical_state', 'smart_q', 'smart_q_entropy') and all_attack_rates:
                print(f'Average Attack Rate: {avg_attack_rate:.2f}%')
                print(f'Attack Rate Range: [{np.min(all_attack_rates):.2f}%, {np.max(all_attack_rates):.2f}%]')

                # confidence模式的额外统计
                if self.attack_trigger_mode == 'confidence':
                    if all_max_distances:
                        print(f'Average Max Distance (Method 1): {np.mean(all_max_distances):.4f}')
                        print(f'Max Distance Range: [{np.min(all_max_distances):.4f}, {np.max(all_max_distances):.4f}]')

                    if all_policy_confidences:
                        print(f'Average Policy Confidence (Method 2): {np.mean(all_policy_confidences):.4f}')
                        print(f'Policy Confidence Range: [{np.min(all_policy_confidences):.4f}, {np.max(all_policy_confidences):.4f}]')

                    if all_method1_triggers:
                        total_m1 = np.sum(all_method1_triggers)
                        total_m2 = np.sum(all_method2_triggers)
                        total_m2_fail = np.sum(all_method2_failures)
                        print(f'Method 1 Triggers (Total): {total_m1}')
                        print(f'Method 2 Triggers (Total): {total_m2}')
                        print(f'Method 2 Failures (Total): {total_m2_fail}')

                # statistical模式的额外统计
                elif self.attack_trigger_mode in ('statistical', 'critical_state'):
                    if all_entropy_avgs:
                        print(f'Average Entropy: {np.mean(all_entropy_avgs):.4f}')
                        print(f'Entropy Range: [{np.min(all_entropy_avgs):.4f}, {np.max(all_entropy_avgs):.4f}]')

                    if all_q_diff_avgs:
                        print(f'Average Q-Value Difference: {np.mean(all_q_diff_avgs):.4f}')
                        print(f'Q-Diff Range: [{np.min(all_q_diff_avgs):.4f}, {np.max(all_q_diff_avgs):.4f}]')

                    if all_variance_avgs:
                        print(f'Average Variance: {np.mean(all_variance_avgs):.6f}')
                        print(f'Variance Range: [{np.min(all_variance_avgs):.6f}, {np.max(all_variance_avgs):.6f}]')

                    if all_entropy_triggers:
                        total_entropy = np.sum(all_entropy_triggers)
                        total_q_diff = np.sum(all_q_diff_triggers)
                        total_variance = np.sum(all_variance_triggers)
                        print(f'Entropy Triggers (Total): {total_entropy}')
                        print(f'Q-Diff Triggers (Total): {total_q_diff}')
                        print(f'Variance Triggers (Total): {total_variance}')

                        # 计算每个触发器的平均触发率
                        avg_entropy_rate = total_entropy / len(all_entropy_triggers) if all_entropy_triggers else 0
                        avg_q_diff_rate = total_q_diff / len(all_q_diff_triggers) if all_q_diff_triggers else 0
                        avg_variance_rate = total_variance / len(all_variance_triggers) if all_variance_triggers else 0
                        print(f'Average Entropy Triggers per Episode: {avg_entropy_rate:.1f}')
                        print(f'Average Q-Diff Triggers per Episode: {avg_q_diff_rate:.1f}')
                        print(f'Average Variance Triggers per Episode: {avg_variance_rate:.1f}')

            # 显示攻击方法和参数
            print(f'Attack Type: {self.attack_type}')
            print(f'Attack Epsilon: {self.attack_epsilon}')
            if self.attack_trigger_mode == 'confidence':
                print(f'Confidence Threshold: {self.attack_confidence_threshold}')
            if self.attack_trigger_mode == 'risk':
                print(f'Risk Threshold: {self.risk_threshold}, Risk Margin: {self.risk_margin}')
            if self.attack_trigger_mode == 'step_interval':
                print(f'Step Interval Parameters: n={self.step_interval_n}, m={self.step_attack_m}')
            if self.attack_trigger_mode == 'random':
                print(f'Random Attack Probability: {self.random_attack_probability:.2f}')
        
        # 打印奖励分布统计
        print('\n' + '-'*80)
        print('REWARD DISTRIBUTION')
        print('-'*80)
        if all_episode_rewards:
            print(f'Min Reward: {np.min(all_episode_rewards):.4f}')
            print(f'Max Reward: {np.max(all_episode_rewards):.4f}')
            print(f'Median Reward: {np.median(all_episode_rewards):.4f}')
            print(f'Std Reward: {np.std(all_episode_rewards):.4f}')
            # 统计正奖励和负奖励的episode数
            positive_rewards = sum(1 for r in all_episode_rewards if r > 0)
            negative_rewards = sum(1 for r in all_episode_rewards if r < 0)
            print(f'Episodes with Positive Reward: {positive_rewards}/{self.eval_ep_num} ({positive_rewards/self.eval_ep_num*100:.2f}%)')
            print(f'Episodes with Negative Reward: {negative_rewards}/{self.eval_ep_num} ({negative_rewards/self.eval_ep_num*100:.2f}%)')
        
        print('='*80 + '\n')

        # 保存结果（保持原有格式以兼容其他代码，添加攻击率）
        results = [avg_reward, success_rate, crash_rate, avg_success_steps]
        if self.enable_attack:
            results.append(avg_attack_rate)
        print(f'Results Array: {results}')
        np.save(eval_folder + '/results', np.array(results))

        return results


    def run_rule_policy(self):
        obs = self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        while episode_num < self.eval_ep_num:
            unscaled_action = rule_based_policy(obs)
            time_step += 1
            new_obs, reward, done, info, = self.env.step(unscaled_action)
            reward_sum[-1] += reward

            obs = new_obs
            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success)
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()


def main():
    eval_path = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs_new\Trees\2022_12_02_21_46_SimpleMultirotor_mlp_SAC'
    config_file = eval_path + '/config/config.ini'
    model_file = eval_path + '/models/model_sb3.zip'

    eval_ep_num = 50
    evaluate_thread = EvaluateThread(eval_path, config_file, model_file,
                                     eval_ep_num)
    evaluate_thread.run()


def run_eval_multi():
    # run evaluation for multi models
    eval_logs_name = 'Maze'
    eval_logs_path = 'logs_eval/' + eval_logs_name
    eval_ep_num = 50
    eval_env_name = 'NH_center'  # 1-Trees 2-SimpleAvoid 3-NH_center
    eval_dynamic_name = 'SimpleMultirotor'  # 1-SimpleMultirotor or Multirotor

    model_list = []
    for train_name in os.listdir(eval_logs_path):
        for repeat_name in os.listdir(eval_logs_path + '/' + train_name):
            model_path = eval_logs_path + '/' + train_name + '/' + repeat_name
            model_list.append(model_path)
            # print(model_path)

    # evaluate model according to model path
    eval_num = len(model_list)
    results_list = []

    for i in tqdm(range(eval_num)):
        eval_path = model_list[i]
        config_file = eval_path + '/config/config.ini'
        model_file = eval_path + '/models/model_sb3.zip'

        print(i, eval_path)
        evaluate_thread = EvaluateThread(eval_path, config_file, model_file, eval_ep_num, eval_env_name,
                                         eval_dynamic_name)
        results = evaluate_thread.run()
        results_list.append(results)

        del evaluate_thread

    # save all results in a numpy file
    print(results_list)
    np.save('logs_eval/results/eval_{}_{}_{}_{}'.format(eval_ep_num, eval_logs_name, eval_env_name, eval_dynamic_name),
            np.array(results_list))


class LongTermImpactPredictor:
    """
    长期影响预测器 (基于论文中的LSTM方法简化实现)
    预测序列攻击的长期累积效果
    """

    def __init__(self, sequence_length=10, hidden_dim=32, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # 简化的序列预测器 (不使用真正的LSTM，改用简单的统计模型)
        self.reward_history = []
        self.attack_history = []
        self.state_history = []

        # 长期影响权重 (根据经验衰减)
        self.decay_factor = 0.9

    def update_history(self, reward, was_attacked, state_features):
        """
        更新历史记录
        """
        self.reward_history.append(reward)
        self.attack_history.append(was_attacked)
        self.state_history.append(state_features)

        # 保持固定长度
        if len(self.reward_history) > self.sequence_length:
            self.reward_history.pop(0)
            self.attack_history.pop(0)
            self.state_history.pop(0)

    def predict_long_term_impact(self, current_reward, state_features, will_attack=False):
        """
        预测长期影响
        返回: 预测的累积影响分数 (负数表示不利，正数表示有利)
        """
        if len(self.reward_history) < 3:
            return 0.0  # 历史不足，返回中性预测

        # 计算历史攻击成功率
        attack_rate = sum(self.attack_history) / len(self.attack_history)

        # 计算历史奖励趋势
        recent_rewards = self.reward_history[-5:] if len(self.reward_history) >= 5 else self.reward_history
        reward_trend = np.mean(recent_rewards) - np.mean(self.reward_history[:len(recent_rewards)])

        # 计算状态相似性 (简化的余弦相似性)
        if len(self.state_history) > 0:
            similarities = []
            for hist_state in self.state_history[-3:]:
                dot_product = np.dot(state_features, hist_state)
                norm_product = (np.linalg.norm(state_features) * np.linalg.norm(hist_state) + 1e-8)
                similarities.append(dot_product / norm_product)
            state_similarity = np.mean(similarities)

            # 如果状态相似且之前攻击有效，则预测攻击有利
            if state_similarity > 0.8 and attack_rate > 0.6:
                long_term_impact = 0.3  # 预测有利
            elif state_similarity > 0.8 and attack_rate < 0.3:
                long_term_impact = -0.3  # 预测不利
            else:
                long_term_impact = 0.0  # 中性
        else:
            long_term_impact = 0.0

        # 结合当前奖励预测
        if will_attack and current_reward < -50:  # 当前奖励很差，攻击可能恶化
            long_term_impact -= 0.2
        elif not will_attack and current_reward > 10:  # 当前表现好，不攻击可能更好
            long_term_impact += 0.1

        return long_term_impact


def predict_attack_impact(obs, model, device, action_space, predictor, current_reward):
    """
    预测攻击的长期影响

    参数:
        obs: 当前观测
        model: 模型
        device: 计算设备
        action_space: 动作空间
        predictor: 长期影响预测器
        current_reward: 当前奖励

    返回:
        attack_impact_score: 攻击影响分数 (-1到1, 负数表示不利)
    """
    try:
        # 提取状态特征 (简化为观测的统计特征)
        if obs.ndim == 3:  # 图像观测
            state_features = np.array([
                np.mean(obs),      # 均值
                np.std(obs),       # 标准差
                np.max(obs),       # 最大值
                np.min(obs),       # 最小值
                np.var(obs)        # 方差
            ])
        else:  # 向量观测
            state_features = obs.flatten()[:10]  # 取前10个特征

        # 标准化特征
        state_features = (state_features - np.mean(state_features)) / (np.std(state_features) + 1e-8)

        # 预测攻击和不攻击的长期影响
        impact_with_attack = predictor.predict_long_term_impact(current_reward, state_features, will_attack=True)
        impact_without_attack = predictor.predict_long_term_impact(current_reward, state_features, will_attack=False)

        # 计算相对影响
        attack_impact_score = impact_with_attack - impact_without_attack

        return attack_impact_score

    except Exception as e:
        print(f"[Impact Prediction Error] {e}")
        return 0.0


# EpisodeAdaptiveTuner类已删除 - 自动调节阈值功能已禁用
# ==================== Smart Q-Entropy Fusion Attack ====================

def compute_action_entropy(actions_array, action_space):
    """
    计算动作分布的熵

    参数:
        actions_array: 采样的动作数组 (n_samples, action_dim)
        action_space: 动作空间

    返回:
        action_entropy: 动作熵值
        normalized_entropy: 归一化的熵值 (相对于最大可能熵)
    """
    try:
        n_samples = len(actions_array)
        if n_samples < 2:
            return 0.0, 0.0

        action_dim = actions_array.shape[1]
        action_range = action_space.high - action_space.low

        # 核密度估计 (简化版)
        # 将动作离散化为bin来计算熵
        n_bins = 10
        bin_width = (action_range) / n_bins
        bin_width = np.maximum(bin_width, 1e-6)  # 避免除零

        # 计算每个维度的直方图
        entropies = []
        for dim in range(action_dim):
            hist, _ = np.histogram(actions_array[:, dim], bins=n_bins,
                                   range=(action_space.low[dim], action_space.high[dim]))
            hist = hist / (n_samples + 1e-8)  # 归一化

            # 计算熵: H = -sum(p * log(p))
            hist = np.maximum(hist, 1e-8)  # 避免log(0)
            dim_entropy = -np.sum(hist * np.log(hist))
            entropies.append(dim_entropy)

        mean_entropy = np.mean(entropies)

        # 归一化熵 (相对于均匀分布的最大熵)
        max_entropy = np.log(n_bins)  # 均匀分布的熵
        normalized_entropy = mean_entropy / (max_entropy + 1e-8)

        return mean_entropy, normalized_entropy

    except Exception as e:
        print(f"[Entropy Calculation Error] {e}")
        return 0.0, 0.0


def compute_q_value_comprehensive(model, obs, action, device, perception_type, silent=True):
    """
    综合计算Q值相关信息

    参数:
        model: SAC/TD3模型
        obs: 当前观测
        action: 动作
        device: 计算设备
        perception_type: 观测类型
        silent: 是否静默模式（True=不打印错误信息）

    返回:
        q_mean: Q值均值
        q_std: Q值标准差
        q_diff: 双Q网络的差异 (对于TD3/SAC)
        q_min: 最小Q值
        q_max: 最大Q值
    """
    try:
        obs_t = _to_torch(obs, device, perception_type)
        action_t = th.from_numpy(np.asarray(action)).float().to(device)

        if action_t.dim() == 1:
            action_t = action_t.unsqueeze(0)

        q_values = []

        # ==================== 优先尝试使用policy内置方法 ====================
        # 方法1: 尝试使用policy._q_value_forward (SB3内置方法，最可靠)
        if not q_values and hasattr(model.policy, '_q_value_forward'):
            try:
                with th.no_grad():
                    q_vals = model.policy._q_value_forward(obs_t, action_t)
                    if isinstance(q_vals, tuple) and len(q_vals) >= 2:
                        q_values = [q_vals[0].mean().item(), q_vals[1].mean().item()]
                    elif th.is_tensor(q_vals):
                        q_values = [q_vals.mean().item()]
            except (RuntimeError, AttributeError, TypeError):
                pass  # 静默处理

        # ==================== 备用方法：手动构建critic输入 ====================
        if not q_values:
            # 尝试获取critic网络
            critic = None
            if hasattr(model, 'critic') and model.critic is not None:
                critic = model.critic
            elif hasattr(model, 'policy') and hasattr(model.policy, 'critic'):
                critic = model.policy.critic

            if critic is not None:
                # 尝试方法2a: 使用critic的q_networks
                if hasattr(critic, 'q_networks') and len(critic.q_networks) >= 2:
                    try:
                        with th.no_grad():
                            # 尝试使用policy的特征提取器
                            if hasattr(model.policy, 'features') and hasattr(model.policy.features, 'forward'):
                                # CNN_GAP类型网络
                                features = model.policy.features(obs_t)
                                features = features.reshape(features.shape[0], -1)
                            else:
                                # 标准方法：展平观测
                                features = obs_t.reshape(obs_t.shape[0], -1)

                            # 连接特征和动作
                            action_flat = action_t.reshape(action_t.shape[0], -1)
                            input_combined = th.cat([features, action_flat], dim=1)

                            # 展平为2D tensor (batch, features)
                            input_combined = input_combined.view(input_combined.size(0), -1)

                            q1 = critic.q_networks[0](input_combined)
                            q2 = critic.q_networks[1](input_combined)
                            q_values = [q1.mean().item(), q2.mean().item()]
                    except (RuntimeError, AttributeError, TypeError):
                        pass  # 静默处理

                # 尝试方法2b: 使用TD3风格的q1/q2
                elif hasattr(critic, 'q1') and hasattr(critic, 'q2'):
                    try:
                        with th.no_grad():
                            q1 = critic.q1(obs_t, action_t)
                            q2 = critic.q2(obs_t, action_t)
                            q_values = [q1.mean().item(), q2.mean().item()]
                    except (RuntimeError, AttributeError, TypeError):
                        pass  # 静默处理

        # ==================== 最后的备用方法：使用动作近似 ====================
        if not q_values:
            # 如果无法获取真实Q值，使用动作方差近似
            # 高动作值通常意味着高Q值（对于连续动作空间）
            action_magnitude = th.abs(action_t).mean().item()
            # 归一化到合理范围
            normalized_q = min(action_magnitude / 2.0, 1.0)
            # 返回归一化的Q值和0差异（表示不确定）
            return normalized_q, 0.1, 0.1, normalized_q - 0.1, normalized_q + 0.1

        # 计算Q值统计
        if q_values:
            q_mean = np.mean(q_values)
            q_std = np.std(q_values)
            q_diff = np.abs(q_values[0] - q_values[1]) if len(q_values) >= 2 else 0.0
            q_min = np.min(q_values)
            q_max = np.max(q_values)
            return q_mean, q_std, q_diff, q_min, q_max
        else:
            # 无法获取Q值时返回默认值
            return 0.0, 0.0, 0.0, 0.0, 0.0

    except Exception as e:
        print(f"[Q-Value Calculation Error] {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0


def compute_fusion_score(risk, action_entropy, q_diff, q_std,
                         risk_weight=0.3, entropy_weight=0.35, q_weight=0.35):
    """
    计算融合分数

    融合公式: fusion_score = risk_weight * risk + entropy_weight * entropy + q_weight * q_normalized

    参数:
        risk: 风险值 (0-1)
        action_entropy: 动作熵 (0-1, 归一化)
        q_diff: Q值差异
        q_std: Q值标准差
        risk_weight: 风险权重
        entropy_weight: 熵权重
        q_weight: Q值权重

    返回:
        fusion_score: 融合分数
        components: 各组成部分的字典
    """
    # 归一化Q值差异 (基于经验范围)
    q_normalized = np.clip(q_diff / (q_std + 1.0 + 1e-6), 0.0, 1.0)
    q_combined = (q_normalized + np.clip(q_std / 10.0, 0.0, 1.0)) / 2.0

    # 计算各部分贡献
    risk_contribution = risk_weight * risk
    entropy_contribution = entropy_weight * action_entropy
    q_contribution = q_weight * q_combined

    fusion_score = risk_contribution + entropy_contribution + q_contribution

    components = {
        'risk': risk,
        'action_entropy': action_entropy,
        'q_diff': q_diff,
        'q_std': q_std,
        'q_normalized': q_normalized,
        'risk_contribution': risk_contribution,
        'entropy_contribution': entropy_contribution,
        'q_contribution': q_contribution
    }

    return fusion_score, components


def check_smart_q_entropy_trigger(env, model, obs, device, action_space,
                                    perception_type, fusion_threshold,
                                    risk_threshold=0.5, risk_margin=3.0,
                                    risk_weight=0.3, entropy_weight=0.35, q_weight=0.35,
                                    stats=None, debug=False):
    """
    Smart Q-Entropy 融合攻击触发条件检查

    融合三个因素:
    1. 环境风险 (risk): 基于与障碍物的距离
    2. 动作熵 (action_entropy): 基于动作分布的不确定性
    3. Q值特征 (Q-value): 基于critic评估的不一致性

    参数:
        env: 环境对象
        model: SAC/TD3模型
        obs: 当前观测
        device: 计算设备
        action_space: 动作空间
        perception_type: 观测类型
        fusion_threshold: 融合分数阈值 (0-1)
        risk_threshold: 风险阈值
        risk_margin: 风险边距
        risk_weight: 风险权重
        entropy_weight: 熵权重
        q_weight: Q值权重
        stats: 统计信息字典
        debug: 调试模式

    返回:
        should_attack: 是否应该攻击
        fusion_score: 融合分数
        components: 各组成部分
    """
    try:
        # 1. 计算风险值
        min_dist = float(getattr(env, "min_distance_to_obstacles", 1e9))
        crash_dist = float(getattr(env, "crash_distance", 2.0))
        risk_distance = max(1e-6, crash_dist + float(risk_margin))
        risk = (risk_distance - min_dist) / risk_distance
        risk = float(np.clip(risk, 0.0, 1.0))

        # 2. 采样计算动作熵
        n_samples = 20
        actions_samples = []
        for _ in range(n_samples):
            sampled_action, _ = model.predict(obs, deterministic=False)
            actions_samples.append(sampled_action)
        actions_array = np.array(actions_samples)

        _, normalized_entropy = compute_action_entropy(actions_array, action_space)

        # 3. 计算Q值特征
        action_mean = np.mean(actions_array, axis=0)
        q_mean, q_std, q_diff, q_min, q_max = compute_q_value_comprehensive(
            model, obs, action_mean, device, perception_type
        )

        # 4. 计算融合分数
        fusion_score, components = compute_fusion_score(
            risk=risk,
            action_entropy=normalized_entropy,
            q_diff=q_diff,
            q_std=q_std,
            risk_weight=risk_weight,
            entropy_weight=entropy_weight,
            q_weight=q_weight
        )

        # 5. 判断是否触发攻击
        # 只有当融合分数超过阈值且风险较高时才攻击
        should_attack = (fusion_score >= fusion_threshold) and (risk >= risk_threshold)

        # 记录统计信息
        if stats is not None:
            stats['fusion_scores'].append(fusion_score)
            stats['risks'].append(risk)
            stats['entropies'].append(normalized_entropy)
            stats['q_diffs'].append(q_diff)
            stats['q_stds'].append(q_std)
            if should_attack:
                stats['triggers'] += 1

        # 调试输出
        if debug and should_attack:
            print(f"[Smart-Q-Entropy Debug] Fusion={fusion_score:.3f} (risk={risk:.3f}, "
                  f"entropy={normalized_entropy:.3f}, q_diff={q_diff:.3f}), "
                  f"threshold={fusion_threshold}, risk_threshold={risk_threshold}")

        return should_attack, fusion_score, components

    except Exception as e:
        print(f"[Smart-Q-Entropy Trigger Error] {e}")
        return False, 0.0, {}


def smart_q_entropy_attack_sac_td3(obs_np, model, epsilon, perception_type, device,
                                    fusion_score=None, risk=None, env=None):
    """
    Smart Q-Entropy 融合攻击

    根据融合分数动态调整攻击强度:
    - 高融合分数: 强攻击
    - 低融合分数: 弱攻击

    参数:
        obs_np: 原始观测
        model: 模型
        epsilon: 基础攻击强度
        perception_type: 观测类型
        device: 计算设备
        fusion_score: 融合分数 (用于调整攻击强度)
        risk: 风险值 (用于调整攻击强度)
        env: 环境对象

    返回:
        adv_np: 对抗观测
    """
    model.policy.eval()

    # 动态调整攻击强度
    base_epsilon = epsilon
    if fusion_score is not None and risk is not None:
        # 融合分数越高，风险越大，攻击越强
        intensity_factor = 1.0 + fusion_score * 0.5 + risk * 0.5
        attack_epsilon = base_epsilon * intensity_factor
    else:
        attack_epsilon = base_epsilon

    # 获取干净动作
    with th.no_grad():
        try:
            action_clean_np, _ = model.predict(obs_np, deterministic=True)
        except Exception:
            action_clean_np = model.predict(obs_np)
    action_clean_t = th.from_numpy(np.asarray(action_clean_np)).float().to(device)
    if action_clean_t.ndim == 1:
        action_clean_t = action_clean_t[None, ...]

    # 准备观测张量
    obs_t = _to_torch(obs_np, device, perception_type)
    obs_t.requires_grad_(True)

    # 获取对抗动作
    action_adv_t = None
    for candidate in (getattr(model, 'actor', None), getattr(model.policy, 'actor', None)):
        if candidate is None:
            continue
        try:
            action_adv_t = candidate(obs_t)
            break
        except Exception:
            action_adv_t = None

    if action_adv_t is None:
        try:
            act_np_fallback = model.policy._predict(obs_t.detach().cpu().numpy())
            action_adv_t = th.from_numpy(np.asarray(act_np_fallback)).float().to(device)
        except Exception:
            raise RuntimeError("无法从模型获得可导的 action 张量。")

    if action_adv_t.ndim == 1:
        action_adv_t = action_adv_t[None, ...]

    # 损失函数: 最大化动作差异 + 考虑Q值
    loss = -F.mse_loss(action_adv_t, action_clean_t)

    # 可选: 结合Q值优化 (使用更可靠的方法)
    try:
        # 方法1: 尝试使用policy内置方法
        if hasattr(model.policy, '_q_value_forward'):
            with th.no_grad():
                q_vals = model.policy._q_value_forward(obs_t.detach(), action_adv_t)
                if isinstance(q_vals, tuple) and len(q_vals) >= 2:
                    q_loss = -0.1 * (q_vals[0].mean() + q_vals[1].mean()) / 2
                    loss = loss + q_loss
                elif th.is_tensor(q_vals):
                    q_loss = -0.1 * q_vals.mean()
                    loss = loss + q_loss
        # 方法2: 手动构建critic输入（适应CNN_GAP网络）
        elif hasattr(model, 'critic') and model.critic is not None:
            critic = model.critic
            if hasattr(critic, 'q_networks') and len(critic.q_networks) >= 2:
                with th.no_grad():
                    # 尝试使用policy的特征提取器
                    if hasattr(model.policy, 'features') and hasattr(model.policy.features, 'forward'):
                        features = model.policy.features(obs_t.detach())
                        features = features.reshape(features.shape[0], -1)
                    else:
                        features = obs_t.detach().reshape(obs_t.shape[0], -1)

                    action_flat = action_adv_t.reshape(action_adv_t.shape[0], -1)
                    input_combined = th.cat([features, action_flat], dim=1)
                    input_combined = input_combined.view(input_combined.size(0), -1)

                    q1 = critic.q_networks[0](input_combined)
                    q2 = critic.q_networks[1](input_combined)
                    q_loss = -0.1 * (q1.mean() + q2.mean()) / 2
                    loss = loss + q_loss
    except Exception:
        pass  # Q值优化失败，继续使用动作差异损失

    # 反向传播
    model.policy.zero_grad()
    if obs_t.grad is not None:
        obs_t.grad.zero_()
    loss.backward(retain_graph=False)

    if obs_t.grad is None:
        raise RuntimeError("obs gradient is None after backward.")

    # FGSM步
    grad_sign = th.sign(obs_t.grad.data)
    adv_t = obs_t + attack_epsilon * grad_sign

    # 转换回numpy
    if perception_type == 'vector' or perception_type == 'lgmd':
        adv_np = adv_t.detach().cpu().numpy()
        if obs_np.ndim == 1 and adv_np.shape[0] == 1:
            adv_np = adv_np[0]
        adv_np = np.clip(adv_np, 0.0, 1.0)
    else:
        adv_cpu = adv_t.detach().cpu().numpy()[0]
        adv_cpu = np.transpose(adv_cpu, (1, 2, 0))
        adv_cpu = np.clip(adv_cpu, 0.0, 1.0)
        adv_np = (adv_cpu * 255.0).astype(np.uint8)

    return adv_np


if __name__ == "__main__":
    try:
        # main()
        run_eval_multi()
    except KeyboardInterrupt:
        print('system exit')
