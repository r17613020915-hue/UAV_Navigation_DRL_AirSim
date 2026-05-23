import os
import sys
from types import SimpleNamespace

import gym
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from backdoor_attacks import BackdoorAttackCallback, ImageTrigger, create_trigger, is_action_target


class FakeReplayBuffer:
    def __init__(self):
        self.buffer_size = 8
        self.n_envs = 1
        self.pos = 4
        self.full = False
        self.optimize_memory_usage = False
        self.observations = np.zeros((8, 1, 2, 60, 90), dtype=np.uint8)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros((8, 1, 2), dtype=np.float32)
        self.rewards = np.zeros((8, 1), dtype=np.float32)
        self.dones = np.zeros((8, 1), dtype=np.float32)

    def size(self):
        return self.pos


class FakeModel:
    def __init__(self, rb):
        self.replay_buffer = rb
        self.num_timesteps = 100
        self.action_space = gym.spaces.Box(
            low=np.array([2.0, -0.524], dtype=np.float32),
            high=np.array([5.0, 0.524], dtype=np.float32),
            dtype=np.float32,
        )

    def get_env(self):
        return None


def test_image_trigger_supports_transposed_and_replay_shapes():
    trigger = create_trigger(trigger_size=20, trigger_type="solid")

    chw = np.zeros((2, 60, 90), dtype=np.uint8)
    chw_poisoned = trigger(chw)
    assert chw_poisoned[0, :20, :20].min() == 255
    assert chw_poisoned[1].max() == 0

    replay_obs = np.zeros((4, 1, 2, 60, 90), dtype=np.uint8)
    replay_poisoned = trigger.apply_to_batch(replay_obs)
    assert replay_poisoned[:, :, 0, :20, :20].min() == 255
    assert replay_poisoned[:, :, 1].max() == 0


def test_sac_replay_poisoning_updates_recent_samples():
    rb = FakeReplayBuffer()
    callback = BackdoorAttackCallback(
        attack_type="sleepernets",
        p_rate=1.0,
        target_action=1,
        trigger_size=20,
        trigger_type="solid",
        reward_positive=40.0,
        reward_negative=-40.0,
        clip_min=-40.0,
        clip_max=40.0,
        start_step=0,
        poison_recent_only=True,
        poison_next_obs=True,
        force_target_action=True,
        verbose=0,
    )
    callback.init_callback(FakeModel(rb))
    callback.locals = {"num_collected_steps": 4}
    callback.num_timesteps = 100

    callback._poison_sac_buffer()

    assert callback.total_poisoned == 4
    assert rb.observations[:4, 0, 0, :20, :20].min() == 255
    assert rb.next_observations[:4, 0, 0, :20, :20].min() == 255
    assert np.allclose(rb.rewards[:4, 0], 40.0)
    for i in range(4):
        assert is_action_target(
            rb.actions[i, 0],
            1,
            action_space=callback.action_space,
            scaled=True,
        )
