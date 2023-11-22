import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps
import numpy as np
import random
import sys
from multiprocessing import Process, Queue
import os

from custom_env import CustomEnv
from params import parse_args
from pref_interface import PrefInterface


def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def train_reward_predictor(args, env):
    # Rating the buffer of segments collected so far
    # Training the reward predictor on those segments
    pass


def main(args):
    args = parse_args(args)

    set_random_seed(args.seed)

    run(args)


def run(args):
    seg_pipe = Queue(maxsize=1)
    pref_pipe = Queue(maxsize=1)
    env = CustomEnv(args)
    policy = PPO("MlpPolicy", env, verbose=1)
    event_callback = EveryNTimesteps(
        n_steps=args.train_reward_interval,
        callback=lambda: train_reward_predictor(args, env),
    )

    # Have policy run for x number of steps first to collect some trajectories
    # Then start the full process
    policy.learn(1e6, callback=event_callback)
    # policy.learn(1e6)

    vec_env = policy.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = policy.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render("human")

def start_preference_labeling_process(args, seg_pipe, pref_pipe, log_dir, max_segs):
    def f():
        # The preference interface needs to get input from stdin. stdin is
        # automatically closed at the beginning of child processes in Python,
        # so this is a bit of a hack, but it seems to be fine.
        sys.stdin = os.fdopen(0)
        pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)

    # Needs to be done in the main process because does GUI setup work
    prefs_log_dir = os.path.join(log_dir, 'pref_interface')
    pi = PrefInterface(max_segs=max_segs,
                       log_dir=prefs_log_dir)
    proc = Process(target=f, daemon=True)
    proc.start()
    return pi, proc



if __name__ == "__main__":
    main(sys.argv[1:])
