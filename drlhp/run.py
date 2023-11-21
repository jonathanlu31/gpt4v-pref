import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps
import numpy as np
import random
import sys

from custom_env import CustomEnv
from params import parse_args


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
    env = CustomEnv(args)
    policy = PPO("MlpPolicy", env, verbose=1)
    # event_callback = EveryNTimesteps(
    #     n_steps=args.train_reward_interval,
    #     callback=lambda: train_reward_predictor(args, env),
    # )

    # policy.learn(1e6, callback=event_callback)
    policy.learn(1e6)

    vec_env = policy.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = policy.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render("human")


if __name__ == "__main__":
    main(sys.argv[1:])
