import gymnasium as gym
import numpy as np

from reward_predictor import RewardPredictorEnsemble

# from pref_db import PrefDB


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, args):
        super().__init__()
        self.base_env = gym.make("Swimmer-v4")
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.include_actions = args.include_actions
        self.reward_predictor = RewardPredictorEnsemble(
            args.ensemble_size,
            self.observation_space.shape,
            self.action_space.shape,
            args.reward_learning_rate,
            args.rwd_mdl_bs,
        )
        self.observations = []
        self.actions = []
        self.dones = []
        # self.preference_db = PrefDB()

    def step(self, action):
        obs, _base_reward, term, trunc, info = self.base_env.step(action)
        reward = self.reward_predictor.get_reward(obs, action)
        # reward = _base_reward
        self.observations.append(obs)
        self.actions.append(action)
        self.dones.append(term)
        return obs, reward, term, trunc, info

    def reset(self, seed=None, options=None):
        self.prev_obs, info = self.base_env.reset(seed=seed, options=options)
        return self.prev_obs, info

    def render(self):
        # TODO: Laryn
        ...

    def close(self):
        ...
