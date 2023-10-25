import gymnasium as gym
import numpy as np

# from reward_predictor import RewardPredictorNetwork
# from pref_db import PrefDB


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, args):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.base_env = gym.make("Swimmer-v4")
        self.action_space = self.base_env.action_space
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = self.base_env.observation_space
        self.include_action = args.include_action
        # self.reward_predictor = RewardPredictorNetwork()
        self.observations = []
        self.actions = []
        self.dones = []
        # self.preference_db = PrefDB()

    def step(self, action):
        obs, _base_reward, term, trunc, info = self.base_env.step(action)
        # if
        # reward = self.reward_predictor(self.observations[-1])
        reward = _base_reward
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
