import gymnasium as gym
import numpy as np
import torch

from reward_predictor import RewardPredictorEnsemble

# from pref_db import PrefDB


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, args):
        super().__init__()
        self.base_env = gym.make(args.base_env, render_mode="rgb_array")
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.include_actions = args.include_actions
        self.reward_predictor = RewardPredictorEnsemble(
            args.ensemble_size,
            self.observation_space.shape,
            (1,) if args.base_env == 'CartPole-v1' else self.action_space.shape,
            args.reward_learning_rate,
            args.rwd_mdl_bs,
            args.reward_model_checkpoint_path,
        )
        self.observations = []
        self.actions = []
        self.dones = []

        if args.pretrained_reward_model_path:
            self.reward_predictor.load_state_dict(torch.load(args.pretrained_reward_model_path))

    def step(self, action):
        obs, _base_reward, term, trunc, info = self.base_env.step(action)
        obs = obs.astype(np.float32)
        action = torch.tensor(action) if np.isscalar(action) else torch.from_numpy(action)
        reward = self.reward_predictor.get_reward(
            torch.from_numpy(obs), action
        )
        # reward = _base_reward
        self.observations.append(obs)
        self.actions.append(action)
        self.dones.append(term)
        return obs, float(reward), term, trunc, info

    def reset(self, seed=None, options=None):
        self.prev_obs, info = self.base_env.reset(seed=seed, options=options)
        return self.prev_obs.astype(np.float32), info

    def render(self):
        return self.base_env.render()

    def close(self):
        ...
