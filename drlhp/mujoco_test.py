import gymnasium as gym

from stable_baselines3 import PPO

import torch

env = gym.make("Swimmer-v4", render_mode='rgb_array')
print('stuff', env.action_space.shape)
model = PPO("MlpPolicy", env, verbose=1, device=torch.device("mps"))
model.learn(total_timesteps=1_000_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
