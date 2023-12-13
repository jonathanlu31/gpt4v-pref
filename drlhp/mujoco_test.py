import gymnasium as gym

from stable_baselines3 import PPO, A2C

import torch
import numpy as np

# print('hi')
env = gym.make("Swimmer-v4", render_mode='rgb_array')
print('stuff', env.action_space.shape)
model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    # print(action)
    action = np.array([[0, 0, 1]])
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
input()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
