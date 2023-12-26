import gymnasium as gym

from stable_baselines3 import PPO, DQN, SAC
from custom_env import CustomEnv

class Args:
    pass

args = Args()
args.base_env = 'Walker2d-v4'
args.include_actions = False
args.ensemble_size = 1
args.reward_learning_rate = 1
args.rwd_mdl_bs = -1
args.reward_model_checkpoint_path = ''
args.pretrained_reward_model_path = 'walker_ai_reward_split.pkl'

env = CustomEnv(args)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=130_000)
new_model = PPO.load('split.zip', env=env)
model.save('split')


vec_env = new_model.get_env()
obs = vec_env.reset()
for i in range(6000):
    action, _state = new_model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
