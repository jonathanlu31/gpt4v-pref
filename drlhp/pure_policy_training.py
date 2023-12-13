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
args.pretrained_reward_model_path = 'walker_human_reward_split.pkl'

env = CustomEnv(args)
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log='./walker_human_policy')
model.learn(total_timesteps=10_000)
model.save('walker_model')

new_model = SAC.load('walker_model.zip', env=env)

vec_env = new_model.get_env()
obs = vec_env.reset()
for i in range(5000):
    action, _state = new_model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
