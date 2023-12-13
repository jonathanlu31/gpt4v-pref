import gymnasium as gym

from stable_baselines3 import PPO, DQN
from custom_env import CustomEnv

class Args:
    pass

args = Args()
args.base_env = 'CartPole-v1'
args.include_actions = True
args.ensemble_size = 1
args.reward_learning_rate = 1
args.rwd_mdl_bs = -1
args.reward_model_checkpoint_path = ''
args.pretrained_reward_model_path = 'cartpole_human_reward.pkl'

env = CustomEnv(args)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./cartpole_human_policy')
model.learn(total_timesteps=200_000)

# vec_env = model.get_env()
# # render_env = gym.make(args.base_env, render_mode='rgb_array')
# obs = vec_env.reset()
# for i in range(5000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")