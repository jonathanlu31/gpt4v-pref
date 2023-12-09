import gymnasium as gym

from stable_baselines3 import A2C

print("h")
env = gym.make("Ant-v4")
print("s")
model = A2C("MlpPolicy", env, verbose=1)
print("hi")
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
