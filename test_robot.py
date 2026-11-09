import torch
import gymnasium as gym
from stable_baselines3 import PPO

print("PyTorch 版本:", torch.__version__)

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

print("开始训练 AI 平衡小车...")
model.learn(total_timesteps=20000)
print("训练完成！")