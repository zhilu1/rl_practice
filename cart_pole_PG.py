import torch
import math
from torch.utils.tensorboard import SummaryWriter

from agents.policy_gradient import PGAgent
import gymnasium as gym

def visualize_in_gym(agent, env_name =  "", inp_env = None, steps=100):
    """
    render a environment and visualize it running N steps

    another possible solution can be gymnasium.experimental.wrappers.RecordVideoV0

    or gymnasium.utils.save_video.save_video?
    """
    if inp_env:
        demo_env = inp_env
    else:
        demo_env = gym.make(env_name, render_mode = "human")
    observation, info = demo_env.reset()

    for _ in range(steps):
        action = agent.get_action(torch.from_numpy(observation))  # agent policy that uses the observation and info
        # insert an algorithm that can interact with env and output an action here
        observation, reward, terminated, truncated, _ = demo_env.step(action)
        if terminated or truncated:
            observation, info = demo_env.reset()

    if not inp_env:
        demo_env.close()

env = gym.make("CartPole-v1")

agent = PGAgent(4, 2)

demo_env = gym.make("CartPole-v1", render_mode = "human")
# gym.utils.play.play(demo_env, fps=128)

writer = SummaryWriter()

DISCOUNTED_FACTOR = 0.9

num_episodes = 2000
for episode in range(num_episodes):
    episode_recorder = []
    observation, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.get_action(torch.from_numpy(observation))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_recorder.append((observation, action, reward, next_observation))
        observation = next_observation
        done = terminated or truncated
        episode_reward += reward

    # for observation, action, reward, next_observation in reversed(episode_recorder):
    loss = agent.update(episode_recorder, DISCOUNTED_FACTOR)

    writer.add_scalar('Loss', loss, episode)
    writer.add_scalar('Reward', episode_reward, episode)
    if episode % 100 == 0:
        visualize_in_gym(agent, inp_env=demo_env)
        # visualize_in_gym(agent, env_name="CartPole-v1")


writer.flush()
writer.close()


visualize_in_gym(agent, inp_env=demo_env)