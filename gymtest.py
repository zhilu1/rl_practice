import gymnasium as gym




env = gym.make("InvertedPendulum-v4")
observation, info = env.reset()

for _ in range(1000):

    action = env.action_space.sample()  # agent policy that uses the observation and info
    # insert an algorithm that can interact with env and output an action here
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

