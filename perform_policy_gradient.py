import torch
import math
from torch.utils.tensorboard import SummaryWriter

from rl_envs.episodic_grid_world_env import EpisodicGridWorldEnv
from agents.policy_gradient import PGAgent
# Set up lists to store episode rewards and lengths
episode_rewards = []
episode_lengths = []

LEARN_RATE = 0.01
def print_actions(agent, env, get_optimal = False):
    with torch.no_grad():
        action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
        for i in range(env.height):
            print("[", end=" ")
            for j in range(env.width):
                state = torch.tensor((i,j), dtype=torch.float)
                action = agent.get_action(state)
                print(action_mapping[action], end=" ")
            print("]")

env = EpisodicGridWorldEnv(5, 5, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=-1, hit_wall_reward=-1)
agent = PGAgent(2, env.possible_actions, lr = LEARN_RATE)
writer = SummaryWriter()


num_episodes = 10000
for episode in range(num_episodes):
    state = (0,0)
    episode_recorder = []
    steps = 0
    while not env.is_terminated:
        action = agent.get_action(torch.tensor(state, dtype=torch.float))
        next_state, reward, done = env.step(state, action)
        episode_recorder.append((state, action, reward, next_state))
        state = next_state
        if steps > 50:
            break
        steps += 1
    
    discounted_reward = 0
    for state, action, reward, next_state in reversed(episode_recorder):
        # value update
        discounted_reward = discounted_reward * env.discounted_factor + reward
        agent.q[state][action] = discounted_reward 

        # policy update
        agent.optimizer.zero_grad()
        """
        特别注意: 这里 log π 中的 π(a|s) 是选择 a 的概率, policy network 得输出一个概率, 而不是什么 a 的值
        当然我们可以用输出的值, 归一化一下作为 action 的概率

        有一个 变体可能, 在 sample action 时就计算 prob并存储, 然后在 update 时就只是计算 reward 从而计算 loss, 将一个 episode 的loss 都加到一起来一起 backward, 然后更新一次 policy network
        """

        actions_val = abs(agent.policy_net(torch.tensor(state, dtype=torch.float)))
        action_probs = actions_val/actions_val.sum()
        loss = torch.log(action_probs[action]) * discounted_reward
        # loss = abs(loss)
        loss.backward()
        agent.optimizer.step()

    writer.add_scalar('Loss', loss, episode)
writer.flush()
writer.close()
print(env)
print_actions(agent, env)
