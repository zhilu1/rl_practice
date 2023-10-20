import argparse
import numpy as np
from itertools import count
from rl_envs.gym_grid_world_env import GridWorldEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tools.helper import *

LEARN_RATE = 1e-5
DISCOUNTED_FACTOR = 0.9

FORBIDDEN_REWARD = -1
HITWALL_REWARD = -1
TARGET_REWARD = 1


# parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor (default: 0.99)')
# parser.add_argument('--seed', type=int, default=543, metavar='N',
#                     help='random seed (default: 543)')
# parser.add_argument('--render', action='store_true',
#                     help='render the environment')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='interval between training status logs (default: 10)')
# args = parser.parse_args()


env = GridWorldEnv(size=3,fixed_map = True, forbidden_grids=[(1,1)], target_grids=[(2,2)], forbidden_reward=FORBIDDEN_REWARD, hit_wall_reward=HITWALL_REWARD, target_reward=TARGET_REWARD)
# env = gym.make('CartPole-v1')
# env.seed(args.seed)
# torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 5)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-8)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    discounted_reward = 0
    policy_loss = []
    inireturns = []
    for r in policy.rewards[::-1]:
        discounted_reward = r + DISCOUNTED_FACTOR * discounted_reward
        inireturns.insert(0, discounted_reward)
    returns = torch.tensor(inireturns)
    std = returns.std() if returns.size(0) != 1 else 0
    returns = (returns - returns.mean()) / (std + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()

    torch.nn.utils.clip_grad.clip_grad_norm_(policy.parameters(), 100)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = -10
    for i_episode in count(1):
        ep_reward = 0
        obs, _ = env.reset()
        state = obs['agent']
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            obs, reward, terminate, truncated, _ = env.step(action)
            state = obs['agent']
            reward += 1
            policy.rewards.append(reward)
            ep_reward += reward
            if terminate or truncated:
                break

        running_reward = 0.05 * (ep_reward-t) + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % 100 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward-t, running_reward))
        if running_reward > 0.8:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
def get_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    return torch.argmax(probs).item()

def demo():
    env = GridWorldEnv(render_mode="human", size=3,fixed_map = True, forbidden_grids=[(1,1)], target_grids=[(2,2)], forbidden_reward=FORBIDDEN_REWARD, hit_wall_reward=HITWALL_REWARD, target_reward=TARGET_REWARD)
    for i in range(env.size):
        print("[", end=" ")
        for j in range(env.size):
            state = np.array([i, j])
            # action = np.argmax(policy(state))
            action = get_action(state)
            print(env.action_mappings[action], end=" ")
        print("]")

    obs, _ = env.reset()
    total_reward = 0
    for i in range(500):
        state = obs['agent']
        action = get_action(state)
        obs, reward, terminated, truncated, info  = env.step(action)
        # VecEnv resets automatically
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
            print('reward: {}, distance: {}'.format(total_reward, info))
            total_reward = 0
            if truncated:
                print("TRUNCATE")
            else:
                print("TERMINATE")
    env.close()




if __name__ == '__main__':
    main()
    demo()