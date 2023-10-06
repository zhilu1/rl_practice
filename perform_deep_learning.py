from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
import torch


# from rl_envs.episodic_grid_world_env import EpisodicGridWorldEnv as GridWorldEnv
from rl_envs.grid_world_env import GridWorldEnv
from ReplayMemory import *

def print_actions(agent, env, get_optimal = False):
    action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i,j))
            print(action_mapping[action], end=" ")
        print("]")


from agents.DQN import DeepQLearningAgent

env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-10, hit_wall_reward=-1)
agent = DeepQLearningAgent(state_space_n= env.height * env.width, action_space_n=env.possible_actions, )

"""
generate samples to replay buffer
"""


replay_buffer = ReplayMemory(1000)

state = env.get_random_start()
for _ in range(100):
    action = agent.get_behavior_acion(state)
    next_state, reward = env.step(state, action)
    replay_buffer.push(torch.tensor(env.state_index(state)).unsqueeze(0), torch.tensor(action).unsqueeze(0), torch.tensor(reward).unsqueeze(0), torch.tensor(env.state_index(next_state)).unsqueeze(0))
    state = next_state

BATCHSIZE = 5
"""
perform executing
"""
for _ in range(100):
    for _ in range(5):
        transitions  = replay_buffer.sample(BATCHSIZE)
        batch = Transition(*zip(*transitions))
        state = torch.stack(batch.state)
        next_state = torch.stack(batch.next_state)
        reward = torch.stack(batch.reward)
        action = torch.stack(batch.action)
        torch.nn.functional.one_hot(action.type(torch.LongTensor), num_classes=env.possible_actions)
        
        agent.optimizer.zero_grad()
        target_q = max(agent.target_net(next_state))
        target_value = env.discounted_factor * target_q + reward

        # minimize distance between policy result and target value, the loss choose can be ..
        q = agent.policy_net(state)[action]
        loss = agent.loss(q, target_value)
        loss.backward()
        agent.optimizer.step()
    # copy target network every C=5 iteration            
    agent.target_net.load_state_dict(agent.policy_net.state_dict())

print_actions(agent, env, True)


