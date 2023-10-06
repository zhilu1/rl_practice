from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

# from rl_envs.episodic_grid_world_env import EpisodicGridWorldEnv as GridWorldEnv
from rl_envs.grid_world_env import GridWorldEnv
from ReplayMemory import *

def print_actions(agent, env, get_optimal = False):
    with torch.no_grad():
        action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
        for i in range(env.height):
            print("[", end=" ")
            for j in range(env.width):
                state = torch.tensor((i,j), dtype=torch.float).unsqueeze(0)
                action = agent.get_action(state)
                print(action_mapping[action.item()], end=" ")
            print("]")

def state_normalize(env, state):
    return ((state[0] - (env.height-1)/2.0)/env.height,(state[1] - (env.width-1)/2.0)/env.width)
     

from agents.DQN import DeepQLearningAgent

BATCHSIZE = 100
LEARN_RATE = 1e-6

env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-1)
agent = DeepQLearningAgent(state_space_n= 2, action_space_n=env.possible_actions, lr = LEARN_RATE)
writer = SummaryWriter()
"""
generate samples to replay buffer
"""


replay_buffer = ReplayMemory(10000)

state = env.get_random_start()
for _ in range(10000):
    action = agent.get_behavior_acion(state)
    next_state, reward = env.step(state, action)
    replay_buffer.push(torch.tensor(state_normalize(env,state), dtype=torch.float), torch.tensor(action, dtype=torch.int64).unsqueeze(0), torch.tensor(reward, dtype=torch.float).unsqueeze(0), torch.tensor(state_normalize(env,next_state), dtype=torch.float))
    state = next_state

"""
perform executing
"""
iter_counter = 0
for _ in range(10000):
    for _ in range(50):
        transitions  = replay_buffer.sample(BATCHSIZE)
        batch = Transition(*zip(*transitions))
        state = torch.stack(batch.state)
        next_state = torch.stack(batch.next_state)
        reward = torch.cat(batch.reward)
        action_indices = torch.cat(batch.action)
        # action_indices = action
        # action = torch.nn.functional.one_hot(action, num_classes=env.possible_actions)
        
        target_q = torch.max(agent.target_net(next_state), dim=1).values
        target_value = env.discounted_factor * target_q + reward

        # minimize distance between policy result and target value, the loss choose can be ..
        agent.optimizer.zero_grad()
        output = agent.policy_net(state)
        q = output[torch.arange(output.size(0)), action_indices]
        # criterion = torch.nn.SmoothL1Loss()
        # loss = criterion(q, target_value)
        # loss.backward()

        loss = agent.loss(q, target_value)
        loss.sum().backward()
        # torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
        agent.optimizer.step()
    # copy target network every C=5 iteration   
    writer.add_scalar('TD error', (q - target_value).sum(), iter_counter)         
    writer.add_scalar('Loss', loss.sum(), iter_counter)
    iter_counter+=1
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    # print(loss)

writer.flush()
writer.close()
print(env)

print_actions(agent, env, True)


