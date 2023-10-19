from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

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
LEARN_RATE = 1e-5
TRUE_RANDOM_STATE_VALUE = [
    [-3.8, -3.8, -3.6, -3.1, -3.2],
    [-3.8, -3.8, -3.8, -3.1, -2.9],
    [-3.6, -3.9, -3.4, -3.2, -2.9],
    [-3.9, -3.6, -3.4, -2.9, -3.2],
    [-4.5, -4.2, -3.4, -3.4, -3.5],         
]

def calculate_state_value_error(env,agent):
    # offline policy have 2 policies, I am using the behavior(random) policy for calculating
    with torch.no_grad():
        state_value_error = 0
        for i in range(env.height):
            for j in range(env.width):
                state = torch.tensor((i,j), dtype=torch.float).unsqueeze(0)
                output = agent.policy_net(state)
                state_value = output.sum()/env.action_n
                state_value_error += (state_value - TRUE_RANDOM_STATE_VALUE[i][j])
    return state_value_error

env = GridWorldEnv(5, 5, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=-1, hit_wall_reward=-1, target_reward=10)
agent = DeepQLearningAgent(state_space_n= 2, action_space_n=env.action_n, lr = LEARN_RATE)
writer = SummaryWriter()
"""
generate samples to replay buffer
"""


replay_buffer = ReplayMemory(2000)

state = env.reset()
for _ in range(2000):
    action = random.randint(0,4)
    # action = agent.get_behavior_acion(state)
    next_state, reward = env.step(state, action)
    replay_buffer.push(torch.tensor(state_normalize(env,state), dtype=torch.float), torch.tensor(action, dtype=torch.int64).unsqueeze(0), torch.tensor(reward, dtype=torch.float).unsqueeze(0), torch.tensor(state_normalize(env,next_state), dtype=torch.float))
    state = next_state



"""
perform DQN
"""
iter_counter = 0
for _ in range(200):
    for _ in range(50):
        transitions  = replay_buffer.sample(BATCHSIZE)
        batch = Transition(*zip(*transitions))
        state = torch.stack(batch.state)
        next_state = torch.stack(batch.next_state)
        reward = torch.cat(batch.reward)
        action_indices = torch.cat(batch.action)

        loss, q_value, target_value = agent.update_Q_network(state, action_indices, reward, next_state, env.discounted_factor)
    # copy target network every C=5 iteration
    # state_value_estimated = output.sum(dim=1) / env.action_n 
    writer.add_scalar('TD error', (q_value - target_value).sum(), iter_counter)         
    writer.add_scalar('Loss', loss.sum(), iter_counter)
    writer.add_scalar('State value error', calculate_state_value_error(env,agent), iter_counter)


    iter_counter+=1
    # agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.sync_target_network()
    # print(loss)

writer.flush()
print(env)

print_actions(agent, env, True)

print()

for i in range(env.height):
    print("[", end=" ")
    for j in range(env.width):
        state = torch.tensor((i,j), dtype=torch.float).unsqueeze(0)
        output = agent.policy_net(state)
        state_value = output.sum()/env.action_n
        state_value_error = (state_value - TRUE_RANDOM_STATE_VALUE[i][j])
        print(state_value_error, end=" ")
    print("]")

# print()
