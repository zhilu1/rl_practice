from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_envs.grid_world_env import GridWorldEnv
from ReplayMemory import *
from math import cos, sin, pi

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

BATCHSIZE = 100
LEARN_RATE = 1e-5
TRUE_RANDOM_STATE_VALUE = [
    [-3.8, -3.8, -3.6, -3.1, -3.2],
    [-3.8, -3.8, -3.8, -3.1, -2.9],
    [-3.6, -3.9, -3.4, -3.2, -2.9],
    [-3.9, -3.6, -3.4, -2.9, -3.2],
    [-4.5, -4.2, -3.4, -3.4, -3.5],         
]


env = GridWorldEnv(5, 5, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=-1, hit_wall_reward=-1)
writer = SummaryWriter()

v_func_parameters = [0 for _ in range(4)]
def phi_s(state, i):
    # t = {0: 1, 1:state[0], 2: state[1], 3:state[0]**2, 4: state[1]**2,5: state[0]*state[1]}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    t = {0: 1, 1:cos(state[0] * pi), 2: cos(state[1] * pi), 3:cos((state[1]+state[0]) * pi)}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    return t[i]
def estimate_v(state, paramters):
    y, x = state
    v = 0
    for i, p in enumerate(paramters):
        v += phi_s(state, i) * p
    # v = paramters[0] * phi_s(state, 0) + paramters[1]*x + paramters[2]*y + paramters[3]*x*x + paramters[4]*y*y + paramters[5]*x*y
    return v

"""
generate samples to replay buffer
"""


replay_buffer = ReplayMemory(10000)

state = env.reset()
for _ in range(10000):
    action = random.randint(0,4)
    next_state, reward = env.step(state, action)
    replay_buffer.push(state, action, reward, next_state)
    state = next_state

iter_counter = 0

for episode_ind in range(5): # 5 episode
    # state = env.get_random_start()
    for step_ind in range(10000): # 10000 steps each episode
        # action = random.randint(0,4)
        # next_state, reward = env.step(state, action)
        state, action, reward, next_state  = replay_buffer.sample()[0]
        TD_error = reward + env.discounted_factor * estimate_v(next_state, v_func_parameters) - estimate_v(state, v_func_parameters)
        for i, param in enumerate(v_func_parameters):
            v_func_parameters[i] = param + 0.001 * TD_error * phi_s(state, i)
        
        # state = next_state

        if iter_counter % 50:
            state_value_error = 0
            for i in range(env.height):
                for j in range(env.width):
                    state_value_error += (estimate_v((i,j), v_func_parameters) - TRUE_RANDOM_STATE_VALUE[i][j]) ** 2
            writer.add_scalar('state value error', state_value_error, iter_counter // 50)   
        iter_counter+=1
        
writer.flush()
writer.close()

for i in range(env.height):
    print("[", end=" ")
    for j in range(env.width):
        v_hat = estimate_v((i,j), v_func_parameters)
        print(v_hat, end=" ")
    print("]")

print()
print("Difference between true value and estimated")

for i in range(env.height):
    print("[", end=" ")
    for j in range(env.width):
        v_hat = estimate_v((i,j), v_func_parameters)
        print(v_hat - TRUE_RANDOM_STATE_VALUE[i][j], end=" ")
    print("]")

    