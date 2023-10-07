from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

# from rl_envs.episodic_grid_world_env import EpisodicGridWorldEnv as GridWorldEnv
from rl_envs.grid_world_env import GridWorldEnv
from ReplayMemory import *
from math import cos, sin, pi

def print_actions(env, parameters):
    with torch.no_grad():
        action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
        for i in range(env.height):
            print("[", end=" ")
            for j in range(env.width):
                state = (i,j)
                max_q_val = - float('inf')
                best_action = 0
                for action in range(env.possible_actions):
                    q_val = estimate_q(state, action, parameters)
                    if q_val > max_q_val:
                        max_q_val = q_val
                        best_action = action
                print(action_mapping[best_action], end=" ")
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


env = GridWorldEnv(5, 5, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=-10, hit_wall_reward=-10)
writer = SummaryWriter()

q_func_parameters = [0 for _ in range(16)]
def phi_sa(state, action, i):
    x,y,z = *state, action
    t = [1,x,y,z,x*y,y*z,x*z,x*x,y*y,z*z,x*x*z,y*y*z,x*x*y,y*y*x,z*z*x,z*z*y,x*y*z]
    # t = {0: 1, 1:x, 2: y, 3:z, 4: x*y, 5: y*z, 6: x*z,7: x*x,}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    # t = {0: 1, 1:state[0], 2: state[1], 3:state[0]**2, 4: state[1]**2,5: state[0]*state[1]}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    # t = {0: 1, 1:cos(state[0] * pi), 2: cos(state[1] * pi), 3:cos((state[1]+state[0]) * pi)}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    return t[i]
def estimate_q(state,  action, paramters):
    y, x = state
    q = 0
    for i, p in enumerate(paramters):
        q += phi_sa(state, action, i) * p
    # v = paramters[0] * phi_s(state, 0) + paramters[1]*x + paramters[2]*y + paramters[3]*x*x + paramters[4]*y*y + paramters[5]*x*y
    return q
def calculate_state_value_error(env: GridWorldEnv, paramters):
    # offline policy have 2 policies, I am using the behavior(random) policy for calculating
    with torch.no_grad():
        state_value_error = 0
        for i in range(env.height):
            for j in range(env.width):
                state = (i,j)
                output = 0
                for action in range(env.possible_actions):
                    output += estimate_q(state,  action, paramters)
                state_value = output/env.possible_actions
                state_value_error += (state_value - TRUE_RANDOM_STATE_VALUE[i][j])
    return state_value_error
"""
generate samples to replay buffer
"""


replay_buffer = ReplayMemory(10000)

state = env.get_random_start()
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
        TD_error = reward + env.discounted_factor * estimate_q(next_state, action, q_func_parameters) - estimate_q(state, action, q_func_parameters)
        for i, param in enumerate(q_func_parameters):
            q_func_parameters[i] = param + 0.001 * TD_error * phi_sa(state, action, i)

        
        # state = next_state

        if iter_counter % 50:
            state_value_error = calculate_state_value_error(env,q_func_parameters)
            writer.add_scalar('TD error', TD_error, iter_counter)         
            writer.add_scalar('state_value_error', state_value_error, iter_counter)         

        #     state_value_error = 0
        #     for i in range(env.height):
        #         for j in range(env.width):
        #             state_value_error += (estimate_q((i,j), action, q_func_parameters) - TRUE_RANDOM_STATE_VALUE[i][j]) ** 2
        #     writer.add_scalar('state value error', state_value_error, iter_counter // 50)   
        iter_counter+=1
        
writer.flush()
writer.close()

# for i in range(env.height):
#     print("[", end=" ")
#     for j in range(env.width):
#         v_hat = estimate_v((i,j), q_func_parameters)
#         print(v_hat, end=" ")
#     print("]")

# print()
# print()

# for i in range(env.height):
#     print("[", end=" ")
#     for j in range(env.width):
#         v_hat = estimate_v((i,j), q_func_parameters)
#         print(v_hat - TRUE_RANDOM_STATE_VALUE[i][j], end=" ")
#     print("]")

    