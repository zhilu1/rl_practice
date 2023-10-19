from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

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
                for action in range(env.action_n):
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


env = GridWorldEnv(5, 5, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=-1, hit_wall_reward=-1)
writer = SummaryWriter()

q_func_parameters = [0 for _ in range(17)]
def phi_sa(state, action, i):
    x,y,z = *state, action
    t = [1,x,y,z,x*y,y*z,x*z,x*x,y*y,z*z,x*x*z,y*y*z,x*x*y,y*y*x,z*z*x,z*z*y,x*y*z]
    # t = {0: 1, 1:x, 2: y, 3:z, 4: x*y, 5: y*z, 6: x*z,7: x*x,}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    # t = {0: 1, 1:state[0], 2: state[1], 3:state[0]**2, 4: state[1]**2,5: state[0]*state[1]}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    # t = {0: 1, 1:cos(state[0] * pi), 2: cos(state[1] * pi), 3:cos((state[1]+state[0]) * pi)}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
    return t[i]
def estimate_q(state,  action, paramters):
    q = 0
    for i, param in enumerate(paramters):
        q += phi_sa(state, action, i) * param
    # v = paramters[0] * phi_s(state, 0) + paramters[1]*x + paramters[2]*y + paramters[3]*x*x + paramters[4]*y*y + paramters[5]*x*y
    return q
# def estimate_max_q(state, parameters):
#     for action in range(5):
#         estimate_q(q, action, parameters)
    # return q
def calculate_state_value_error(env: GridWorldEnv, paramters):
    # offline policy have 2 policies, I am using the behavior(random) policy for calculating
    with torch.no_grad():
        state_value_error = 0
        for i in range(env.height):
            for j in range(env.width):
                state = (i,j)
                output = 0
                for action in range(env.action_n):
                    output += estimate_q(state,  action, paramters)
                state_value = output/env.action_n
                state_value_error += (state_value - TRUE_RANDOM_STATE_VALUE[i][j])
    return state_value_error
"""
generate samples to replay buffer
"""


"""
replay_buffer = ReplayMemory(10000)

state = env.get_random_start()
for _ in range(10000):
    action = random.randint(0,4)
    next_state, reward = env.step(state, action)
    replay_buffer.push(state, action, reward, next_state)
    state = next_state

iter_counter = 0

for episode_ind in range(1): # 5 episode
    # state = env.get_random_start()
    for step_ind in range(10000): # 10000 steps each episode
        # action = random.randint(0,4)
        # next_state, reward = env.step(state, action)
        state, action, reward, next_state  = replay_buffer.sample()[0]

        next_q = -float('inf')
        for poss_action in range(env.action_n):
            next_q = max(next_q, estimate_q(next_state, poss_action, q_func_parameters))
                       
        TD_error = reward + env.discounted_factor * next_q - estimate_q(state, action, q_func_parameters)
        for i, param in enumerate(q_func_parameters):
            q_func_parameters[i] = param + 0.0001 * TD_error * phi_sa(state, action, i)

        
        # state = next_state

        if iter_counter % 500:
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

print_actions(env, q_func_parameters)
"""


"""
try start at initial position 
"""
from agents.approx_Q_Learning import QLearningAgent

iter_counter = 0

agent = QLearningAgent(env.action_n)

for episode_ind in range(1): # 5 episode
    state = env.get_random_start()
    for step_ind in range(10000): # 10000 steps each episode
        action = agent.get_action(state)
        next_state, reward = env.step(state, action)

        next_q = -float('inf')
        for poss_action in range(env.action_n):
            next_q = max(next_q, estimate_q(next_state, poss_action, q_func_parameters))
                       
        TD_error = reward + env.discounted_factor * next_q - estimate_q(state, action, q_func_parameters)
        for i, param in enumerate(q_func_parameters):
            q_func_parameters[i] = param + 0.001 * TD_error * phi_sa(state, action, i)

        max_q_val = - float('inf')
        best_action = 0
        for poss_action in range(env.action_n):
            q_val = estimate_q(state, poss_action, q_func_parameters)
            if q_val > max_q_val:
                max_q_val = q_val
                best_action = action
                

        agent.policy_improvement(state, best_action)
        state = next_state

        if iter_counter % 500:
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

action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
for i in range(env.height):
    print("[", end=" ")
    for j in range(env.width):
        action = agent.get_action((i,j), True)
        print(action_mapping[action], end=" ")
    print("]")