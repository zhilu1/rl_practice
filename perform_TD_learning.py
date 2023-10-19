from pathlib import Path
import sys
import numpy as np
from collections import defaultdict


from tools.helper import *

from rl_envs.gym_grid_world_env import GridWorldEnv


"""
start at somewhere and find optimal path using n-step Sarsa
"""
# from agents.nstep_Sarsa import NstepSarsaAgent as AgentAlgo

# env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-10)
# agent = AgentAlgo(step_num=5, action_space_n=env.action_n)

# for _ in range(5000): # find target 500 times
#     state = (2,0) # initial_state
#     action = agent.get_action(state)
#     while state not in env.target_grids:
#         rewards_ls = []
#         next_state = state
#         next_action = action
#         for _ in range(agent.step_num):
#             next_state, reward = env.step(next_state, next_action)
#             next_action = agent.get_action(next_state)
#             rewards_ls.append(reward)
#             if next_state in env.target_grids:
#                 break
        
#         td_target = 0
#         for reward in reversed(rewards_ls):
#             td_target *= env.discounted_factor
#             td_target += reward
        
#         td_target += agent.q[next_state][next_action]
#         # agent.update_q(state, action, reward, next_state, next_action, env.discounted_factor)
#         agent.td_learn(state, action, td_target)
#         agent.policy_improvement(state)
#         state = next_state
#         action = next_action

"""
perform online Q-learning
"""
from agents.Q_Learning import QLearningAgent as AgentAlgo


# env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-10)
# agent = AgentAlgo(action_space_n=env.action_n)



# for _ in range(500): # find target 500 times
#     state = (2,0) # initial_state
#     action = agent.get_action(state)
#     while state not in env.target_grids:
#         next_state, reward = env.step(state, action)
#         td_target = reward + env.discounted_factor * max(agent.q[next_state])
#         # td_target += agent.q[next_state][next_action]
#         # agent.update_q(state, action, reward, next_state, next_action, env.discounted_factor)
#         agent.td_learn(state, action, td_target)
#         agent.policy_improvement(state)
#         state = next_state
#         action = agent.get_action(state)

"""
perform offline Q-learning, but calculating all state's optimal paths
"""
from agents.Q_Learning import QLearningAgent as AgentAlgo


env = GridWorldEnv(fixed_map = True, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=-1, hit_wall_reward=-1, target_reward=1)
# env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-10)
agent = AgentAlgo(action_space_n=int(env.action_n), epsilon=0)

# obs, _ = env.reset(options = {'start_position': (0,0)})
# state = tuple(obs['agent'])
episode_len = 100000
# collecting experiences
obs, _ = env.reset()
trajectory = []
for _ in range(episode_len):
    state = tuple(obs['agent'])
    action = agent.get_behavior_action(state)
    obs, reward, terminated , truncated, info = env.step(action)
    next_state = tuple(obs['agent'])
    trajectory.append((state, action, reward , next_state))
    state = next_state

for state, action, reward, next_state in trajectory:
    td_target = reward + 0.9 * max(agent.q[next_state])
    agent.td_learn(state, action, td_target)
    agent.policy_improvement(state)




"""
output
"""
print(env)

print_actions(agent, env, get_optimal = True)

Q = agent.q
V = {}
for state in Q.keys():
    V[state] = max(Q[state])
print_by_dict(env, V)

print()
V, iteration_count = compute_state_value(env.height, env.width, env, agent.policy)
print_by_dict(env, V)
print(iteration_count)

gridworld_demo(agent, forbidden_reward=-1, hit_wall_reward=-1, target_reward=1)
# for i in range(env.height):
#     print("[", end=" ")
#     for j in range(env.width):
#         print('{:3f}'.format(agent.v[(i,j)]), end=" ")
#     print("]")