from pathlib import Path
import sys
import numpy as np
from collections import defaultdict



# from rl_envs.episodic_grid_world_env import EpisodicGridWorldEnv as GridWorldEnv
from rl_envs.grid_world_env import GridWorldEnv


def print_actions(agent, env, get_optimal = False):
    action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i,j), get_optimal)
            print(action_mapping[action], end=" ")
        print("]")



"""
start at somewhere and find optimal path using n-step Sarsa
"""
# from agents.nstep_Sarsa import NstepSarsaAgent as AgentAlgo

# env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-10)
# agent = AgentAlgo(step_num=5, action_space_n=env.possible_actions)

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
# agent = AgentAlgo(action_space_n=env.possible_actions)



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


env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-10)
agent = AgentAlgo(action_space_n=env.possible_actions, epsilon=0)

state = (0,0)
trajectorys = []
episode_num = 500
# collecting experiences
for _ in range(episode_num):
    trajectory = []
    while state not in env.target_grids:
        action = agent.get_behavior_acion(state)
        next_state, reward = env.step(state, action)
        trajectory.append((state, action, reward , next_state))
        state = next_state
    trajectorys.append(trajectory)

    for state, action, reward, next_state in trajectory:
        td_target = reward + env.discounted_factor * max(agent.q[next_state])
        agent.td_learn(state, action, td_target)
        agent.policy_improvement(state)




"""
output
"""
print(env)

print_actions(agent, env, get_optimal = True)

# for i in range(env.height):
#     print("[", end=" ")
#     for j in range(env.width):
#         print('{:3f}'.format(agent.v[(i,j)]), end=" ")
#     print("]")