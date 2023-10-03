from pathlib import Path
import sys
import numpy as np
from collections import defaultdict



# from rl_envs.episodic_grid_world_env import EpisodicGridWorldEnv as GridWorldEnv
from rl_envs.grid_world_env import GridWorldEnv
from agents.nstep_Sarsa import NstepSarsaAgent as AgentAlgo


def print_actions(agent, env, get_optimal = False):
    action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i,j), get_optimal)
            print(action_mapping[action], end=" ")
        print("]")



env = GridWorldEnv(5, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-10)


agent = AgentAlgo(step_num=1, action_space_n=env.possible_actions)

for _ in range(500): # find target 500 times
    state = (2,0) # initial_state
    action = agent.get_action(state)
    while state not in env.target_grids:
        rewards_ls = []
        next_state = state
        next_action = action
        for _ in range(agent.step_num):
            next_state, reward = env.step(next_state, next_action)
            next_action = agent.get_action(next_state)
            rewards_ls.append(reward)
            if next_state in env.target_grids:
                break
        
        td_target = 0
        for reward in reversed(rewards_ls):
            td_target += reward
            td_target *= env.discounted_factor
        
        td_target += agent.q[next_state][next_action]
        # agent.update_q(state, action, reward, next_state, next_action, env.discounted_factor)
        agent.td_learn(state, action, td_target)
        agent.policy_improvement(state)
        state = next_state
        action = next_action


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