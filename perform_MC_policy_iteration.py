from pathlib import Path
import sys
import numpy as np
from collections import defaultdict



from rl_envs.grid_world_env import GridWorldEnv
from agents.MC_epsilon_greedy import MC_epsilon_greedy as AgentAlgo

env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=0, hit_wall_reward=-1)

"""
如果说, 在一个 episode 中, 最后得到的 return 是负数, 那么一路下来的 q(s,a) 就都是负数. 那么会导致得到的算得很糟

当 forbidden 设的很大时, 那么 return 是负数的几率也很大, 尤其是当我们到达 target 就停止时, 很容易就变成负的

尤其当我们比起 target 更容易进入 forbidden 时, 在早期的随机的行动容易使我们的 return 变成负数, 而后面不随机时会训练困难

同时, 由于我们撞墙的可能性很高, 所以导致撞墙的负数也阻碍训练

而 epsilon 较大时, 为什么反而能更好地训练呢? 
"""
# env = GridWorldEnv(3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)])

def print_actions(agent, env, get_optimal = False):
    action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i,j), get_optimal)
            print(action_mapping[action], end=" ")
        print("]")


agent = AgentAlgo(epsilon=0, action_space_n=env.possible_actions)
agent.initialize_policy(defaultdict(lambda: {0: 1}))
# agent.initialize_policy(defaultdict(lambda: {default_action: 1}))
print_actions(agent, env, get_optimal = True)

# episode generation input env, generate an episode

# exploring start
for i in range(env.height):
    for j in range(env.width):
        state = (i,j)
        for action in range(env.valid_actions(state)):
            episode = []
            # while state not in env.target_grids: # 到 target 就停止
            for _ in range(500): # 循环不停
                next_state, reward = env.step(state, action)
                episode.append((state, action, reward))
                state = next_state
                action = agent.get_action(state) # get policy from an action
            else:
                # end episode, start evaluation
                episode.append((state, action, reward))
                cumulated_return = 0
                for epi_step in reversed(episode):
                    state, action, reward = epi_step
                    cumulated_return = env.discounted_factor * cumulated_return + reward
                    state_action = (state, action)
                    agent.state_action_returns[state_action] = agent.state_action_returns[state_action] + cumulated_return
                    agent.state_action_nums[state_action] = agent.state_action_nums[state_action] + 1
                    # policy evaluation
                    avg_return = agent.state_action_returns[state_action] / agent.state_action_nums[state_action]
                    agent.q[state][action] = avg_return
                    # policy improvement
                    agent.policy_improvement(state)
            
                    

"""
output
"""
print(env)

print_actions(agent, env, get_optimal = True)
# print_optimal_actions(agent, env)

# for i in range(env.height):
#     print("[", end=" ")
#     for j in range(env.width):
#         print('{:3f}'.format(agent.v[(i,j)]), end=" ")
#     print("]")







