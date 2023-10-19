from pathlib import Path
import sys
import numpy as np
from collections import defaultdict
from tools.helper import *


# from rl_envs.grid_world_env import GridWorldEnv
from agents.MC_epsilon_greedy import MC_epsilon_greedy as AgentAlgo
from rl_envs.episodic_grid_world_env import EpisodicGridWorldEnv as GridWorldEnv

# env = GridWorldEnv (3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)], forbidden_reward=-1, hit_wall_reward=-1,  target_reward=1)
env = GridWorldEnv (5, 5, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)],
                   discounted_factor=1 , forbidden_reward=-1, hit_wall_reward=-1, target_reward=5)

"""
如果说，在一个 episode 中，最后得到的 return 是负数，那么一路下来的 q (s,a) 就都是负数。那么会导致得到的算得很糟

当 forbidden 设的很大时，那么 return 是负数的几率也很大，尤其是当我们到达 target 就停止时，很容易就变成负的

尤其当我们比起 target 更容易进入 forbidden 时，在早期的随机的行动容易使我们的 return 变成负数，而后面不随机时会训练困难

同时，由于我们撞墙的可能性很高，所以导致撞墙的负数也阻碍训练

而 epsilon 较大时，为什么反而能更好地训练呢？
"""
# env = GridWorldEnv (3, 4, forbidden_grids=[(2,1), (1,3)], target_grids=[(2,3)])

# def print_actions (agent, env, get_optimal = False):
#     # action_mapping = ["↓","↑","→","←","↺"]
#     for i in range (env.height):
#         print ("[", end=" ")
#         for j in range (env.width):
#             action = agent.get_action ((i,j), get_optimal)
#             print (env.action_mappings [action], end=" ")
#         print ("]")


agent = AgentAlgo (epsilon=0.5, action_space_n=env.action_n)
agent.initialize_policy ()
# agent.initialize_policy (initial_policy = defaultdict (lambda: {4: 1}))
# # agent.initialize_policy (defaultdict (lambda: {default_action: 1}))

# # episode generation input env, generate an episode
# """
# every strategy

# exploring start
# """
episode_length = 100

# # for _ in range (10000):
# for i in range (env.height):
#     for j in range (env.width):
#         state = (i,j)
#         for action in range (env.valid_actions (state)):
#             episode = []
#             # while state not in env.target_grids: # 到 target 就停止
#             reward = 0
#             for _ in range (episode_length): # 循环 到 target 就停止
#                 next_state, reward, done = env.step (state, action)
#                 episode.append ((state, action, reward))
#                 if done:
#                     break
#                 state = next_state
#                 action = agent.get_action (state) # get policy from an action

#             # Find all (state, action) pairs we've visited in this episode
#             # We convert each state to a tuple so that we can use it as a dict key
#             sa_in_episode = set ([(tuple (x [0]), x [1]) for x in episode])
#             for state, action in sa_in_episode:
#                 sa_pair = (state, action)
#                 # Find the first occurance of the (state, action) pair in the episode
#                 first_occurence_idx = next (i for i,x in enumerate (episode)
#                                         if x [0] == state and x [1] == action)
#                 # Sum up all rewards since the first occurance
#                 G = sum ([x [2]*(env.discounted_factor**i) for i,x in enumerate (episode [first_occurence_idx:])])
#                 # Calculate average return for this state over all sampled episodes
#                 agent.state_action_returns [sa_pair] += G
#                 agent.state_action_nums [sa_pair] += 1.0
#                 agent.q [state][action] = agent.state_action_returns [sa_pair] /agent.state_action_nums [sa_pair]
#             agent.policy_improvement (env.height,env.width)
            

# """                # end episode collection, start evaluation
#                 cumulated_return = 0
#                 for epi_step in reversed (episode):
#                     state, action, reward = epi_step
#                     cumulated_return = env.discounted_factor * cumulated_return + reward
#                     state_action = (state, action)
#                     agent.state_action_returns [state_action] = agent.state_action_returns [state_action] + cumulated_return
#                     agent.state_action_nums [state_action] = agent.state_action_nums [state_action] + 1
#                     # policy evaluation
#                     avg_return = agent.state_action_returns [state_action] /agent.state_action_nums [state_action]
#                     agent.q [state][action] = avg_return
#                     # policy improvement
#                     agent.policy_improvement (state)"""



# """
# output
# """
# print (env)

# print_actions (agent, env, get_optimal = True)

# def get_state_value (Q, policy):
#     V = {}
#     for state in Q.keys ():
#         V [state] = sum (policy [state] * Q [state])
#     return V

# V = get_state_value (agent.q, agent.policy)

# print ()

# print_by_dict (env,V)

# """
# first occurance
# """
num_episodes = 100000
discount_factor = 0.9
for i_episode in range (1, num_episodes + 1):
    # Print out which episode we're on, useful for debugging.
    if i_episode % 1000 == 0:
        print ("\rEpisode {}/{}.".format (i_episode, num_episodes), end="")
        sys.stdout.flush ()

    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    state = env.reset ()
    for _ in range (episode_length): # 循环 到 target 就停止
        action = agent.get_action (state) # get policy from an action
        next_state, reward, done = env.step (state, action)
        episode.append ((state, action, reward))
        if done:
            break
        state = next_state

    
    sa_in_episode = set ([(tuple (x [0]), x [1]) for x in episode])
    for state, action in sa_in_episode:
        state_action = (state, action)
        # Find the first occurance of the (state, action) pair in the episode
        first_occurence_idx = next (i for i,x in enumerate (episode)
                                    if x [0] == state and x [1] == action)
        # Sum up all rewards since the first occurance
        cumulated_return = sum ([x [2]*(discount_factor**i) for i,x in enumerate (episode [first_occurence_idx:])])
        # Calculate average return for this state over all sampled episodes
        agent.state_action_returns [state_action] += cumulated_return
        agent.state_action_nums [state_action] = agent.state_action_nums [state_action] + 1
        # policy evaluation
        avg_return = agent.state_action_returns [state_action] /agent.state_action_nums [state_action]
        agent.q [state][action] = avg_return
        # policy improvement
    agent.policy_improvement (env.height,env.width)


# """
# output
# """
print (env)

print_actions (agent, env, get_optimal = True)

def get_state_value (Q, policy):
    V = {}
    for state in Q.keys ():
        V [state] = sum (policy [state] * Q [state])
    return V

V = get_state_value (agent.q, agent.policy)

print ()

print_by_dict (env,V)

plot_value_function (V, block=True)