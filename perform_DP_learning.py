"""
Be aware, this one is not used anymore, check value iteration solution and policy iteration solution
"""
# from pathlib import Path
# import sys



# from tools.helper import *
                
# from rl_envs.grid_world_env import GridWorldEnv
# from agents.value_iteration_agent import ValueIterationAgent
# from agents.policy_iteration_agent import TruncatedPolicyIterationAgent


# env = GridWorldEnv(5, 5, forbidden_grids=[(1,1),(1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=-1, hit_wall_reward=-1)
# # env = GridWorldEnv(2, 2, forbidden_grids=[(0,1)], target_grids=[(1,1)])


# def print_actions(agent, env):
#     for i in range(env.height):
#         print("[", end=" ")
#         for j in range(env.width):
#             action = agent.get_action((i,j))
#             print(env.action_mappings[action], end=" ")
#         print("]")


# """
# value iteration play
# """
# agent = ValueIterationAgent(action_space_n=env.action_n, discounted_factor=0.9, threshold=0.0001)

# amount_update = float('inf')
# while agent.not_converged(amount_update):
#     amount_update = 0
#     for i in range(env.height):
#         for j in range(env.width):
#             state = (i,j)
#             # action = agent.get_action(state)
#             for action in range(env.valid_actions(state)):
#                 next_state, imm_reward = env.step(state, action)
#                 agent.q_table_update(state, action, imm_reward, next_state)
#             agent.policy_update(state)
#             amount_update = max(amount_update,agent.value_update(state))
#     agent.old_v = agent.v.copy()

# print_actions(agent, env)
# print()
# print(agent.v)
# print()

# """
# model based policy iteration play
# """
# # agent = TruncatedPolicyIterationAgent(action_space_n=env.action_n, discounted_factor=0.9, threshold=0.001)

# # env.init_model_based_transitions()
# # agent.initialize_policy()

# # amount_update = float('inf')
# # while agent.not_converged(amount_update):
# #     amount_update = 0

# #     # policy evaluation
# #     for _ in range(agent.truncate_time):
# #         for i in range(env.height):
# #             for j in range(env.width):
# #                 state = (i,j)
# #                 amount = agent.policy_evaluation(state, env.expected_rewards, env.transition_probs)
# #                 amount_update = max(amount_update, amount)
# #         agent.old_v = agent.v.copy()
# #     # policy improvement
# #     for i in range(env.height):
# #         for j in range(env.width):
# #             state = (i,j)
# #             agent.policy_improvement(state, env.expected_rewards, env.transition_probs)
    

# """
# model free policy iteration play
# """
# # agent = TruncatedPolicyIterationAgent(action_space_n=env.action_n, discounted_factor=0.9, threshold=0.0001)

# # agent.initialize_policy()

# # amount_update = float('inf')
# # while agent.not_converged(amount_update):
# #     amount_update = 0
# #     for _ in range(agent.truncate_time):
# #         for i in range(env.height):
# #             for j in range(env.width):
# #                 state = (i,j)
# #                 agent.v[state] = 0
# #                 for action, action_prob in agent.policy[state].items():
# #                     if action_prob <= 0:
# #                         continue
# #                     next_state, immediate_reward = env.step(state, action)
# #                     s_update = agent.v_update(state, action_prob, immediate_reward, next_state)
# #                     amount_update = max(s_update, amount_update)
# #         agent.old_v = agent.v.copy()

# #     for i in range(env.height):
# #         for j in range(env.width):
# #             state = (i,j)
# #             agent.q[state] *= 0
# #             for action in range(env.valid_actions(state)):
# #                 next_state, immediate_reward = env.step(state, action)
# #                 agent.q_table_update(state, action, immediate_reward, next_state)
# #             agent.policy_update(state)
    

# """
# output
# """
# print(env)

# print_actions(agent, env)

# # for i in range(env.height):
# #     print("[", end=" ")
# #     for j in range(env.width):
# #         print('{:3f}'.format(agent.v[(i,j)]), end=" ")
# #     print("]")

# print_by_dict(env, agent.v)

# V, _ = compute_state_value(env.height, env.width, env, agent.policy)

# print_by_dict(env, V)
