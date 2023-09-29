from pathlib import Path
import sys

# path_root = Path(__file__).parents[2]
# sys.path.append(str(path_root))
# print(sys.path)

# agent 的代码看不出什么错误
# 想想, 关键是在终点无限等待的收益太高了, 所以无论如何都是直接冲终点赚
# 目前有两个道路, 一是实现 policy iteration 和 truncated policy iteration, 看看是否一致
# 此外还应该仔细看看老师的 setting, 是否是到了终点能停止, 然后做一个到终点后停止的 environment
                
from rl_envs.grid_world_env import GridWorldEnv
from agents.value_iteration_agent import ValueIterationAgent
from agents.policy_iteration_agent import TruncatedPolicyIterationAgent

env = GridWorldEnv(3, 4, forbidden_grids=[(1,1), (1,3)], target_grids=[(2,3)])


def print_actions():
    action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i,j))
            print(action_mapping[action], end=" ")
        print("]")


"""
value iteration play
"""
# agent = ValueIterationAgent(action_space_n=env.possible_actions, discounted_factor=0.9, threshold=0.0001)

# amount_update = float('inf')
# while agent.not_converged(amount_update):
#     print_actions()
#     print()
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


"""
model based policy iteration play
"""
agent = TruncatedPolicyIterationAgent(action_space_n=env.possible_actions, discounted_factor=0.9, threshold=0.001)

env.model_based_transitions()
agent.initialize_policy()

amount_update = float('inf')
while agent.not_converged(amount_update):
    amount_update = 0

    # policy evaluation
    for _ in range(agent.truncate_time):
        for i in range(env.height):
            for j in range(env.width):
                state = (i,j)
                amount = agent.policy_evaluation(state, env.expected_rewards, env.transition_probs)
                amount_update = max(amount_update, amount)
        agent.old_v = agent.v.copy()
    # policy improvement
    for i in range(env.height):
        for j in range(env.width):
            state = (i,j)
            agent.policy_improvement(state, env.expected_rewards, env.transition_probs)
    

"""
model free policy iteration play
"""
# agent = TruncatedPolicyIterationAgent(action_space_n=env.possible_actions, discounted_factor=0.9, threshold=0.0001)


# amount_update = float('inf')
# while agent.not_converged(amount_update):
#     amount_update = 0
#     for _ in range(agent.truncate_time):
#         for i in range(env.height):
#             for j in range(env.width):
#                 state = (i,j)
#                 agent.policy_evaluation(state, env.expected_rewards, env.transition_probs)



"""
output
"""
print(env)

print_actions()

for i in range(env.height):
    print("[", end=" ")
    for j in range(env.width):
        print('{:3f}'.format(agent.v[(i,j)]), end=" ")
    print("]")