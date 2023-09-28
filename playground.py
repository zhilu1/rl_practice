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
from agents.value_iteration_agent import ModelBasedValueIterationAgent

env = GridWorldEnv(3, 4, forbidden_grids=[(1,1), (1,3)], target_grids=[(2,3)])

agent = ModelBasedValueIterationAgent(action_space_n=env.possible_actions, gamma=0.9, threshold=0.0001)

def print_actions():
    action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i,j))
            print(action_mapping[action], end=" ")
        print("]")

amount_update = float('inf')
while agent.not_converged(amount_update):
    print_actions()
    print()
    amount_update = 0
    for i in range(env.height):
        for j in range(env.width):
            state = (i,j)
            # action = agent.get_action(state)
            for action in range(env.valid_actions(state)):
                next_state, imm_reward = env.step(state, action)
                agent.q_table_update(state, action, imm_reward, next_state)
            agent.policy_update(state)
            amount_update = max(amount_update,agent.value_update(state))
    agent.old_v = agent.v.copy()
    
            # state = env.get_obs(i, j)
print(env)

print_actions()

for i in range(env.height):
    print("[", end=" ")
    for j in range(env.width):
        print('{:3f}'.format(agent.v[(i,j)]), end=" ")
    print("]")