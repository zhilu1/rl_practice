"""
用 Bellman 公式计算精准的 state value 
(当然理论上是需要无穷多步才能达到, 但我们规定只要更新幅度小于 delta 就算精准)
"""
import numpy as np

def compute_state_value(height, width, env, policy, in_place=True, discount=1.0):
    new_state_values = np.zeros((height, width))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(height):
            for j in range(width):
                value = 0
                state = (i, j)
                for action in range(env.possible_actions):
                    (next_i, next_j), reward = env.step(state, action)
                    value += policy[state][action] * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration