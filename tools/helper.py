import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
import gymnasium as gym
from rl_envs.gym_grid_world_env import GridWorldEnv


def plot_value_function(V, title="Value Function", block=False):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[1] for k in V.keys())
    max_x = max(k[1] for k in V.keys())
    min_y = min(k[0] for k in V.keys())
    max_y = max(k[0] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z = np.apply_along_axis(lambda _: V[(_[0], _[1])], 2, np.dstack([X, Y]))
    # Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(121, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="GnBu")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("State Value")
        ax.set_title(title)
        ax.view_init(75, 0)
        fig.colorbar(surf)

        ax = fig.add_subplot(122, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="GnBu")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("State Value")
        ax.set_title(title)
        ax.view_init()

        # ax = fig.add_subplot(2, 2, 3, projection='3d')
        # ax.plot_wireframe(X, Y, Z)

        # ax = fig.add_subplot(224, projection='3d')
        # ax.voxels(X, Y, Z, filled)
        # ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
        # ax.set_aspect('equal')
        plt.show(block=block)

    # plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z, "{} (Grid State )".format(title))


def print_actions(agent, env, get_optimal=False):
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i, j), get_optimal)
            print(env.action_mappings[action], end=" ")
        print("]")


def print_actions_policy(policy, env):
    index = 0
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            state = (i, j)
            action = np.argmax(policy(state))
            index += 1
            print(env.action_mappings[action], end=" ")
        print("]")


def print_actions_index(agent, env):
    index = 0
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action(index)
            print(env.action_mappings[action], end=" ")
            index += 1
        print("]")


def print_by_dict(env, dic):
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            if isinstance(dic[(i, j)], float):
                print("%.2f" % dic[(i, j)], end=" ")
            else:
                print(dic[(i, j)], end=" ")
        print("]")


def visualize_in_gym(agent, env_name="", inp_env=None, steps=1000):
    """
    render a environment and visualize it running N steps

    another possible solution can be gymnasium.experimental.wrappers.RecordVideoV0
    """
    if inp_env:
        demo_env = inp_env
    else:
        demo_env = gym.make(env_name, render_mode="human")
    observation, info = demo_env.reset()

    for _ in range(steps):
        action = agent.get_action(
            observation
        )  # agent policy that uses the observation and info
        # insert an algorithm that can interact with env and output an action here
        observation, reward, terminated, truncated, info = demo_env.step(action)
        if terminated or truncated:
            observation, info = demo_env.reset()

    if not inp_env:
        demo_env.close()

def compute_state_value(height, width, env: GridWorldEnv, policy, in_place=True, discount=0.9):
    """
    用 Bellman 公式计算精准的 state value 
    (当然理论上是需要无穷多步才能达到, 但我们规定只要更新幅度小于 delta 就算精准)
    """
    env.initialize_model_based()
    new_state_values = np.zeros((height, width))
    iteration = 0
    while iteration < 1000:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(height):
            for j in range(width):
                state = (i, j)
                value = 0
                for action in range(env.action_n):
                    q_val = 0
                    for prob, reward in env.Prsa[state][action]:
                        q_val += reward * prob
                    for prob, next_state in env.Pssa[state][action]:
                        q_val += discount * (old_state_values[next_state[0],next_state[1]] * prob)
                    value += policy[state][action] * q_val
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-5:
            break

        iteration += 1

    return new_state_values, iteration

def gridworld_demo(agent, forbidden_reward=-1, hit_wall_reward=-1, target_reward=1):
    env = GridWorldEnv(fixed_map = True, render_mode="human", forbidden_grids=[(1,1), (1,2), (2,2),(3,1),(3,3),(4,1)], target_grids=[(3,2)], forbidden_reward=forbidden_reward, hit_wall_reward=hit_wall_reward, target_reward=target_reward)
    obs, _ = env.reset()
    total_reward = 0
    routine = [obs['agent']]
    for i in range(500):
        obs = tuple(obs['agent'])
        action = agent.get_action(obs, optimal=True)
        obs, reward, terminated, truncated, info  = env.step(action)
        # VecEnv resets automatically
        total_reward += reward
        routine.append(obs['agent'])
        if terminated or truncated:
            obs, _ = env.reset()
            print('reward: {}, distance: {}'.format(total_reward, routine))
            total_reward = 0
            routine = [obs['agent']]
            if truncated:
                print("TRUNCATE")
            else:
                print("TERMINATE")
    env.close()