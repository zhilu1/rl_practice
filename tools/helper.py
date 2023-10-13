import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym

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
        ax = fig.add_subplot(121, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='GnBu')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('State Value')
        ax.set_title(title)
        ax.view_init(75,0)
        fig.colorbar(surf)

        ax = fig.add_subplot(122, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='GnBu')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('State Value')
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


def print_actions(agent, env, get_optimal = False):
    # action_mapping = [" ↓ "," ↑ "," → "," ← "," ↺ "]
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            action = agent.get_action((i,j), get_optimal)
            print(env.action_mappings[action], end=" ")
        print("]")
def print_actions_policy(policy, env):
    index = 0
    for i in range(env.height):
        print("[", end=" ")
        for j in range(env.width):
            state = (i,j)
            action = np.argmax(policy(state))
            index+=1
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
            print("%.2f" %dic[(i,j)], end=" ")
        print("]")

def visualize_in_gym(agent, env_name =  "", inp_env = None, steps=1000):
    """
    render a environment and visualize it running N steps

    another possible solution can be gymnasium.experimental.wrappers.RecordVideoV0
    """
    if inp_env:
        demo_env = inp_env
    else:
        demo_env = gym.make(env_name, render_mode = "human")
    observation, info = demo_env.reset()

    for _ in range(steps):
        action = agent.get_action(observation)  # agent policy that uses the observation and info
        # insert an algorithm that can interact with env and output an action here
        observation, reward, terminated, truncated, info = demo_env.step(action)
        if terminated or truncated:
            observation, info = demo_env.reset()

    if not inp_env:
        demo_env.close()