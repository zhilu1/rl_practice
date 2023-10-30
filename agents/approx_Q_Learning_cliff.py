"""
用线性基函数模型来代替 tabular Qlearning 中的 Q-table 来拟合估计 action value

其中基函数 (basis function) 使用 the Fourier function of order 5 (参考 "Mathematical Foundations of Reinforcement Learning" 书上 P.176 和 P.188)

"""

from collections import defaultdict
import numpy as np
import torch
from rl_envs.new_gym_grid_world_env import GridWorldEnv
# import math
from math import cos, pi
from torch.utils.tensorboard import SummaryWriter # type: ignore
import itertools
from random import shuffle
class CliffApproxQLearningAgent:
    def __init__(
        self, action_space_n, epsilon=0.1, learning_rate=1e-4, discounted_factor=0.9
    ) -> None:
        self.action_space_n = action_space_n
        self.epsilon = epsilon
        self.discounted_factor = discounted_factor
        # self.policy = defaultdict(lambda: np.zeros(self.action_space_n))
        self.policy = defaultdict(
            lambda: np.ones(self.action_space_n) * (1 / self.action_space_n)
        )
        self.learning_rate = learning_rate  # could depend on t, but never mind
        self.parameters = np.array([0. for _ in range(25)]) # 2^3, 3^3, ...
        self.phi_ls = np.array(list(itertools.product(range(5), repeat=2))) 

    # def phi_sa(self, state, action, i):
    #     x, y, z = *state, action
    #     # the linear feature vector is selected as the Fourier function of order 5 (参考书上 P.176 和 P.188)
    #     c_1, c_2, c_3 = self.phi_ls[i]
    #     phi = cos(c_1*x + c_2*y + c_3*z)
    #     # φ(s, a) = [1, cosx, cosy, cosz, cos(x+y), cos(x+z), cos(y+z), cos(y+z), cos(x+y+z), cos(2x),... ] * parameters
    #     return phi

    def estimate_q(self, state, action):
        s, z = state, action
        # s, z = (state)/96, (action)/6
        phi_mat = np.cos((self.phi_ls @ np.array([s,z]))*pi)
        q = phi_mat @ self.parameters
        # for i, param in enumerate(self.parameters):
        #     q += self.phi_sa(state, action, i) * param
        return q

    def update_parameters(self, TD_error, state, action, next_state, best_action):
        """
        随机梯度下降法.
        """

        s, z = state, action
        # s, z = (state)/96, (action)/6
        phi_mat = np.cos((self.phi_ls @ np.array([s,z]))*pi)
        s, z = next_state, best_action
        # s, z = (next_state)/96, (best_action)/6
        # x, y, z = *next_state, best_action
        next_phi_mat = np.cos((self.phi_ls @ np.array([s,z]))*pi)
        gradient = next_phi_mat* self.discounted_factor - phi_mat

        # x, y, z = *state, action
        # phi_mat = np.cos(self.phi_ls @ np.array([x,y,z]))
        # gradient = phi_mat
        self.parameters -= self.learning_rate * (2 * TD_error * gradient)
        # for i, param in enumerate(self.parameters):
        #     # gradient = self.phi_sa(
        #     #     next_state, best_action, i
        #     # ) * self.discounted_factor - self.phi_sa(state, action, i)
        #     gradient = self.phi_sa(state, action, i)
        #     self.parameters[i] += self.learning_rate * (2 * TD_error * gradient)

    def policy_improvement(self, state):
        """
        利用函数估计 Q 值, 从而建立 epsilon-greedy policy
        """
        best_action = 0
        best_q_value = -float("inf")
        for action in range(self.action_space_n):
            q_value = self.estimate_q(state, action)
            self.policy[state][action] = self.epsilon / self.action_space_n
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        self.policy[state][best_action] = (
            1 - ((self.action_space_n - 1) / self.action_space_n) * self.epsilon
        )

    def get_action(self, state, optimal=False):
        if optimal:
            return np.argmax(self.policy[state])
        return np.random.choice(len(self.policy[state]), 1, p=self.policy[state])[
            0
        ]  # random choose an action based on policy
    def estimate_state(self, state):
        q_values = []
        q_sum = 0
        for action in range(self.action_space_n):
            q_value = self.estimate_q(state, action)
            q_values.append(q_value)
            q_sum += q_value
        action_probs = self.softmax(q_values)
        return q_values, action_probs
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def RUN(self, env: GridWorldEnv, num_episodes=1000, episode_len=100000):
        writer = SummaryWriter()
        """
        collect replay experience
        """
        # for _ in range(episode_len):
        #     state = next_state
        #     action = self.action_space_n.sample()
        #     next_state, reward, terminated, truncated, info = env.step(action)
        #     episode_recorder.append(
        #         (
        #             tuple(state),
        #             action,
        #             reward,
        #             tuple(next_state),
        #         )
        #     )
        """
        # input normalization                 
        (state[0] / float(env.height), state[1] / float(env.width)),
        action / env.action_n,
        reward,
        (
            next_state[0] / float(env.height),
            next_state[1] / float(env.width),
        ),"""

        episode_rewards = defaultdict(float)
        TD_error = 0.
        t = 0
        for i_episode in range(num_episodes):
            episode_recorder = []
            next_state, _ = env.reset()
            # 首先, 根据 policy 生成 episode
            for t in itertools.count():
                state = next_state
                """
                把这里改成 用 q_value 计算 action probs, 然后用这个 probs 而非 epsilon greedy 来操作
                """
                qs, action_probs = self.estimate_state(state)
                action = np.random.choice(len(action_probs), 1, p=action_probs)[0]
                # action = np.argmax(action_probs)
                # action = self.get_action(state)  
                next_state, reward, terminated, truncated, info = env.step(action)
                if reward == -100:
                    next_state = 37
                    truncated = True
                episode_recorder.append((state, action, reward, next_state))
                # Update statistics
                episode_rewards[i_episode] += reward
                if t % 1000 == 0:
                    print("\rStep {} @ Episode {}/{} ({}, {})".format(
                        t, i_episode + 1, num_episodes, episode_rewards[i_episode], episode_rewards[i_episode - 1]))
                if terminated or truncated:
                    break
            # writer.add_scalar("reward", episode_rewards[i_episode], i_episode)
            # writer.add_scalar("episode length", t, i_episode)
            for i_step, (state, action, reward, next_state) in enumerate(
                episode_recorder
            ):
                q_value = self.estimate_q(state, action)
                next_q = -float("inf")
                best_action = 0
                for a in range(self.action_space_n):
                    q_eval = self.estimate_q(next_state, a)
                    if q_eval > next_q:
                        next_q = q_eval
                        best_action = a

                TD_error = reward + self.discounted_factor * next_q - q_value

                self.update_parameters(
                    TD_error, state, action, next_state, best_action
                )  # loss is MSE(TD_target, q_value)
                # writer.add_scalar(
                # "TD error", TD_error, i_episode*episode_len + i_step
                # )

            if i_episode % 100 == 0:
                print("\r len {} Episode {}/{} ( TD_error: {}, reward: {},{})".format(
                    t, i_episode, num_episodes, TD_error,  episode_rewards[i_episode], episode_rewards[i_episode - 1]))

            """
            在结束时更新 policy, 供 get_action 使用
            """
            # for y in range(env.height):
        for s in range(48):
            self.policy_improvement(s)
        writer.flush()
        writer.close()