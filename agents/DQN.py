"""
一方面是 value approximation,
另一方面是 神经网络的构建

可以使用最基础的 shallow net? 但我还不想用 torch 的话, 我要怎么手写 反向传播的梯度计算? 自己手算求导?

DQN 的不成功, 我们可以从两个方面旁敲侧击
1. 使用 DQN 到 Gym 的其他env中实验
2. 尝试 value approximation of Q learning, 但是使用线性方程而非神经网络来拟合 q function
或者干脆先去写 policy gradient
"""
from torch import nn
from torch import optim
from collections import defaultdict
import numpy as np
import torch
from itertools import zip_longest

class DeepQLearningAgent:
    def __init__(self,
                 state_space_n,
                 action_space_n,
                 lr: float = 0.01,
                 TAU =  0.005,
                 discounted_factor = 0.99,
                 hidden_dim = 100
                 ) -> None:
        self.action_space_n = action_space_n
        self.policy_net = self.initialize_network(state_space_n, hidden_dim, action_space_n)
        self.target_net = self.initialize_network(state_space_n, hidden_dim, action_space_n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.behavior_policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))
        self.TAU = TAU
        self.discounted_factor = discounted_factor

        # self.target_net = torch.
    def initialize_network(self, in_feature, hidden_dim, out_dim, q_net = None):
        # `in_feature` input feature dim depends on encoding  of (state, action) pair
        if q_net:
            self.QNetStruct = q_net
        else:
            self.QNetStruct = nn.Sequential(
                        nn.Linear(in_feature, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim,out_dim)
                    )
        return self.QNetStruct

    def get_behavior_acion(self, state):
        return np.random.choice(len(self.behavior_policy[state]),1,p=self.behavior_policy[state])[0] # random choose an action based on policy
    def get_action(self, state):
        return np.argmax(self.policy_net(state))
    def loss(self, inp, target):
        return (inp-target) ** 2
    def update_Q_network(self, state, action_indices, reward, next_state):
        # 更新 self 的 main network
        target_q = torch.max(self.target_net(next_state), dim=1).values # torch.max 在 dim 指定时似乎才会有 values 和 indices 来表示
        target_value = self.discounted_factor * target_q + reward
        # STABLE-BASELINES3 target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # minimize distance between policy result and target value, the loss choose can be ..
        self.optimizer.zero_grad()
        output = self.policy_net(state)
        q_value = output[torch.arange(output.size(0)), action_indices] # q value of (state, action)
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(q_value, target_value)
        loss.backward()
            # Compute Huber loss (less sensitive to outliers)

            # Clip gradient norm
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        return loss, q_value, target_value

    def sync_target_network(self):
        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for param, target_param in self.zip_strict(self.policy_net.parameters(), self.target_net.parameters()):
                target_param.data.mul_(1 - self.TAU)
                torch.add(target_param.data, param.data, alpha=self.TAU, out=target_param.data)

    def zip_strict(self,*iterables):
        r"""
        ``zip()`` function but enforces that iterables are of equal length.
        Raises ``ValueError`` if iterables not of equal length.
        Code inspired by Stackoverflow answer for question #32954486.

        :param \*iterables: iterables to ``zip()``
        """
        # As in Stackoverflow #32954486, use
        # new object for "empty" in case we have
        # Nones in iterable.
        sentinel = object()
        for combo in zip_longest(*iterables, fillvalue=sentinel):
            if sentinel in combo:
                raise ValueError("Iterables have different lengths")
            yield combo

    def generate_Q_table(self, height, width):
        Q = {}
        for y in range(height):
            for x in range(width):
                q_values = self.policy_net((y,x))
                Q[(y,x)] = q_values
        return Q