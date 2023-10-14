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
                 input_dim,
                 output_dim,
                 action_space,
                 lr: float = 0.01,
                 TAU =  0.005,
                 discounted_factor = 0.99,
                 hidden_dim = 100
                 ) -> None:
        self.output_dim = output_dim
        self.action_space = action_space
        self.policy_net = self.initialize_network(input_dim, hidden_dim, output_dim)
        self.target_net = self.initialize_network(input_dim, hidden_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.behavior_policy = defaultdict(lambda: np.ones(self.action_space) * (1/self.action_space))
        self.TAU = TAU
        self.discounted_factor = discounted_factor

        # self.target_net = torch.
    def initialize_network(self, in_feature, hidden_dim, out_dim, q_net = None):
        # `in_feature` input feature dim depends on encoding  of (state, action) pair

        self.QNetStruct = nn.Sequential(
                    nn.Linear(in_feature, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim,out_dim)
                )
        return self.QNetStruct
    def state_normalize(self, state, height, width):
        # normalize each to [0,1]
        return (state[0]/(height-1),state[1]/(width-1))
    
    def get_behavior_acion(self, state):
        return np.random.choice(len(self.behavior_policy[state]),1,p=self.behavior_policy[state])[0] # random choose an action based on policy

    def get_action(self, state, optimal=True):
        best_action = 0
        max_q = - float('inf')
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        for action_ind in range(self.action_space):
            action = torch.tensor(action_ind/(self.action_space-1), dtype=torch.float).view(-1,1)
            sa_pair = torch.cat([state,action], dim=1)
            q_value = self.policy_net(sa_pair)
            if q_value >= max_q:
                best_action = action_ind
                max_q = q_value
        return best_action
    
    def loss(self, inp, target):
        return (inp-target) ** 2
    def update_Q_network(self, state, action_indices, reward, next_state):
        # 更新 self 的 main network
        action_tiles = torch.arange(5).reshape(5, 1).repeat(100,1)
        next_sa = torch.cat([next_state.repeat(5,1),action_tiles], dim=1)
        target_output = self.target_net(next_sa)
        target_q = torch.max(target_output.reshape(-1,5), dim=1).values
        # target_q = max(target_q, self.target_net(next_sa).item()) # torch.max 在 dim 指定时似乎才会有 values 和 indices 来表示
        target_value = self.discounted_factor * target_q + reward
        # STABLE-BASELINES3 target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # minimize distance between policy result and target value, the loss choose can be ..
        self.optimizer.zero_grad()
        sa_pair = torch.cat([state, action_indices.unsqueeze(1)], dim=1)
        q_value = self.policy_net(sa_pair)
        # q_value = output[torch.arange(output.size(0)), action_indices] # q value of (state, action)
        # criterion = torch.nn.SmoothL1Loss()
        # loss = criterion(q_value, target_value)
        # loss.backward()
        loss = self.loss(q_value.squeeze(), target_value)
        loss.sum().backward()
        # torch.nn.utils.clip_grad.clip_grad_norm_(self.policy_net.parameters(), 10)
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

        # target_net_state_dict = self.target_net.state_dict()
        # policy_net_state_dict = self.policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        # self.target_net.load_state_dict(target_net_state_dict)
        # return self.QNetStruct
    # def encoding_state_action(self, state, action):
        # state 和 action, 假如都是离散的, 可以直接 one_hot encoding
        # 假如都是连续的, 那么就直接输入给神经网络似乎也可以
        # 考虑到 grid world 环境 state 是有上下关系的, 那么可以 state 直接作为有顺序的数字 encode, action 则 one_hot encoding
        # 后注: 上面的想法都不必要, 实际上用 state 输入 而非 (s,a) pair, 然后同时输出多个 action 的 action value

