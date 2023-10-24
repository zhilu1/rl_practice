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


class DeepQLearningAgent:
    def __init__(
        self,
        input_dim=3,
        output_dim=1,
        action_space=5,
        state_space = (5,5),
        lr: float = 0.01,
        TAU=0.005,
        discounted_factor=0.99,
        hidden_dim=100,
    ) -> None:
        self.output_dim = output_dim
        self.action_space = action_space
        self.state_space = state_space

        self.policy_net = self.initialize_network(input_dim, hidden_dim, output_dim)
        self.target_net = self.initialize_network(input_dim, hidden_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())


        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.behavior_policy = defaultdict(
            lambda: np.ones(self.action_space) * (1 / self.action_space)
        )
        self.TAU = TAU
        self.discounted_factor = discounted_factor
        self.criterion = torch.nn.SmoothL1Loss()
        self._policy_table = defaultdict(lambda: [0.0 for _ in range(self.action_space-1)] + [1.0])
        # self.target_net = torch.

    def initialize_network(self, in_dim, hidden_dim, out_dim, q_net=None):
        # `in_feature` input feature dim depends on encoding  of (state, action) pair

        self.QNetStruct = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=out_dim),
            # nn.Linear(in_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim,out_dim)
        )
        return self.QNetStruct

    def state_normalize(self, state):
        # normalize each to [0,1]
        return (state[0] / (self.state_space[0] - 1), state[1] / (self.state_space[1] - 1))
    def sa_normalize(self, sa_pair):
        res = (sa_pair[0] / (self.state_space[0] - 1), sa_pair[1] / (self.state_space[1] - 1), sa_pair[2] / (self.action_space - 1))
        return res
    def call_policy_net(self, batched_state_action):
        """
        preprocess the data and call the policy_network
        """
        # modified_batch = torch.zeros_like(batched_state_action)
        # for i, state_action in enumerate(batched_state_action):
        #     modified_batch[i] = torch.tensor(self.sa_normalize(state_action))
        return self.policy_net(batched_state_action)
    def call_target_net(self, batched_state_action):
        """
        preprocess the data and call the target_network
        """
        # modified_batch = torch.zeros_like(batched_state_action)
        # for i, state_action in enumerate(batched_state_action):
        #     modified_batch[i] = torch.tensor(self.sa_normalize(state_action))
        return self.target_net(batched_state_action)

    def get_behavior_action(self, state):
        return np.random.choice(len(self.behavior_policy[state]), 1, p=self.behavior_policy[state])[0]  # random choose an action based on policy
    def get_action(self, raw_state, optimal=True):
        best_action = 0
        max_q = -float("inf")
        state = torch.tensor(raw_state, dtype=torch.float).unsqueeze(0)
        for action_ind in range(self.action_space):
            action = torch.tensor(action_ind, dtype=torch.float).view(-1, 1)
            sa_pair = torch.cat([state, action], dim=1)
            
            q_value = self.call_policy_net(sa_pair)
            # q_value = self.policy_net(sa_pair)
            if q_value >= max_q:
                best_action = action_ind
                max_q = q_value
            
            self._policy_table[raw_state][action_ind] = 0.0
        self._policy_table[raw_state][best_action] = 1.0
        return best_action

    def loss(self, inp, target):
        return ((inp - target) ** 2) / target.size(0)

    def update_Q_network(self, state_action, reward, next_state):


        # q_value_target = torch.empty((batch_size, 0))  # 定义空的张量
        # for action in range(self.action_space):
        #     s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
        #     q_value_target = torch.cat((q_value_target, self.target_net(s_a)), dim=1)
        # q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]
        # target_value = (reward + self.discounted_factor * q_star).squeeze()
        """
        计算 TD-target, 因为 batch 的存在, 需要做一些维度上的操作
        WARNING: 错误看起来就是在这里这部分, 稍等之前我似乎是忘记把这里的action normalize 了?
        问题找到了: 此前 next_state.repeat(5, 1) 是整个一块地复制, 导致 reshape 时的 连续 5 行 state 是不同的
        [[1,2],
         [3,4]] 
        被复制成
        [[1,2],
        [3,4],
        [1,2],
        [3,4]] 
        而我实际想要的是 
        [[1,2],
        [1,2],
        [3,4],
        [3,4]]
        """

        # 更新 self 的 main network
        batch_size = 100

        action_tiles = torch.arange(5).reshape(5, 1).repeat(batch_size, 1)
        # for action in range(self.action_space):
        next_sa = torch.cat(
            [torch.repeat_interleave(next_state, repeats=5, dim=0), action_tiles], dim=1
        )
        target_output = self.call_target_net(next_sa)
        # 注意这里 max 是 Qlearn 和 Sarsa 的关键区别
        target_q = torch.max(
            target_output.reshape(-1, 5), dim=1, keepdim=True
        ).values  #
        target_value = self.discounted_factor * target_q + reward

        # STABLE-BASELINES3 target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # 计算 q(s,a,w) 的估计
        # sa_pair = torch.cat([state, action_indices.unsqueeze(1)], dim=1)
        q_value = self.call_policy_net(state_action)
        # q_value = self.policy_net(state_action)
        # q_value = output[torch.arange(output.size(0)), action_indices] # q value of (state, action)

        # minimize distance between policy result and target value, the loss choose can be ..
        # criterion = torch.nn.SmoothL1Loss()
        # # criterion = torch.nn.HuberLoss()
        self.optimizer.zero_grad()
        loss = self.criterion(q_value, target_value)
        # loss = criterion(q_value, target_value)
        loss.backward()

        # loss = self.loss(q_value.squeeze(), target_value)
        # loss.sum().backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss, q_value, target_value

    def sync_target_network(self):
        with torch.no_grad():
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def generate_Q_table(self, height, width):
        """
        Generate a 'Q table' for debug use
        DQN don't really need a Q table
        """
        Q = {}
        for y in range(height):
            for x in range(width):
                state = torch.tensor((y,x), dtype=torch.float).unsqueeze(0)
                q_values = []
                for action_ind in range(self.action_space):
                    action = torch.tensor(action_ind, dtype=torch.float).view(-1, 1)
                    sa_pair = torch.cat([state, action], dim=1)
                    q_value = self.call_policy_net(sa_pair)
                    q_values.append(q_value.item())
                Q[(y,x)] = q_values
        return Q