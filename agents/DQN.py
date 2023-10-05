from torch import nn
from torch import optim
from collections import defaultdict
import numpy as np

class DeepQLearningAgent:
    def __init__(self,
                 state_space_n,
                 action_space_n,
                 lr: float = 0.01,
                 ) -> None:
        self.action_space_n = action_space_n
        self.policy_net = self.initialize_network(state_space_n, 128, action_space_n)
        self.target_net = self.initialize_network(state_space_n, 128, action_space_n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.behavior_policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))

        # self.target_net = torch.
    def initialize_network(self, in_feature = 2, hidden_dim = 16, out_dim = 1, q_net = None):
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
        # return self.QNetStruct
    # def encoding_state_action(self, state, action):
        # state 和 action, 假如都是离散的, 可以直接 one_hot encoding
        # 假如都是连续的, 那么就直接输入给神经网络似乎也可以
        # 考虑到 grid world 环境 state 是有上下关系的, 那么可以 state 直接作为有顺序的数字 encode, action 则 one_hot encoding
        # 后注: 上面的想法都不必要, 实际上用 state 输入 而非 (s,a) pair, 然后同时输出多个 action 的 action value

