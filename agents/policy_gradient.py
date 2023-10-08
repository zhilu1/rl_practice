"""
Policy gradient using MLP
"""
import torch
from torch import nn
from torch import optim
from collections import defaultdict
import numpy as np

class PGAgent:
    def __init__(self,
                 state_space_n,
                 action_space_n,
                 lr: float = 0.01,
                 TAU = 0.5
                 ) -> None:
        self.action_space_n = action_space_n
        self.policy_net = self.initialize_network(state_space_n, 128, action_space_n)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.behavior_policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))
        self.TAU = TAU
        self.q = defaultdict(lambda: np.zeros(action_space_n))

        # self.target_net = torch.
    def initialize_network(self, in_feature, hidden_dim, out_dim):
        # `in_feature` input feature dim depends on encoding  of (state, action) pair

        self.net_struct = nn.Sequential(
                    nn.Linear(in_feature, 32),
                    nn.ReLU(),
                    nn.Linear(32, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim,out_dim),
                    nn.Softmax(dim=-1)
                )
        return self.net_struct

    def get_behavior_acion(self, state):
        return np.random.choice(len(self.behavior_policy[state]),1,p=self.behavior_policy[state])[0] # random choose an action based on policy
    def get_action(self, state, optimal=False):
        action_probs = self.policy_net(state)
        # action_probs = (actions_val/actions_val.sum()).detach().numpy()
        if optimal:
            return torch.argmax(action_probs).item()
        index = action_probs.multinomial(num_samples=1, replacement=True)
        return index.item()
        # return np.random.choice(len(action_probs),1,p=action_probs)[0]

    def loss(self, inp, target):
        pass



