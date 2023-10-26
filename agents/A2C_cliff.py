"""
Policy gradient using MLP
"""
import torch
from torch import nn
from torch import optim
from collections import defaultdict
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fc1 = nn.Linear(kwargs['in_dim'], kwargs['out_dim'])
        torch.nn.init.zeros_(self.fc1.weight)
    def forward(self, inp):
        out = torch.tensor(inp, dtype=torch.int64)
        out1 = torch.nn.functional.one_hot(out, 48).to(torch.float).unsqueeze(0)
        out3 = self.fc1(out1)
        probs = torch.nn.functional.softmax(out3, dim=-1)
        return probs
class ValueNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fc1 = nn.Linear(kwargs['in_dim'], 1)
        torch.nn.init.zeros_(self.fc1.weight)
    def forward(self, inp):
        out = torch.tensor(inp, dtype=torch.int64)
        out1 = torch.nn.functional.one_hot(out, 48).to(torch.float).unsqueeze(0)
        out3 = self.fc1(out1)
        return out3

class A2CAgent:
    def __init__(self,
                 state_space_n,
                 action_space_n,
                lr_policy = 0.001,
                lr_v = 0.0015,
                 discounted_factor = 0.9,
                 save_action = False
                 ) -> None:
        self.action_space = action_space_n
        self.save_actionprob = save_action
        # self.policy_net = self.initialize_network(state_space_n, 128, self.action_space)
        self.policy_net = PolicyNet(in_dim=state_space_n, out_dim=self.action_space)
        self.value_net = ValueNet(in_dim=state_space_n, out_dim=self.action_space)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy) # 输入 state, 输出每个 action 的概率
        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=lr_v) # 输入 state, 输出每个 action 的概率
        self.behavior_policy = defaultdict(lambda: np.ones(self.action_space) * (1/self.action_space))
        self.q = defaultdict(lambda: defaultdict(lambda: -100))
        self.v = defaultdict(lambda: -100)
        self.discounted_factor = discounted_factor
        self.saved_log_probs = []
        self.saved_log_prob = 0
        self.first_occ_set = set()
    def initialize_network(self, in_feature, hidden_dim, out_dim):
        # `in_feature` input feature dim depends on encoding  of (state, action) pair
        net_struct = nn.Sequential(
                    nn.Linear(in_feature,  out_dim),
                    # nn.Dropout(0.5),
                    # nn.ReLU(),
                    # nn.Linear(hidden_dim, hidden_dim//2),
                    # nn.Dropout(0.5),
                    # nn.ReLU(),
                    # nn.Linear(hidden_dim//2,out_dim),
                    nn.Softmax(dim=-1)
                )
        return net_struct

    def get_behavior_action(self, state):
        return np.random.choice(len(self.behavior_policy[state]),1,p=self.behavior_policy[state])[0] # random choose an action based on policy
    def get_action(self, in_state, optimal=False):
        # with torch.no_grad(): # 哪里都 no_grad 只会害了你 
        # state = torch.tensor(in_state, dtype=torch.int64)
        # state = torch.nn.functional.one_hot(state, 48)
        # with torch.no_grad():
        action_probs = self.policy_net(in_state)
        # action_probs = (actions_val/actions_val.sum()).detach().numpy()
        if optimal:
            return torch.argmax(action_probs).item()
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()

        if self.save_actionprob:
            logProb = m.log_prob(action)
            self.saved_log_prob = logProb
            self.saved_log_probs.insert(0, logProb)
        return action.item()


