from torch import nn
from torch import optim
from collections import defaultdict
import numpy as np
import torch
from rl_envs.gym_grid_world_env import GridWorldEnv
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter # type: ignore
class PolicyNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.observation_space = self.height * self.width
        self.fc1 = nn.Linear(self.observation_space, kwargs['out_dim'])
        torch.nn.init.zeros_(self.fc1.weight)
    def forward(self, inp):
        sq_inp = inp[0] * self.width + inp[1]
        out = torch.tensor(sq_inp, dtype=torch.int64)
        out1 = torch.nn.functional.one_hot(out, self.observation_space).to(torch.float).unsqueeze(0)
        out3 = self.fc1(out1)
        probs = torch.nn.functional.softmax(out3, dim=-1)
        return probs
class ValueNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # self.fc1 = nn.Linear(kwargs['in_dim'], 1)
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.observation_space = self.height * self.width
        self.fc1 = nn.Linear(self.observation_space, 1)
        # self.observation_space = kwargs['in_dim']
        torch.nn.init.zeros_(self.fc1.weight)
    def forward(self, inp):
        sq_inp = inp[0] * self.width + inp[1]
        out = torch.tensor(sq_inp, dtype=torch.int64)
        out1 = torch.nn.functional.one_hot(out, self.observation_space).to(torch.float).unsqueeze(0)
        out3 = self.fc1(out1)
        return out3
class A2CAgent:
    """
    The simplest actor-critic algorithm (QAC) 
    和 PG 对比, 如果 q(s, a) 是通过另一个函数来估算的，那么相应的算法就是 actor-critic
    """
    def __init__(self,
                state_space,
                action_space,
                lr_policy = 0.001,
                lr_v = 0.0015,
                discounted_factor = 0.9,
                height = 3,
                width = 3,
                save_action = False
                 ) -> None:
        self.save_actionprob = save_action
        
        self.lr_policy = lr_policy
        self.lr_v = lr_v
        self.state_space = state_space
        self.action_space = action_space
        self.discounted_factor = discounted_factor

        self.policy_net = PolicyNet(in_dim=state_space, out_dim=action_space, height=height, width=width)
        # self.v_net = self.initialize_v_net(input_dim=state_space, output_dim=action_space)
        self.value_net = ValueNet(in_dim=state_space, out_dim=1, height=height, width=width)


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
    def generate_policy_table(self, height, width):
        """
        only for debug use, AC doesn't own nor need a real policy table
        """
        policy = {}
        for y in range(height):
            for x in range(width):
                state = torch.tensor((y,x), dtype=torch.float)
                policy_prob = self.policy_net(state)
                policy[(y,x)] = policy_prob.detach().numpy()
        return policy

    def generate_v_table(self, height, width):
        """
        only for debug use, AC doesn't own nor need a real V table
        """
        V = {}
        for y in range(height):
            for x in range(width):
                state = torch.tensor((y,x), dtype=torch.float)
                state_value = self.value_net(state)
                V[(y,x)] = state_value.detach().numpy()
        return V
    