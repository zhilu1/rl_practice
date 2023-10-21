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
        self.fc1 = nn.Linear(kwargs['in_dim'], 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, kwargs['out_dim'])
    def forward(self, inp):
        out1 = self.fc1(inp)
        out1 = torch.nn.functional.relu(out1)
        out2 = self.fc2(out1)
        out2 = torch.nn.functional.relu(out2)
        out3 = self.fc3(out2)
        out3 = torch.nn.functional.relu(out3)
        probs = torch.nn.functional.softmax(out3, dim=-1)
        return probs


class PGAgent:
    def __init__(self,
                 state_space_n,
                 action_space_n,
                 lr: float = 0.01,
                 discounted_factor = 0.9
                 ) -> None:
        self.action_space = action_space_n
        self.policy_net = self.initialize_network(state_space_n, 128, self.action_space)
        # self.policy_net = PolicyNet(in_dim=state_space_n, out_dim=self.action_space)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) # 输入 state, 输出每个 action 的概率
        self.behavior_policy = defaultdict(lambda: np.ones(self.action_space) * (1/self.action_space))
        self.q = defaultdict(lambda: defaultdict(lambda: -100))
        self.v = defaultdict(lambda: -100)
        self.discounted_factor = discounted_factor
        self.saved_log_probs = []
    def initialize_network(self, in_feature, hidden_dim, out_dim):
        # `in_feature` input feature dim depends on encoding  of (state, action) pair
        net_struct = nn.Sequential(
                    nn.Linear(in_feature, hidden_dim),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Linear(hidden_dim//2,out_dim),
                    nn.Softmax(dim=-1)
                )
        return net_struct

    def get_behavior_action(self, state):
        return np.random.choice(len(self.behavior_policy[state]),1,p=self.behavior_policy[state])[0] # random choose an action based on policy
    def get_action(self, state, optimal=False):
        # with torch.no_grad(): # 哪里都 no_grad 只会害了你 
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action_probs = self.policy_net(state)
        # action_probs = (actions_val/actions_val.sum()).detach().numpy()
        if optimal:
            return torch.argmax(action_probs).item()
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        # action = np.random.choice(len(action_probs),1,p=action_probs.numpy())[0]
        return action.item()
        # index = action_probs.multinomial(num_samples=1, replacement=True)
        # return index.item()
        # return np.random.choice(len(action_probs),1,p=action_probs)[0]
    def update(self, trajectory):
        discounted_reward = 0
        for state, action, reward in reversed(trajectory):
            # REINFORCE 或叫 Monte Carlo policy gradient, 其中原因就是 q(s, a) 是由 Monte Carlo learning 方法估计
            # 也就是这里的 self.q[state][action] = discounted_reward, 但 PG 算法 实际上并不需要记录 q 值因此不写出
            discounted_reward = discounted_reward * self.discounted_factor + reward
            # self.q[state][action] = discounted_reward 

            # policy update
            self.optimizer.zero_grad()
            """
            特别注意: 这里 log π 中的 π(a|s) 是选择 a 的概率, policy network 得输出一个概率, 而不是什么 a 的值
            当然我们可以用输出的值, 归一化一下作为 action 的概率

            有一个 变体可能, 在 sample action 时就计算 prob并存储, 然后在 update 时就只是计算 reward 从而计算 loss, 将一个 episode 的loss 都加到一起来一起 backward, 然后更新一次 policy network
            """

            action_probs = self.policy_net(torch.tensor(state, dtype=torch.float))
            # action_probs = actions_val/actions_val.sum()
            loss = -torch.log(action_probs[action]) * discounted_reward
            # loss = abs(loss)
            loss.backward()
            self.optimizer.step()
            return loss
    def generate_policy_table(self, height, width):
        """
        only for debug use, PG doesn't own nor need a real policy table
        """
        policy = {}
        for y in range(height):
            for x in range(width):
                state = torch.tensor((y,x), dtype=torch.float).unsqueeze(0)
                policy_prob = self.policy_net(state)
                policy[(y,x)] = policy_prob.detach().numpy()
        return policy

    def loss(self, inp, target):
        pass



