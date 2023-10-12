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
    def update(self, episode_recorder, discounted_factor):
        discounted_reward = 0
        for observation, action, reward, next_observation in reversed(episode_recorder):
            discounted_reward = discounted_reward * discounted_factor + reward
            # agent.q[observation][action] = discounted_reward 

            # policy update
            self.optimizer.zero_grad()
            """
            特别注意: 这里 log π 中的 π(a|s) 是选择 a 的概率, policy network 得输出一个概率, 而不是什么 a 的值
            当然我们可以用输出的值, 归一化一下作为 action 的概率

            有一个 变体可能, 在 sample action 时就计算 prob并存储, 然后在 update 时就只是计算 reward 从而计算 loss, 将一个 episode 的loss 都加到一起来一起 backward, 然后更新一次 policy network
            """

            action_probs = self.policy_net(torch.from_numpy(observation))
            # action_probs = actions_val/actions_val.sum()
            loss = -torch.log(action_probs[action]) * discounted_reward
            # loss = abs(loss)
            loss.backward()
            self.optimizer.step()
            return loss

    def loss(self, inp, target):
        pass



