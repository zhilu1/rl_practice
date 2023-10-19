from torch import nn
from torch import optim
from collections import defaultdict
import numpy as np
import torch
from rl_envs.gym_grid_world_env import GridWorldEnv

class QACAgent:
    """
    The simplest actor-critic algorithm (QAC) 
    和 PG 对比, 如果 q(s, a) 是通过另一个函数来估算的，那么相应的算法就是 actor-critic
    """
    def __init__(self,
                state_space,
                action_space,
                lr_policy = 0.001,
                lr_q = 0.0015,
                discounted_factor = 0.9
                ) -> None:
        self.lr_policy = lr_policy
        self.lr_q = lr_q
        self.state_space = state_space
        self.action_space = action_space
        self.discounted_factor = discounted_factor

        self.policy_net = self.initialize_policy_net(input_dim=state_space, output_dim=action_space)
        self.q_net = self.initialize_q_net(input_dim=state_space, output_dim=action_space)
        self.optimizer_p = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_q = torch.optim.Adam(self.q_net.parameters(), lr=lr_q)

    def initialize_policy_net(self, input_dim, output_dim):
        policy_netstruct = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
            nn.Softmax(dim=1)
        )
        return policy_netstruct
    def initialize_q_net(self, input_dim, output_dim):
        q_netstruct = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),
        )
        return q_netstruct

    def RUN(self, env:GridWorldEnv):
        epochs = 2
        episode_len = 1000
        for episode in range(epochs):
            obs, _ = env.reset()
            state = tuple(obs['agent'])
            for _ in range(episode_len):
                # generate (state, action, reward, next_state, next_action)
                state = torch.tensor(state, dtype = torch.float)
                action_probs = self.policy_net(state)
                action = torch.max(action_probs).indices
                obs, reward, terminated , truncated, info = env.step(action)
                next_state = tuple(obs['agent'])
                next_action = torch.max(self.policy_net(next_state)).indices
                # Actor: policy update
                q_values = self.q_net(state)
                # TODO 这里 q values 可能有问题 (尤其在 Batch 后一定会有问题)
                self.optimizer_p.zero_grad()
                loss_p = torch.log(action_probs[action]) * q_values[action] # loss 或有问题
                loss_p.sum().backward()
                self.optimizer_p.step()

                # Critic: value update
                target_q = self.q_net(next_state)[next_action]
                self.optimizer_q.zero_grad()
                loss_q = (reward + self.discounted_factor * target_q - q_values[action]) ** 2
                loss_q.sum().backward()
                self.optimizer_q.step()
        

                 



