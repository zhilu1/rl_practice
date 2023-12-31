from torch import nn
from torch import optim
from collections import defaultdict
import numpy as np
import torch
from rl_envs.gym_grid_world_env import GridWorldEnv
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter # type: ignore
import torch.utils.data

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
                discounted_factor = 0.9
                ) -> None:
        
        self.state_space = state_space
        self.action_space = action_space
        self.discounted_factor = discounted_factor

        self.policy_net = self.initialize_policy_net(input_dim=state_space, output_dim=action_space)
        # self.v_net = self.initialize_v_net(input_dim=state_space, output_dim=action_space)
        self.v_net = self.initialize_v_net(input_dim=state_space, output_dim=1)

        self.optimizer_p = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=lr_v)
        self.behavior_policy = defaultdict(lambda: np.ones(self.action_space) * (1./self.action_space))

    def initialize_policy_net(self, input_dim, output_dim):
        policy_netstruct = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
            nn.Softmax(dim=-1)
        )
        return policy_netstruct
    
    def initialize_v_net(self, input_dim, output_dim):
        v_netstruct = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=64),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=64),
            # nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),
        )
        return v_netstruct
    
    def get_behavior_action(self, state):
        return np.random.choice(len(self.behavior_policy[state]),1,p=self.behavior_policy[state])[0] # random choose an action based on policy
    def get_action(self, observation, optimal=True):
        state = observation['agent']
        action_probs = self.policy_net(torch.tensor(state, dtype=torch.float))
        if optimal:
            action = torch.argmax(action_probs)
        else:
            m = Categorical(action_probs)
            action = m.sample()
        return action.item()
    def generate_policy_table(self, height, width):
        """
        only for debug use, AC doesn't own nor need a real policy table
        """
        policy = {}
        for y in range(height):
            for x in range(width):
                state = torch.tensor((y,x), dtype=torch.float).unsqueeze(0)
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
                state = torch.tensor((y,x), dtype=torch.float).unsqueeze(0)
                state_value = self.v_net(state)
                V[(y,x)] = state_value.detach().numpy()
        return V
    

    def RUN(self, env:GridWorldEnv):
        """
        collect experience replay
        """
        episode_len = 1000
        batch_size = 100

        obs, _ = env.reset()
        next_state = torch.tensor(obs['agent'], dtype = torch.float)
        total_reward  = 0
        episode = []
        for _ in range(episode_len):
            # generate (state, action, reward, next_state, next_action)
            state = next_state
            action_probs = self.policy_net(state)
            # action = torch.max(action_probs, dim=-1).indices
            # choose action based on policy
            m = Categorical(action_probs)
            action = m.sample()
            obs, reward, terminated , truncated, info = env.step(action)
            next_state = torch.tensor(obs['agent'], dtype = torch.float)
            episode.append(
                {
                    "state": state,
                    "action": action.item(),
                    "reward": reward,
                    "next_state": next_state,
                    "action_probs": action_probs
                }
            )

        """
        generate pytorch data iter
        """

        reward = []
        state = []
        action = []
        action_probs = []
        next_state = []
        for i in range(len(episode)):
            reward.append(episode[i]["reward"])
            action.append(episode[i]["action"])
            action_probs.append(episode[i]["action_probs"])
            state.append(episode[i]["state"])
            next_state.append(episode[i]["next_state"])
        reward = torch.tensor(reward, dtype=torch.float).reshape(-1, 1)
        next_state = torch.stack(next_state)
        action = torch.tensor(action, dtype=torch.int).unsqueeze(1)
        state = torch.stack(state)
        action_probs = torch.stack(action_probs)
        data_arrays = (state, action, action_probs, reward, next_state)
        dataset = torch.utils.data.TensorDataset(*data_arrays)
        data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
        """
        execute algorithm        
        """

        writer = SummaryWriter()
        epochs = 100
        iter_counter = 0
        for episode in range(epochs):
            # next_action = torch.max(self.policy_net(next_state), dim=-1).indices
            for state, action, action_probs, reward, next_state in data_iter:

                v_values = self.v_net(state)
                TD_error = reward + self.discounted_factor * self.v_net(next_state) - v_values

                # Actor: policy update
                self.optimizer_p.zero_grad()
                action_prob = action_probs[torch.arange(end=len(action_probs)), action.squeeze()].unsqueeze(1)

                loss_actor = TD_error * -torch.log(action_prob) * (action_prob * self.action_space) # loss 或有问题
                loss_actor.sum().backward(retain_graph=True) # 这里这样是否会出问题
                # torch.nn.utils.clip_grad.clip_grad_norm_(self.policy_net.parameters(), 100)
                self.optimizer_p.step()

                # Critic: value update
                self.optimizer_v.zero_grad()
                loss_critic = TD_error * v_values * (action_prob * self.action_space)
                # loss_critic = TD_error * v_values * (action_prob * self.action_space)
                loss_critic.sum().backward()
                # torch.nn.utils.clip_grad.clip_grad_norm_(self.v_net.parameters(), 100)
                self.optimizer_v.step()

                writer.add_scalar('loss_critic', loss_critic.sum(), iter_counter)
                writer.add_scalar('loss_actor', loss_actor.sum(), iter_counter)
                total_reward += reward
                iter_counter += 1

                writer.add_scalar('reward', total_reward, episode)
                writer.flush()
        writer.close()












