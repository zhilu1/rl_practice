import numpy as np
from collections import defaultdict

class ModelBasedValueIterationAgent():
    def __init__(self,
                 state_space_n: int,
                 action_space_n: int,
                 initial_state_values = None,
                 threshold: float = 0.01,
                 gamma: float = 0.9
                 ) -> None:
        self.v = np.zeros(state_space_n)
        if initial_state_values:
            self.v[:] = initial_state_values[:] # 初始化 v0
        self.threshold = threshold
        self.q = defaultdict(lambda: np.zeros(action_space_n))
        self.policy = defaultdict(lambda: np.zeros(action_space_n))
        self.gamma = gamma
        
    def get_action(self, state):
        action_index = np.argmax(self.q[state])
        return action_index
    def value_update(self):
        for s in range(self.state_space_n):
            self.v[s] = max(self.q[s])
            
        return
        # pass
    def policy_update(self, expected_reward, transition_prob):

        for s in range(self.state_space_n):
            for a in range(self.action_space_n):
                expected_future_reward = np.sum([transition_prob[s][a][s1] * self.v[s1] for s1 in range(self.state_space_n)])
                self.q[s][a] = expected_reward[s][a] + self.gamma * expected_future_reward 
                self.policy[s][a] = 0
            optimal_a = self.get_action(self, s)
            self.policy[s][optimal_a] = 1
        return
        # for i, state in enumerate(obs):
            # self.v[]