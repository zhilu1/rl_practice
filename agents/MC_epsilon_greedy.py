from collections import defaultdict
import numpy as np

class MC_epsilon_greedy:
    def __init__(self,
                action_space_n: int,
                epsilon: float,
                is_every_visit: bool = True
                ) -> None:
        # 
        self.epsilon = epsilon
        self.is_every_visit = is_every_visit # perform every visit or first visit
        self.state_action_returns = defaultdict(float)
        self.state_action_nums = defaultdict(int)
        self.q = defaultdict(lambda: np.zeros(action_space_n))
        self.action_space_n = action_space_n
        # self.policy = defaultdict(lambda: defaultdict(lambda: 1/action_space_n))


    def initialize_policy(self, default_action = 0, initial_policy = None):
        if initial_policy:
            self.policy = initial_policy.copy()
            return
        self.policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))
        
    def get_action(self, state, get_optimal = False) -> int:
        if get_optimal:
            return np.argmax(self.policy[state])
        return np.random.choice(len(self.policy[state]),1,p=self.policy[state])[0] # random choose an action based on policy
        # action_index = max(self.policy[state], key=self.policy[state].get)
        
    def policy_improvement(self, state):
        optimal_action = np.argmax(self.q[state])
        for action in range(self.action_space_n):
            self.policy[state][action] = self.epsilon/self.action_space_n
        self.policy[state][optimal_action] = 1-(self.action_space_n - 1) / self.action_space_n * self.epsilon

            
