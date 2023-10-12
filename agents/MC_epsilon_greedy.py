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
        self.state_action_nums = defaultdict(float)
        self.q = defaultdict(lambda: np.zeros(action_space_n))
        self.action_space_n = action_space_n
        # self.policy = defaultdict(lambda: defaultdict(lambda: 1/action_space_n))


    def initialize_policy(self, initial_policy = None):
        if initial_policy:
            self.policy = initial_policy.copy()
            return
        else:
            # self.policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))
            self.policy = defaultdict(lambda: [0.0 for _ in range(self.action_space_n-1)] + [1.0])
        
    def get_action(self, state, get_optimal = False) -> int:
        if get_optimal:
            return np.argmax(self.policy[state])
        return np.random.choice(len(self.policy[state]),1,p=self.policy[state])[0] # random choose an action based on policy
        # action_index = max(self.policy[state], key=self.policy[state].get)
        
    # def policy_improvement(self, state):
    #     optimal_action = np.argmax(self.q[state])
    #     for action in range(self.action_space_n):
    #         self.policy[state][action] = self.epsilon/self.action_space_n
    #     self.policy[state][optimal_action] = 1-((self.action_space_n - 1) / self.action_space_n) * self.epsilon

    def policy_improvement(self, height, width):
        for i in range(height):
            for j in range(width):
                state = (i,j)
                optimal_action = self.action_space_n - np.argmax(list(reversed(self.q[state]))) - 1
                for action in range(self.action_space_n):
                    self.policy[state][action] = self.epsilon/self.action_space_n
                self.policy[state][optimal_action] = 1-((self.action_space_n - 1) / self.action_space_n) * self.epsilon
            
