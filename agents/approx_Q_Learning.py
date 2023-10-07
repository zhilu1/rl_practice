from collections import defaultdict
import numpy as np

class QLearningAgent:
    def __init__(self,
                action_space_n,
                initial_q = None,
                epsilon = 0.1,
                learning_rate = 0.01,
                behavior_policy = None
                ) -> None:
        self.action_space_n = action_space_n
        self.epsilon = epsilon
        # self.policy = defaultdict(lambda: np.zeros(self.action_space_n))
        self.policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))

        self.learning_rate = learning_rate # could depend on t, but never mind

        if initial_q:
            self.q = initial_q
            for state in self.q.keys():
                self.policy_improvement(state)
        else:
            self.q = defaultdict(lambda: np.zeros(action_space_n))
        if behavior_policy:
            self.behavior_policy = behavior_policy
        else:
            self.behavior_policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))
            

    def td_learn(self, state, action,  td_target):
        self.q[state][action] = self.q[state][action] - self.learning_rate * ( self.q[state][action] - td_target)
        

    def policy_improvement(self, state, optimal_action):
        # optimal_action = np.argmax(self.q[state])
        for action in range(self.action_space_n):
            self.policy[state][action] = self.epsilon/self.action_space_n
        self.policy[state][optimal_action] = 1-(self.action_space_n - 1) / self.action_space_n * self.epsilon

    def get_action(self, state, get_optimal = False) -> int:
        if get_optimal:
            return np.argmax(self.policy[state])
        return np.random.choice(len(self.policy[state]),1,p=self.policy[state])[0] # random choose an action based on policy
        
    def get_behavior_acion(self, state):
        return np.random.choice(len(self.behavior_policy[state]),1,p=self.behavior_policy[state])[0] # random choose an action based on policy
