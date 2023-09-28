import numpy as np
from collections import defaultdict

class ModelBasedValueIterationAgent():
    def __init__(self,
                 action_space_n: int,
                 initial_state_values = None,
                 threshold: float = 0.001,
                 gamma: float = 0.9
                 ) -> None:
        self.v = defaultdict(int)
        self.old_v = defaultdict(int)
        if initial_state_values:
            self.v = initial_state_values.copy() # 初始化 v0
            self.old_v = initial_state_values.copy()
        self.threshold = threshold
        self.q = defaultdict(lambda: np.zeros(action_space_n))
        self.policy = defaultdict(lambda: np.zeros(action_space_n))
        self.gamma = gamma
    def get_action(self, state):
        action_index = np.argmax(self.q[state])
        return action_index

    def value_update(self, state):
        update_amount = 0
        # for s in self.q.keys():
        self.v[state] = max(self.q[state])
        update_amount += (self.v[state] - self.old_v[state])
        return update_amount
    def not_converged(self, total_update):
        if total_update < self.threshold:
            return False
        else:
            return True
        # pass
    def q_table_update(self, s, a, expected_immediate_reward, next_state):
        expected_future_reward = self.old_v[next_state]
        self.q[s][a] = expected_immediate_reward + self.gamma * expected_future_reward
        return
    def policy_update(self, state):
        # for state in self.q.keys():
            # policy_prob[a] is probabiliy of action a
        a = np.argmax(self.q[state])
        for i in range(len(self.policy[state])):
            self.policy[state][i] = 0
        self.policy[state][a] = 1
        return
            

        # for s in range(self.state_space_n):
        #     for a in range(self.action_space_n):
        #         expected_future_reward = np.sum([transition_prob[s][a][s1] * self.v[s1] for s1 in transition_prob[s][a].keys()])
        #         self.q[s][a] = expected_reward[s][a] + self.gamma * expected_future_reward 
        #         self.policy[s][a] = 0
        #     optimal_a = self.get_action(self, s)
        #     self.policy[s][optimal_a] = 1
        # return
        # for i, state in enumerate(obs):
            # self.v[]