from collections import defaultdict
import numpy as np


class TruncatedPolicyIterationAgent:
    def __init__(self,
                action_space_n: int,
                threshold: float = 0.001,
                discounted_factor: float = 0.9,
                truncate_time: int = 20
                ) -> None:
        self.v = defaultdict(int)
        self.old_v = defaultdict(int)
        self.threshold = threshold
        self.q = defaultdict(lambda: np.zeros(action_space_n))
        # self.policy = defaultdict(lambda: defaultdict(float))
        self.discounted_factor = discounted_factor
        self.truncate_time = truncate_time
    def not_converged(self, total_update):
        if total_update < self.threshold:
            return False
        else:
            return True
    def initialize_policy(self, default_action = 0, initial_policy = None):
        if initial_policy:
            self.policy = initial_policy.copy()
            return
        self.policy = defaultdict(lambda: {default_action: 1})

    def get_action(self, state):
        action_index = np.argmax(self.policy[state])
        return action_index

    def policy_evaluation(self, s, expected_rewards, transition_probs):
        # for _ in range(self.truncate_time):
        self.v[s] = 0
        for a, action_prob in self.policy[s].items():
            if action_prob <= 0:
                continue
            future_reward = np.sum([prob * self.old_v[next_s] for next_s, prob in transition_probs[(s,a)].items()])
            # expected_reward = np.sum([prob * reward for reward, prob in reward_probs[(s,a)].items()])
            self.q[s][a] = expected_rewards[s][a] + self.discounted_factor * future_reward
            self.v[s] += action_prob * self.q[s][a]
        return abs(self.v[s] - self.old_v[s])

    def q_table_update(self, s, a, expected_immediate_reward, next_state):
        expected_future_reward = self.old_v[next_state]
        self.q[s][a] = expected_immediate_reward + self.discounted_factor * expected_future_reward
        return
    def policy_update(self, s):
        a = np.argmax(self.q[s])
        for i in range(len(self.policy[s])):
            self.policy[s][i] = 0
        self.policy[s][a] = 1
        return
    
    def policy_improvement(self, s, expected_rewards, transition_probs):

        for a, action_prob in self.policy[s].items():
            if action_prob <= 0:
                continue
            future_reward = np.sum([prob * self.v[next_s] for next_s, prob in transition_probs[(s,a)].items()])
            self.q[s][a] = expected_rewards[s][a] + self.discounted_factor * future_reward
        
        a = np.argmax(self.q[s])
        for i in range(len(self.policy[s])):
            self.policy[s][i] = 0
        self.policy[s][a] = 1
        return


