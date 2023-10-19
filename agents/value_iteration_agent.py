import numpy as np
from collections import defaultdict
from rl_envs.grid_world_env import GridWorldEnv

class ValueIterationAgent():
    def __init__(self,
                 action_space_n: int,
                 initial_state_values = None,
                 threshold: float = 0.001,
                 discounted_factor: float = 0.9
                 ) -> None:
        self.v = defaultdict(int)
        self.old_v = defaultdict(int)
        if initial_state_values:
            self.v = initial_state_values.copy() # 初始化 v0
            self.old_v = initial_state_values.copy()
        self.threshold = threshold
        self.q = defaultdict(lambda: np.zeros(action_space_n))
        self.policy = defaultdict(lambda: np.zeros(action_space_n))
        self.discounted_factor = discounted_factor
    def get_action(self, state):
        action_index = np.argmax(self.policy[state])
        return action_index

    def value_update(self, state):
        self.v[state] = max(self.q[state])
        return abs(self.v[state] - self.old_v[state])
        # diff = abs(max(self.q[state]) - self.v[state]) 
        # self.v[state] = max(self.q[state])
        # return diff
    
    def not_converged(self, total_update):
        if total_update < self.threshold:
            return False
        else:
            return True
        
    def q_table_update(self, s, a, expected_immediate_reward, next_state, prob_sr):
        expected_future_reward = self.old_v[next_state]
        # expected_future_reward = self.v[next_state]
        self.q[s][a] += prob_sr * (expected_immediate_reward + self.discounted_factor * expected_future_reward)
        return
    
    def policy_update(self, all_states):
        for state in range(all_states):
            a = np.argmax(self.q[state])
            for i in range(len(self.policy[state])):
                self.policy[state][i] = 0
            self.policy[state][a] = 1
        return

    def run(self, env: GridWorldEnv):
        amount_update = float('inf')
        while self.not_converged(amount_update):
            amount_update = 0
            for s in range(env.nS):
                for action in range(env.valid_actions(s)):
                    self.q[s][action] = 0
                    for prob, next_state, imm_reward, done in env.P[s][action]:
                        self.q_table_update(s, action, imm_reward, next_state, prob)
                amount_update = max(amount_update,self.value_update(s))
            self.old_v = self.v.copy()
        self.policy_update(env.nS) # this step is not necessary, policy is not actually used in VI
        