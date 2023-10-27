from collections import defaultdict
import numpy as np
import torch
from rl_envs.new_gym_grid_world_env import GridWorldEnv
class ApproxQLearningAgent:
    def __init__(self,
                action_space_n,
                epsilon = 0.1,
                learning_rate = 0.01,
                discounted_factor = 0.9
                ) -> None:
        self.action_space_n = action_space_n
        self.epsilon = epsilon
        self.discounted_factor = discounted_factor
        # self.policy = defaultdict(lambda: np.zeros(self.action_space_n))
        self.policy = defaultdict(lambda: np.ones(self.action_space_n) * (1/self.action_space_n))
        self.learning_rate = learning_rate # could depend on t, but never mind
        self.parameters = [0 for _ in range(17)]

    def phi_sa(self, state, action, i):
        x,y,z = *state, action
        t = [1,x,y,z,x*y,y*z,x*z,x*x,y*y,z*z,x*x*z,y*y*z,x*x*y,y*y*x,z*z*x,z*z*y,x*y*z]
        # t = {0: 1, 1:x, 2: y, 3:z, 4: x*y, 5: y*z, 6: x*z,7: x*x,}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
        # t = {0: 1, 1:state[0], 2: state[1], 3:state[0]**2, 4: state[1]**2,5: state[0]*state[1]}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
        # t = {0: 1, 1:cos(state[0] * pi), 2: cos(state[1] * pi), 3:cos((state[1]+state[0]) * pi)}  # φ(s) = [1, x, y, x2, y2, xy]T ∈ R6.
        return t[i]
    def estimate_q(self, state,  action):
        q = 0
        for i, param in enumerate(self.parameters):
            q += self.phi_sa(state, action, i) * param
        return q
    def update_parameters(self, loss):
        pass

    def policy_improvement(self, state):
        best_action = 0
        best_q_value = -float('inf')
        for action in range(self.action_space_n):
            q_value = self.estimate_q(state,  action)
            self.policy[state][action] = self.epsilon / self.action_space_n
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        self.policy[state][best_action] = 1-((self.action_space_n - 1) / self.action_space_n) * self.epsilon

    def get_action(self, state, optimal = False) -> int:
        if optimal:
            return np.argmax(self.policy[state])
        return np.random.choice(len(self.policy[state]),1,p=self.policy[state])[0] # random choose an action based on policy


    def RUN(self, env:GridWorldEnv):
        num_episodes = 1000
        episode_len = 100000

        next_state, _ = env.reset()
        episode_recorder = []
        for _ in range(episode_len):
            state = next_state
            action = env.action_space.sample()
            next_state, reward, terminated , truncated, info = env.step(action)
            episode_recorder.append((state, action, reward, next_state))

        for i_episode in range(num_episodes):
            for state, action, reward, next_state in episode_recorder:
                q_value = self.estimate_q(state, action)

                next_q =  -float('inf')
                for a in range(env.action_n):
                    q_eval = self.estimate_q(next_state, a)
                    next_q = q_eval if q_eval > next_q else next_q

                TD_error = reward + self.discounted_factor * next_q - q_value

                self.update_parameters(TD_error ** 2) # loss is MSE(TD_target, q_value)                
                self.policy_improvement(state)


