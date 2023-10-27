from collections import defaultdict
import numpy as np
import torch
from rl_envs.new_gym_grid_world_env import GridWorldEnv


class ApproxQLearningAgent:
    def __init__(
        self, action_space_n, epsilon=0.1, learning_rate=1e-5, discounted_factor=0.9
    ) -> None:
        self.action_space_n = action_space_n
        self.epsilon = epsilon
        self.discounted_factor = discounted_factor
        # self.policy = defaultdict(lambda: np.zeros(self.action_space_n))
        self.policy = defaultdict(
            lambda: np.ones(self.action_space_n) * (1 / self.action_space_n)
        )
        self.learning_rate = learning_rate  # could depend on t, but never mind
        self.parameters = [1 for _ in range(17)]

    def phi_sa(self, state, action, i):
        x, y, z = *state, action
        t = [
            1,
            x,
            y,
            z,
            x * y,
            y * z,
            x * z,
            x * x,
            y * y,
            z * z,
            x * x * z,
            y * y * z,
            x * x * y,
            y * y * x,
            z * z * x,
            z * z * y,
            x * y * z,
        ]
        # t = {0: 1, 1:x, 2: y, 3:z, 4: x*y, 5: y*z, 6: x*z,7: x*x,}  # φ(s) = [1, x, y, x^2, y^2, xy]T ∈ R6.
        # t = {0: 1, 1:state[0], 2: state[1], 3:state[0]**2, 4: state[1]**2,5: state[0]*state[1]}  # φ(s) = [1, x, y, x^2, y^2, xy]T ∈ R6.
        # t = {0: 1, 1:cos(state[0] * pi), 2: cos(state[1] * pi), 3:cos((state[1]+state[0]) * pi)}  # φ(s) = [1, x, y, x^2, y^2, xy]T ∈ R6.
        return t[i]

    def estimate_q(self, state, action):
        q = 0
        for i, param in enumerate(self.parameters):
            q += self.phi_sa(state, action, i) * param
        return q

    def update_parameters(self, TD_error, state, action, next_state, best_action):
        # 可以在 update parameters 时顺便算出所有 state, action 的 phisa 并存储
        for i, param in enumerate(self.parameters):
            gradient = self.phi_sa(
                next_state, best_action, i
            ) * self.discounted_factor - self.phi_sa(state, action, i)
            self.parameters[i] += self.learning_rate * (2 * TD_error * gradient)

    def policy_improvement(self, state):
        best_action = 0
        best_q_value = -float("inf")
        for action in range(self.action_space_n):
            q_value = self.estimate_q(state, action)
            self.policy[state][action] = self.epsilon / self.action_space_n
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        self.policy[state][best_action] = (
            1 - ((self.action_space_n - 1) / self.action_space_n) * self.epsilon
        )

    def get_action(self, state, optimal=False) -> int:
        state = tuple(state)
        if optimal:
            return np.argmax(self.policy[state])
        return np.random.choice(len(self.policy[state]), 1, p=self.policy[state])[
            0
        ]  # random choose an action based on policy

    def RUN(self, env: GridWorldEnv, num_episodes=1000, episode_len=100000):
        next_state, _ = env.reset()
        episode_recorder = []
        for _ in range(episode_len):
            state = next_state
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_recorder.append(
                (
                    (state[0] / float(env.height), state[1] / float(env.width)),
                    action / env.action_n,
                    reward,
                    (
                        next_state[0] / float(env.height),
                        next_state[1] / float(env.width),
                    ),
                )
            )

        for i_episode in range(num_episodes):
            for i_step, (state, action, reward, next_state) in enumerate(
                episode_recorder
            ):
                q_value = self.estimate_q(state, action)

                next_q = -float("inf")
                best_action = 0
                for a in range(env.action_n):
                    q_eval = self.estimate_q(next_state, a)
                    if q_eval > next_q:
                        next_q = q_eval
                        best_action = a

                TD_error = reward + self.discounted_factor * next_q - q_value

                self.update_parameters(
                    TD_error, state, action, next_state, best_action
                )  # loss is MSE(TD_target, q_value)
                self.policy_improvement(state)
