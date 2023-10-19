from collections import defaultdict
from random import randint
import numpy as np

class GridWorldEnv():
    def __init__(self,
                height,
                width, 
                forbidden_grids, 
                target_grids, 
                target_reward = 1,
                forbidden_reward = -1, 
                normal_reward = 0,
                hit_wall_reward = -1,
                discounted_factor = 0.9
                ) -> None:
        self.height = height
        self.width = width
        self.discounted_factor = discounted_factor # To remove
        self.nS = width * height
        # 初始化 grids
        self.grids = np.zeros((height, width), dtype=float)
        # [[normal_reward for _ in range(width)] for _ in range(height)]
        for f_grid in forbidden_grids:
            self.grids[f_grid[0]][f_grid[1]] = forbidden_reward
        for t_grid in target_grids:
            self.grids[t_grid[0]][t_grid[1]] = target_reward
        self.target_grids = target_grids
        self.hit_wall_reward = hit_wall_reward
        #  初始化 action 相关
        self._action_space = [(-1,0),(0, 1),(1, 0),(0, -1),(0,0)]
        self.action_mappings = [" ↑ "," → "," ↓ ", " ← "," ↺ "]
        self._state_ind_change = [-width,1,width,-1,0] # state index change based on action
        self.action_n = len(self._action_space)
        #  model-based 的初始化
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.expected_rewards = defaultdict(lambda: defaultdict(float))
        self.P = defaultdict(lambda: defaultdict(list)) # P[s][a] = (prob, next_state, reward, is_done)
        
    def init_model_based_transitions(self):
        it = np.nditer(self.grids, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            i, j = it.multi_index
            
            for action, move in enumerate(self._action_space):
                y, x = i+move[0], j+move[1]
                next_s = s+self._state_ind_change[action]
                if x >= self.width or x < 0 or y >= self.height or y < 0:
                    # hitwall
                    self.P[s][action] = [(1, s, self.hit_wall_reward, False)]
                else:
                    if (y,x) in self.target_grids:
                        self.P[s][action] = [(1, next_s, self.grids[y][x], True)]
                    else:
                        self.P[s][action] = [(1, next_s, self.grids[y][x], False)]
            it.iternext()

        for i in range(self.height):
            for j in range(self.width):
                for action, move in enumerate(self._action_space):
                    y, x = i+move[0], j+move[1]
                    if x >= self.width or x < 0 or y >= self.height or y < 0:
                        # hitwall
                        self.transition_probs[((i,j), action)][(i,j)] = 1
                    else:
                        self.transition_probs[((i,j), action)][(y,x)] = 1

        for i in range(self.height):
            for j in range(self.width):
                state = (i,j)
                for action, move in enumerate(self._action_space):
                    # 注意在 MDP 中 reward 只取决于当前的状态和动作,  与未来无关
                    y, x = i+move[0], j+move[1]
                    if x >= self.width or x < 0 or y >= self.height or y < 0:
                        # hitwall
                        self.expected_rewards[state][action] = self.hit_wall_reward
                    else:
                        self.expected_rewards[state][action] = self.grids[y][x]
            
    def state_to_index(self, state):
        return state[0] * self.width + state[1]
    def index_to_state(self, index):
        return (index // self.width, index % self.width)
    def valid_actions(self, state):
        return self.action_n

    def step(self, state, a):
        i, j = state
        y, x = i + self._action_space[a][0], j + self._action_space[a][1]
        if x >= self.width or x < 0 or y >= self.height or y < 0:
            # hitwall
            return (i, j), self.hit_wall_reward
        else:
            return (y, x), self.grids[y][x]
    def reset(self):
        # get a random start state
        return randint(0, self.height - 1), randint(0, self.width - 1)
    def __str__(self) -> str:
        to_print = ""
        for i in range(self.height):
            to_print += "[ "
            for j in range(self.width):
                to_print += '{:2f}'.format(self.grids[i][j])
                to_print += " "
                # print(self.grids[i][j], end=" ")
            to_print += "]\n"
        return to_print