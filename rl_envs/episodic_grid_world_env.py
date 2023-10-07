from collections import defaultdict
# NOTDONE yet
"""
1. 设置 episode length?
2. 计算reward时, 需要将前一步的 state 带入? 看看是否已经在
3. 总之就是想要让到达 target 过就不能再次获得 reward
    - 感觉就是一个针对每个 agent 的 state

"""

class EpisodicGridWorldEnv():
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
        self.discounted_factor = discounted_factor
        self.grids = [[normal_reward for _ in range(width)] for _ in range(height)]
        for f_grid in forbidden_grids:
            self.grids[f_grid[0]][f_grid[1]] = forbidden_reward
        for t_grid in target_grids:
            self.grids[t_grid[0]][t_grid[1]] = target_reward
        self._action_space = [(1, 0),(-1,0),(0, 1),(0, -1),(0,0)]
        self.hit_wall_reward = hit_wall_reward
        self.possible_actions = len(self._action_space)
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.target_grids = target_grids
        self.is_terminated = False
    def termination(self):
        self.is_terminated = True
    def restart(self):
        self.is_terminated = False

    def valid_actions(self, state):
        return self.possible_actions

    def step(self, state, a):
        if self.is_terminated:
            return (i,j), 0

        i, j = state
        y, x = i + self._action_space[a][0], j + self._action_space[a][1]
        if x >= self.width or x < 0 or y >= self.height or y < 0:
            # hitwall
            return (i, j), self.hit_wall_reward, False
        else:
            if (y, x) in self.target_grids:
                self.is_terminated = True
            return (y, x), self.grids[y][x], True

    def __str__(self) -> str:
        to_print = ""
        for i in range(self.height):
            to_print += "[ "
            for j in range(self.width):
                to_print += '{:3d}'.format(self.grids[i][j])
                to_print += " "
            to_print += "]\n"
        return to_print

    # def get_obs(self, i, j):
    #     return self.grids