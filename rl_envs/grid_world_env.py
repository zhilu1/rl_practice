from collections import defaultdict

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
                ) -> None:
        self.height = height
        self.width = width
        self.grids = [[normal_reward for _ in range(width)] for _ in range(height)]
        for f_grid in forbidden_grids:
            self.grids[f_grid[0]][f_grid[1]] = forbidden_reward
        for t_grid in target_grids:
            self.grids[t_grid[0]][t_grid[1]] = target_reward
        self._action_space = [(1, 0),(-1,0),(0, 1),(0, -1),(0,0)]
        self.hit_wall_reward = hit_wall_reward
        self.possible_actions = len(self._action_space)
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.expected_rewards = defaultdict(lambda: defaultdict(float))
        
    def model_based_transitions(self, certain_transitions={}):
        for i in range(self.height):
            for j in range(self.width):
                for action, move in enumerate(self._action_space):
                    y, x = i+move[0], j+move[1]
                    self.transition_probs[((i,j), action)][(y,x)] = 1
        
        for state_action in certain_transitions.keys():
            for next_state, prob  in certain_transitions[state_action].items():
                self.transition_probs[state_action][next_state] = prob

        for i in range(self.height):
            for j in range(self.width):
                state = (i,j)
                for action, _ in enumerate(self._action_space):
                    for next_state, prob in self.transition_probs[(state, action)].items():
                        if prob <= 0:
                            continue
                        y, x = next_state
                        if x >= self.width or x < 0 or y >= self.height or y < 0:
                            # hitwall
                            self.expected_rewards[state][action] += prob * self.hit_wall_reward
                        else:
                            self.expected_rewards[state][action] += prob * self.grids[y][x]
            
        

    def valid_actions(self, state):
        return self.possible_actions
    # def step_all(self):
    #     # a0 up, a1 left, a2 down, a3 right, a4 stay
    #     total_reward = 0
    #     immediate_reward = defaultdict(0)
    #     for i in range(self.height):
    #         for j in range(self.width):
    #             a = self.action_grids[i][j]
    #             x, y = j + self._action_space[a][0], i + self._action_space[a][1]
    #             if x >= self.width or x < 0 or y >= self.height or y < 0:
    #                 # hitwall
    #                 total_reward += self.hit_wall_reward
    #                 immediate_reward[(i,j)] = self.hit_wall_reward
    #             else:
    #                 total_reward += self.grids[y][x]
    #                 immediate_reward[(i,j)] = self.grids[y][x]
    #     return total_reward, immediate_reward
    def step(self, state, a):
        i, j = state
        y, x = i + self._action_space[a][0], j + self._action_space[a][1]
        if x >= self.width or x < 0 or y >= self.height or y < 0:
            # hitwall
            return (i, j), self.hit_wall_reward
        else:
            return (y, x), self.grids[y][x]

    def __str__(self) -> str:
        to_print = ""
        for i in range(self.height):
            to_print += "[ "
            for j in range(self.width):
                to_print += '{:3d}'.format(self.grids[i][j])
                to_print += " "
                # print(self.grids[i][j], end=" ")
            to_print += "]\n"
        return to_print

    # def get_obs(self, i, j):
    #     return self.grids