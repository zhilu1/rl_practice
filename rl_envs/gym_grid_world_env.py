import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import torch
from collections import defaultdict

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, 
                fixed_map = True,
                forbidden_grids=[], 
                target_grids=[], 
                target_reward = 1.0,
                forbidden_reward = -1.0, 
                hit_wall_reward = -1.0,):
        self.size = size  # The size of the square grid
        self.height = size
        self.width = size
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=float),
                "target": spaces.Box(0, size - 1, shape=(len(target_grids),2,), dtype=float),
                "forbidden": spaces.Box(0, size - 1, shape=(len(forbidden_grids),2,), dtype=float),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "stay"
        self.action_space = spaces.Discrete(5)
        self.action_n = int(self.action_space.n)

        
        """
        generate the map randomly or use a fixed map
        """
        self.fixed_map = fixed_map
        self.target_grids = target_grids
        self.forbidden_grids = forbidden_grids
        self.target_reward = target_reward
        self.forbidden_reward = forbidden_reward
        self.hit_wall_reward = hit_wall_reward
        self.max_steps = 100 # maximum step in a run until truncate
        self.num_step = 0 # step counter

        """
        model-based 时所使用的参数
        """
        self.Prsa = defaultdict(lambda: defaultdict(list)) # Prsa[s][a] = [(prob, reward)]
        self.Pssa = defaultdict(lambda: defaultdict(list)) # Pssa[s][a] = [(prob, next_state)]
            
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }
        self.action_mappings = [" ↓ "," → "," ↑ ", " ← "," ↺ "]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "forbidden": self._forbidden_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Choose the agent's location uniformly at random
        if options and options['start_position']:
            self._agent_location = options['start_position']
        else:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        if self.fixed_map:
            self._target_location = np.array(self.target_grids)
            self._forbidden_location = np.array(self.forbidden_grids)
        else:
            # We will sample the target's location randomly until it does not coincide with the agent's location
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )

            self._forbidden_location = self._agent_location
            while np.array_equal(self._forbidden_location, self._agent_location) or np.array_equal(self._forbidden_location, self._target_location):
                self._forbidden_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )


        observation = self._get_obs()
        info = self._get_info()

        self.num_step = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # An episode is done iff the agent has reached the target
        if not isinstance(action, int):
            action = action.item()
        direction = self._action_to_direction[action]
        new_loc =  self._agent_location + direction
        terminated = any(np.equal(self._target_location, new_loc).all(1))
        if (new_loc >= self.size).any() or (new_loc < 0).any():
            reward = self.hit_wall_reward
        else:
            self._agent_location = new_loc
            reward = self.target_reward if terminated else 0  # Binary sparse rewards
            reward = self.forbidden_reward if any(np.equal(self._forbidden_location,self._agent_location).all(1)) else reward
            
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.num_step+=1
        truncated = False
        if self.num_step > self.max_steps:
            truncated = True

        return observation, reward, terminated , truncated, info

    def initialize_model_based(self):
        """
        initialize this before using model-based algorithm

        Prsa: p(r|s, a)
        Pssa: p(s'|s, a)
        """
        for i in range(self.height):
            for j in range(self.width):
                state = (i,j)
                for action, move in self._action_to_direction.items():
                    y, x = i+move[0], j+move[1]
                    if x >= self.width or x < 0 or y >= self.height or y < 0:
                        # hitwall
                        self.Prsa[state][action] = [(1, self.hit_wall_reward)]
                        self.Pssa[state][action] = [(1, (i,j))]
                    else:
                        if (y,x) in self.target_grids:
                            self.Prsa[state][action] = [(1, self.target_reward)]
                            self.Pssa[state][action] = [(1, (y,x))]
                        elif (y,x) in self.forbidden_grids:
                            self.Prsa[state][action] = [(1, self.forbidden_reward)]
                            self.Pssa[state][action] = [(1, (y,x))]
                        else:
                            self.Prsa[state][action] = [(1, 0)]
                            self.Pssa[state][action] = [(1, (y,x))]
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target grids
        for loc in self._target_location:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * loc[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )
        # Then we draw the forbidden grids
        for loc in self._forbidden_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * loc[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()