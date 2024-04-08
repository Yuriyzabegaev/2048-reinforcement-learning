from game import Game, PyGame2048
import time
import pygame
import numpy as np

from gymnasium import spaces
from gymnasium import Env

# FPS30 = 1 / 30
FPS30 = 1


class Env2048(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="rgb_array"):
        self.observation_space = spaces.Box(0, 20, shape=(16,))
        self.t = 0
        self.game = Game()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        if render_mode == "human":
            pygame.init()
            self.pygame = PyGame2048(self.game)

    def close(self):
        super().close()
        pygame.quit()

    def _get_obs(self):
        obs = self.game.grid.copy().reshape(-1)
        obs[obs == 0] = 1
        return np.log2(obs)

    def _get_info(self):
        return {"score": self.game.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game = Game()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.pygame = PyGame2048(self.game)
            self._render_frame()

        return observation, info

    def step(self, action):
        if action == 0:
            self.game.up()
            reward_mul = 0.1
        elif action == 1:
            self.game.down()
            reward_mul = 1
        elif action == 2:
            self.game.left()
            reward_mul = 0.9
        elif action == 3:
            self.game.right()
            reward_mul = 0.5
        else:
            raise ValueError

        if self.game.nothing_moved:
            reward = -10
        else:
            reward = self.game.current_reward
            if reward == 0:
                reward = 1
            reward = np.log2(reward) * reward_mul
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        terminated = self.game.is_failed() or self.game.is_complete()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            self.pygame.draw_grid()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            pygame.time.Clock().tick(30)


if __name__ == "__main__":
    env = Env2048(render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.1)

    env.close()
