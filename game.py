import os
import numpy as np
import pygame


def clear_console():
    os.system("clear")


class Game:
    def __init__(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self.current_reward = 0
        self.nothing_moved = False
        self.score = 0
        for _ in range(3):
            self.spawn_tile()

    def print_grid(self):
        clear_console()
        print(self.grid)

    def is_failed(self):
        return len(np.where(self.grid == 0)[0]) == 0

    def is_complete(self):
        return np.max(self.grid) >= 2048

    def spawn_tile(self):
        two_or_four = np.random.random()
        if two_or_four < 0.75:
            val = 2
        else:
            val = 4
        empty = np.where(self.grid == 0)
        pos = np.random.choice(empty[0].size, size=1)
        self.grid[empty[0][pos], empty[1][pos]] = val

    def up(self):
        count_moves = 0
        self.current_reward = 0
        for col in range(4):
            vacant = None
            stack_pos = None
            stack_value = None
            for row in range(4):
                current_val = self.grid[row, col]
                if stack_value is not None and current_val == stack_value:
                    stack_value *= 2
                    self.grid[stack_pos, col] = stack_value
                    self.grid[row, col] = 0
                    vacant = stack_pos + 1
                    self.score += current_val
                    self.current_reward += current_val
                    count_moves += 1
                elif vacant is not None and current_val != 0:
                    self.grid[vacant, col] = current_val
                    self.grid[row, col] = 0
                    stack_pos = vacant
                    stack_value = current_val
                    vacant += 1
                    count_moves += 1
                if vacant is None and current_val == 0:
                    vacant = row
                if self.grid[row, col] != 0:
                    stack_pos = row
                    stack_value = current_val
        self.nothing_moved = count_moves == 0
        if count_moves > 0:
            self.spawn_tile()

    def down(self):
        count_moves = 0
        self.current_reward = 0
        for col in range(4):
            vacant = None
            stack_pos = None
            stack_value = None
            for row in reversed(range(4)):
                current_val = self.grid[row, col]
                if stack_value is not None and current_val == stack_value:
                    stack_value *= 2
                    self.grid[stack_pos, col] = stack_value
                    self.grid[row, col] = 0
                    vacant = stack_pos - 1
                    self.score += current_val
                    self.current_reward += current_val
                    count_moves += 1
                elif vacant is not None and current_val != 0:
                    self.grid[vacant, col] = current_val
                    self.grid[row, col] = 0
                    stack_pos = vacant
                    stack_value = current_val
                    vacant -= 1
                    count_moves += 1
                if vacant is None and current_val == 0:
                    vacant = row
                if self.grid[row, col] != 0:
                    stack_pos = row
                    stack_value = current_val
        self.nothing_moved = count_moves == 0
        if count_moves > 0:
            self.spawn_tile()

    def right(self):
        count_moves = 0
        self.current_reward = 0
        for row in range(4):
            vacant = None
            stack_pos = None
            stack_value = None
            for col in reversed(range(4)):
                current_val = self.grid[row, col]
                if stack_value is not None and current_val == stack_value:
                    stack_value *= 2
                    self.grid[row, stack_pos] = stack_value
                    self.grid[row, col] = 0
                    vacant = stack_pos - 1
                    self.score += current_val
                    self.current_reward += current_val
                    count_moves += 1
                elif vacant is not None and current_val != 0:
                    self.grid[row, vacant] = current_val
                    self.grid[row, col] = 0
                    stack_pos = vacant
                    stack_value = current_val
                    vacant -= 1
                    count_moves += 1
                if vacant is None and current_val == 0:
                    vacant = col
                if self.grid[row, col] != 0:
                    stack_pos = col
                    stack_value = current_val
        self.nothing_moved = count_moves == 0
        if count_moves > 0:
            self.spawn_tile()

    def left(self):
        count_moves = 0
        self.current_reward = 0
        for row in range(4):
            vacant = None
            stack_pos = None
            stack_value = None
            for col in range(4):
                current_val = self.grid[row, col]
                if stack_value is not None and current_val == stack_value:
                    stack_value *= 2
                    self.grid[row, stack_pos] = stack_value
                    self.grid[row, col] = 0
                    vacant = stack_pos + 1
                    self.score += current_val
                    self.current_reward += current_val
                    count_moves += 1
                elif vacant is not None and current_val != 0:
                    self.grid[row, vacant] = current_val
                    self.grid[row, col] = 0
                    stack_pos = vacant
                    stack_value = current_val
                    vacant += 1
                    count_moves += 1
                if vacant is None and current_val == 0:
                    vacant = col
                if self.grid[row, col] != 0:
                    stack_pos = col
                    stack_value = current_val
        self.nothing_moved = count_moves == 0
        if count_moves > 0:
            self.spawn_tile()


# Constants
GRID_SIZE = 4
TILE_SIZE = 100
GRID_MARGIN = 10
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * GRID_MARGIN

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
TILE_COLORS = {
    2: (255, 255, 128),
    4: (255, 255, 0),
    8: (255, 200, 0),
    16: (255, 150, 0),
    32: (255, 100, 0),
    64: (255, 50, 0),
    128: (255, 0, 0),
    256: (200, 0, 0),
    512: (150, 0, 0),
    1024: (100, 0, 0),
    2048: (50, 0, 0),
}


class PyGame2048:
    def __init__(self, game):
        self.game: Game = game

        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("2048 Game")

    # Function to draw the game grid
    def draw_grid(self):
        grid = self.game.grid
        screen = self.screen

        screen.fill(WHITE)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = grid[i, j]
                color = TILE_COLORS.get(value, GRAY)
                pygame.draw.rect(
                    screen,
                    color,
                    [
                        j * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN,
                        i * (TILE_SIZE + GRID_MARGIN) + GRID_MARGIN,
                        TILE_SIZE,
                        TILE_SIZE,
                    ],
                )
                if value != 0:
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(value), True, BLACK)
                    text_rect = text.get_rect(
                        center=(
                            j * (TILE_SIZE + GRID_MARGIN) + TILE_SIZE / 2 + GRID_MARGIN,
                            i * (TILE_SIZE + GRID_MARGIN) + TILE_SIZE / 2 + GRID_MARGIN,
                        )
                    )
                    screen.blit(text, text_rect)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.game.up()
                elif event.key == pygame.K_DOWN:
                    self.game.down()
                elif event.key == pygame.K_LEFT:
                    self.game.left()
                elif event.key == pygame.K_RIGHT:
                    self.game.right()
                print(game.current_reward)
        return False


if __name__ == "__main__":
    pygame.init()
    game = Game()
    ui = PyGame2048(game)
    while True:
        end = ui.handle_events()
        if end:
            break
        ui.draw_grid()
        pygame.display.flip()
        pygame.time.Clock().tick(30)

    pygame.quit()
