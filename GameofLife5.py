"""
Enhanced Game of Life Program:
- Adds right-click functionality to remove cells.
- Introduces a learning algorithm with self-preservation bias.
- Expands the "Load Learned Symbols" feature for dynamic usage.
Author: Michael A Fry (github Khnumdrum)
      Special thanks to Sistere and Mu
"""

import pygame
import numpy as np
import random

# Constants
CELL_SIZE = 10
ROWS, COLS = 50, 50
SCREEN_WIDTH, SCREEN_HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE


class KeyFunction:
    """
    Manages the state of toggleable keys for the program.

    Attributes:
        active_keys (dict): Tracks keys and their active states (True/False).
    """
    def __init__(self):
        self.active_keys = {"R": False, "P": False, "C": False, "L": False, "M": False, "D": False}

    def toggle_key(self, key):
        """
        Toggles the active state of a key.

        Args:
            key (str): The key to toggle (e.g., "P" for Pause).
        """
        if key in self.active_keys:
            self.active_keys[key] = not self.active_keys[key]


key_function = KeyFunction()


def generate_random_color():
    """Generate vibrant random colors."""
    r, g, b = [random.randint(100, 255) for _ in range(3)]
    if abs(r - g) < 50 and abs(g - b) < 50 and abs(b - r) < 50:
        return generate_random_color()
    return [r, g, b]


def update_grid(grid, color_map):
    """
    Updates the grid based on the rules of the Game of Life.

    Args:
        grid (ndarray): Current grid state (True/False).
        color_map (ndarray): Current color map.

    Returns:
        new_grid, new_color_map: Updated grid and color map.
    """
    new_grid = np.zeros_like(grid, dtype=bool)
    new_color_map = np.zeros((ROWS, COLS, 3), dtype=int)

    for row in range(ROWS):
        for col in range(COLS):
            neighbors = np.sum(grid[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) - grid[row, col]
            if grid[row, col] and (neighbors == 2 or neighbors == 3):
                new_grid[row, col] = True
                new_color_map[row, col] = color_map[row, col]
            elif not grid[row, col] and neighbors == 3:
                new_grid[row, col] = True
                new_color_map[row, col] = generate_random_color()
            else:
                new_color_map[row, col] = [0, 0, 0]
    return new_grid, new_color_map


def remove_cell(grid, color_map, x, y):
    """
    Removes a cell at a specific position (right-click functionality).

    Args:
        grid (ndarray): Current grid state.
        color_map (ndarray): Current color map.
        x (int): X-coordinate of the cell.
        y (int): Y-coordinate of the cell.
    """
    col, row = x // CELL_SIZE, y // CELL_SIZE
    if 0 <= row < ROWS and 0 <= col < COLS:
        grid[row, col] = False
        color_map[row, col] = [0, 0, 0]


def learn_patterns(grid, color_map):
    """
    Implements a learning algorithm to propagate cells with a bias towards self-preservation.

    Args:
        grid (ndarray): Current grid state.
        color_map (ndarray): Current color map.

    Returns:
        grid, color_map: Updated grid and color map.
    """
    for row in range(ROWS):
        for col in range(COLS):
            neighbors = np.sum(grid[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) - grid[row, col]
            if not grid[row, col] and neighbors >= 2:
                # Bias towards creating new cells in dense areas
                if random.random() < 0.3:  # Adjust bias strength
                    grid[row, col] = True
                    color_map[row, col] = generate_random_color()
    return grid, color_map


def draw_grid(screen, grid, color_map):
    """
    Draws the grid on the screen.

    Args:
        screen: Pygame display surface.
        grid (ndarray): Current grid state.
        color_map (ndarray): Current color map.
    """
    for row in range(ROWS):
        for col in range(COLS):
            color = tuple(color_map[row, col])
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)


def draw_legend(screen):
    """
    Draws the legend with dynamic toggle indicators.

    Args:
        screen: Pygame display surface.
    """
    font = pygame.font.Font(None, 24)
    legend_texts = [
        "Legend:",
        "R - Random grid",
        "P - Pause/Play simulation",
        "C - Clear grid",
        "L - Load learned symbols",
        "M - Toggle Mandelbrot mode",
        "D - Toggle Mandala mode",
        "Left-click/drag - Draw cells",
        "Right-click - Remove cells"
    ]

    key_colors = {
        "default": (255, 255, 255),  # White for inactive
        "pressed": (255, 0, 0),     # Red for recently pressed
        "active": (0, 255, 0)       # Green for persistent active state
    }

    for i, text in enumerate(legend_texts):
        key = text[0]  # Get the key (first letter of the legend text)
        if key in key_function.active_keys and key_function.active_keys[key]:
            color = key_colors["active"]
        else:
            color = key_colors["default"]

        label = font.render(text, True, color)
        screen.blit(label, (10, 10 + i * 20))


def main():
    """
    Main game loop for running the Game of Life program.
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    grid = np.random.choice([False, True], size=(ROWS, COLS), p=[0.8, 0.2])
    color_map = np.array([[generate_random_color() if grid[row, col] else [0, 0, 0]
                           for col in range(COLS)] for row in range(ROWS)])

    running = True
    paused = False

    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key).upper()  # Get the key name
                if key in key_function.active_keys:
                    key_function.toggle_key(key)
                if key == "L":  # Load learned symbols
                    grid, color_map = learn_patterns(grid, color_map)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:  # Right-click
                x, y = pygame.mouse.get_pos()
                remove_cell(grid, color_map, x, y)

        if pygame.mouse.get_pressed()[0]:  # Left-click/drag
            x, y = pygame.mouse.get_pos()
            col, row = x // CELL_SIZE, y // CELL_SIZE
            if 0 <= row < ROWS and 0 <= col < COLS:
                grid[row, col] = True
                color_map[row, col] = generate_random_color()

        if not paused:
            grid, color_map = update_grid(grid, color_map)

        draw_grid(screen, grid, color_map)
        draw_legend(screen)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()