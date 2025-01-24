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
import json
import os

# Memory storage for learned patterns
MEMORY_FILE = "learned_patterns.json"

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

import json
import os
# Memory storage for learned patterns
MEMORY_FILE = "learned_patterns.json"
def save_pattern_to_memory(grid, color_map, label):
    """
    Saves a specific grid pattern and its corresponding colors to memory.
    Args:
        grid (ndarray): Current grid state.
        color_map (ndarray): Current color map.
        label (str): A label to identify the saved pattern.
    """
    pattern = {
        "label": label,
        "grid": grid.tolist(),
        "color_map": color_map.tolist()
    }
    if not os.path.exists(MEMORY_FILE):
        memory_data = []
    else:
        with open(MEMORY_FILE, "r") as file:
            memory_data = json.load(file)
    # Check for duplicate labels
    memory_data = [p for p in memory_data if p["label"] != label]
    memory_data.append(pattern)
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory_data, file, indent=4)
    print(f"Pattern '{label}' saved successfully.")
def load_learned_pattern(label):
    """
    Loads a specific grid pattern by its label.
    Args:
        label (str): The label identifying the pattern to load.
    Returns:
        grid, color_map: The pattern's grid and color map, or None if not found.
    """
    if not os.path.exists(MEMORY_FILE):
        print("No memory file found.")
        return None, None
    with open(MEMORY_FILE, "r") as file:
        memory_data = json.load(file)
    for pattern in memory_data:
        if pattern["label"] == label:
            print(f"Pattern '{label}' loaded.")
            return np.array(pattern["grid"], dtype=bool), np.array(pattern["color_map"], dtype=int)
    print(f"Pattern '{label}' not found.")
    return None, None
def match_pattern_in_grid(grid):
    """
    Checks if a section of the current grid matches any stored patterns.
    Args:
        grid (ndarray): Current grid state.
    Returns:
        matched_label (str): The label of the matching pattern, or None if no match.
    """
    if not os.path.exists(MEMORY_FILE):
        return None
    with open(MEMORY_FILE, "r") as file:
        memory_data = json.load(file)
    for pattern in memory_data:
        stored_grid = np.array(pattern["grid"], dtype=bool)
        if np.array_equal(grid[:stored_grid.shape[0], :stored_grid.shape[1]], stored_grid):
            return pattern["label"]
    return None
def propagate_with_long_memory(grid, color_map):
    """
    Enhanced propagation algorithm using learned patterns and self-preservation bias.
    Args:
        grid (ndarray): Current grid state.
        color_map (ndarray): Current color map.
    Returns:
        grid, color_map: Updated grid and color map.
    """
    # Identify if a part of the grid matches a stored pattern
    matched_label = match_pattern_in_grid(grid)
    if matched_label:
        print(f"Pattern '{matched_label}' recognized. Propagating...")
        learned_grid, learned_color_map = load_learned_pattern(matched_label)
        if learned_grid is not None:
            grid[:learned_grid.shape[0], :learned_grid.shape[1]] = learned_grid
            color_map[:learned_color_map.shape[0], :learned_color_map.shape[1]] = learned_color_map
    # Introduce propagation bias (self-preservation)
    for row in range(ROWS):
        for col in range(COLS):
            neighbors = np.sum(grid[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) - grid[row, col]
            if not grid[row, col] and neighbors >= 2:
                if random.random() < 0.3:  # Adjust bias strength
                    grid[row, col] = True
                    color_map[row, col] = generate_random_color()
    return grid, color_map

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
                if event.key == pygame.K_r:  # Random grid
                    key_function.key = "R"
                    grid = np.random.choice([False, True], size=(ROWS, COLS), p=[0.8, 0.2])
                    color_map = np.array([[generate_random_color() if grid[row, col] else [0, 0, 0]
                                           for col in range(COLS)] for row in range(ROWS)])
                if event.key == pygame.K_p:  # Pause/Play
                    key_function.key = "P"
                    paused = not paused
                if event.key == pygame.K_c:  # Clear grid
                    key_function.key = "C"
                    grid.fill(False)
                    color_map.fill(0)
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