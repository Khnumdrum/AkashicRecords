
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
NUM_FIXED_ENTITIES = 3  # Number of fixed entities in the grid
LEGEND_MINIMIZED = False  # Track if the legend is minimized

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Enhanced Game of Life")
clock = pygame.time.Clock()

# KeyFunction class
class KeyFunction:
    def __init__(self):
        self.active_keys = {"R": False, "P": False, "C": False, "L": False, "M": False, "D": False, "SHIFT": False}

    def toggle_key(self, key):
        if key in self.active_keys:
            self.active_keys[key] = not self.active_keys[key]

key_function = KeyFunction()

# Functions for memory storage and pattern manipulation
def save_pattern_to_memory(grid, color_map, label):
    if not os.path.exists(MEMORY_FILE):
        memory_data = []
    else:
        try:
            with open(MEMORY_FILE, "r") as file:
                memory_data = json.load(file)
        except FileNotFoundError:
            memory_data = []
    
    memory_data = [p for p in memory_data if p["label"] != label]  # Check for duplicates
    pattern = {
        "label": label,
        "grid": grid.tolist(),
        "color_map": color_map.tolist()
    }
    memory_data.append(pattern)
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory_data, file, indent=4)
    print(f"Pattern '{label}' saved successfully.")

def load_learned_pattern(label):
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

def generate_random_color():
    r, g, b = [random.randint(100, 255) for _ in range(3)]
    if abs(r - g) < 50 and abs(g - b) < 50 and abs(b - r) < 50:
        return generate_random_color()
    return [r, g, b]

def initialize_fixed_entities():
    fixed_entities = []
    for _ in range(NUM_FIXED_ENTITIES):
        x, y = random.randint(5, COLS - 6), random.randint(5, ROWS - 6)
        color = generate_random_color()
        fixed_entities.append({"x": x, "y": y, "color": color, "size": 1, "propagation_rate": 0.05, "memory": []})
    return fixed_entities

def move_fixed_entities(fixed_entities):
    for entity in fixed_entities:
        # Random movement with bounds check
        move_x = random.choice([-1, 0, 1])
        move_y = random.choice([-1, 0, 1])
        entity["x"] = max(0, min(COLS - 1, entity["x"] + move_x))
        entity["y"] = max(0, min(ROWS - 1, entity["y"] + move_y))
    return fixed_entities

def propagate_with_communication(grid, color_map, fixed_entities):
    # Propagate form and color based on fixed entities
    for entity in fixed_entities:
        # Spread to nearby cells in a neighborhood based on entity's propagation rate and size
        for dx in range(-entity["size"], entity["size"] + 1):
            for dy in range(-entity["size"], entity["size"] + 1):
                nx, ny = entity["x"] + dx, entity["y"] + dy
                if 0 <= nx < COLS and 0 <= ny < ROWS:
                    if not grid[ny, nx]:  # If the cell is not already active
                        grid[ny, nx] = True  # Activate the cell
                        color_map[ny, nx] = entity["color"]  # Propagate color
        entity["memory"].append({"grid": grid.tolist(), "color_map": color_map.tolist()})  # Store current state in memory
    return grid, color_map

def update_grid(grid, color_map):
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
    col, row = x // CELL_SIZE, y // CELL_SIZE
    if 0 <= row < ROWS and 0 <= col < COLS:
        grid[row, col] = False
        color_map[row, col] = [0, 0, 0]

def draw_grid(screen, grid, color_map):
    for row in range(ROWS):
        for col in range(COLS):
            color = tuple(color_map[row, col])
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

def draw_legend(screen):
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
        "default": (255, 255, 255),
        "pressed": (255, 0, 0),
        "active": (0, 255, 0)
    }

    if not LEGEND_MINIMIZED:  # Only draw legend if it's not minimized
        for i, text in enumerate(legend_texts):
            key = text[0]
            color = key_colors["active"] if key in key_function.active_keys and key_function.active_keys[key] else key_colors["default"]
            label = font.render(text, True, color)
            screen.blit(label, (10, 10 + i * 20))

def main():
    grid = np.random.choice([False, True], size=(ROWS, COLS), p=[0.8, 0.2])
    color_map = np.array([[generate_random_color() if grid[row, col] else [0, 0, 0]
                           for col in range(COLS)] for row in range(ROWS)])

    fixed_entities = initialize_fixed_entities()

    running = True
    paused = False

    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key).upper()
                if key in key_function.active_keys:
                    key_function.toggle_key(key)
                if key == "L":  
                    label = input("Enter label to load pattern: ")
                    grid, color_map = load_learned_pattern(label)
                if key == "S":  
                    label = input("Enter label to save pattern: ")
                    save_pattern_to_memory(grid, color_map, label)
                if event.key == pygame.K_r:  
                    grid = np.random.choice([False, True], size=(ROWS, COLS), p=[0.8, 0.2])
                    color_map = np.array([[generate_random_color() if grid[row, col] else [0, 0, 0]
                                           for col in range(COLS)] for row in range(ROWS)])
                if event.key == pygame.K_p:  
                    paused = not paused
                if event.key == pygame.K_c:  
                    grid.fill(False)
                    color_map.fill(0)
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:  
                    global LEGEND_MINIMIZED
                    LEGEND_MINIMIZED = not LEGEND_MINIMIZED  # Toggle the state
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:  
                x, y = pygame.mouse.get_pos()
                remove_cell(grid, color_map, x, y)

        if pygame.mouse.get_pressed()[0]:  
            x, y = pygame.mouse.get_pos()
            col, row = x // CELL_SIZE, y // CELL_SIZE
            if 0 <= row < ROWS and 0 <= col < COLS:
                grid[row, col] = True
                color_map[row, col] = generate_random_color()

        if not paused:
            grid, color_map = update_grid(grid, color_map)
            fixed_entities = move_fixed_entities(fixed_entities)
            grid, color_map = propagate_with_communication(grid, color_map, fixed_entities)

        draw_grid(screen, grid, color_map)
        draw_legend(screen)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()