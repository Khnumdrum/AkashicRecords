import pygame
import numpy as np
import random
import json
import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim

from game_of_life.neural_network import MemoryManager, ColorMemoryCNN, load_learned_pattern, save_pattern_to_memory
from transforms import mo
# Constants
CELL_SIZE = 105
ROWS, COLS =55, 55
SCREEN_WIDTH, SCREEN_HEIGHT = 1920,1080 #= COLS * CELL_SIZE, ROWS * CELL_SIZE
FPS = 55
NUM_FIXED_ENTITIES = 5  # Number of fixed entities in the grid
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

# Memory storage for learned patterns
MEMORY_FILE = "learned_patterns.json"

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
        move_x = random.choice([-1, 0, 1]) #i changed this y axis to make it matirical
        move_y = random.choice([1, 0, -1])
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

def extract_rgb_vector(grid, color_map):
    r_count = g_count = b_count = 0
    for row in range(ROWS):
        for col in range(COLS):
            if grid[row, col]:
                r, g, b = color_map[row, col]
                r_count += r
                g_count += g
                b_count += b
    return np.array([r_count, g_count, b_count])  # Return as a 3D vector

# Initialize grid and color map
def initialize_grid():
    grid = np.random.choice([False, True], size=(ROWS, COLS), p=[0.8, 0.2])
    color_map = np.array([[generate_random_color() if grid[row, col] else [0, 0, 0] for col in range(COLS)] for row in range(ROWS)])
    return grid, color_map

# Drawing function (adjust based on your existing one)
def draw_grid(screen, grid, color_map):
    for row in range(ROWS):
        for col in range(COLS):
            color = tuple(color_map[row, col])
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

def main():
    grid, color_map = initialize_grid()
    
    # Initialize memory manager
    #memory_manager = MemoryManager()
    
    # Initialize neural network model
    # model = ColorMemoryCNN()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.BCELoss()
    
    fixed_entities = initialize_fixed_entities()

    running = True
    paused = False
    assimilation_in_progress = False  # Flag to indicate assimilation

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
                    grid, color_map = initialize_grid()
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
            # Extract the RGB vector from the grid
            # rgb_vector = extract_rgb_vector(grid, color_map)
            
            # # Apply Möbius transformation
            # transformed_vector = mobius_transform(rgb_vector)
            
            # # Prepare the input tensor for the neural network
            # input_tensor = torch.tensor(transformed_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # # Forward pass through the network
            # output = model(input_tensor)
            
            # # Decision based on network output
            # if output.item() > 0.5:
            #     print("Retaining state")
            #     memory_manager.store_vector(transformed_vector)
            # else:
            #     print("Deleting state")
            
            # Example of stored memory
            # print("Memory:", memory_manager.get_memory())

            grid, color_map = update_grid(grid, color_map)
            fixed_entities = move_fixed_entities(fixed_entities)
            grid, color_map = propagate_with_communication(grid, color_map, fixed_entities)

            # Assimilation process: attempting to recreate the user input
            # if assimilation_in_progress:
                # Check if the last pattern has been recreated (in the "assimilation" phase)
                # print("Assimilation sequence initiated.")
                # Implement logic for assimilating based on propagation and game of life rules

        draw_grid(screen, grid, color_map)
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
