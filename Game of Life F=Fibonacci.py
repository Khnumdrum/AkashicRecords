
import random
import pygame
import numpy as np

# Function to generate vibrant random colors
def generate_random_color():
    """Generate vibrant random colors."""
    r, g, b = [random.randint(100, 255) for _ in range(3)]
    # Ensure there's sufficient difference between RGB components to avoid muted tones
    if abs(r - g) < 50 and abs(g - b) < 50 and abs(b - r) < 50:
        return generate_random_color()
    return [r, g, b]

# Example: Update color_map during propagation
def update_grid(grid, color_map):
    rows, cols = grid.shape
    new_grid = np.zeros((rows, cols), dtype=bool)
    new_color_map = np.zeros((rows, cols, 3), dtype=int)

    for row in range(rows):
        for col in range(cols):
            neighbors = np.sum(grid[max(0, row - 1):row + 2, max(0, col - 1):col + 2]) - grid[row, col]
            if grid[row, col] and (neighbors == 2 or neighbors == 3):
                new_grid[row, col] = True
                new_color_map[row, col] = color_map[row, col]  # Retain existing color
            elif not grid[row, col] and neighbors == 3:
                new_grid[row, col] = True
                new_color_map[row, col] = generate_random_color()  # Generate vibrant color
            else:
                new_color_map[row, col] = [0, 0, 0]  # Default black for dead cells
    return new_grid, new_color_map

# Function to draw the grid
def draw_grid(screen, grid, color_map, cell_size):
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            color = tuple(color_map[row, col])
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)

# Example usage with Pygame
def main():
    pygame.init()
    cell_size = 10
    rows, cols = 50, 50
    screen_width, screen_height = cols * cell_size, rows * cell_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    # Initialize the grid and color map
    grid = np.random.choice([False, True], size=(rows, cols), p=[0.8, 0.2])
    color_map = np.array([[generate_random_color() if grid[row, col] else [0, 0, 0]
                           for col in range(cols)] for row in range(rows)])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        draw_grid(screen, grid, color_map, cell_size)
        grid, color_map = update_grid(grid, color_map)
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
