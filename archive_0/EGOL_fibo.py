import pygame
import random
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)

# Screen dimensions and grid setup
WIDTH, HEIGHT = 800, 800
TILE_SIZE = 2
GRID_WIDTH = WIDTH // TILE_SIZE
GRID_HEIGHT = HEIGHT // TILE_SIZE
FPS = 55

# Pygame screen and clock
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Precompute Fibonacci sequence
FIBONACCI = [0, 1]
while FIBONACCI[-1] <= 255:
    FIBONACCI.append(FIBONACCI[-1] + FIBONACCI[-2])

# Generate Fibonacci-based color triplets
def precompute_fibonacci_triplets():
    """Precomputes valid RGB triplets with Fibonacci values."""
    triplets = []
    for i in range(len(FIBONACCI) - 2):
        r, g, b = FIBONACCI[i], FIBONACCI[i + 1], FIBONACCI[i + 2]
        if r <= 255 and g <= 255 and b <= 255:
            triplets.append((r, g, b))
    return triplets

VALID_FIBONACCI_TRIPLETS = precompute_fibonacci_triplets()

def find_valid_fibonacci_rgb_triplet():
    return random.choice(VALID_FIBONACCI_TRIPLETS)

# Procedural cell generation with Fibonacci bias
def generate_procedural_rgb_triplet():
    return find_valid_fibonacci_rgb_triplet() if random.random() < 0.7 else tuple(random.randint(0, 255) for _ in range(3))

def get_neighbors(pos):
    """Gets all neighbors for a given position."""
    x, y = pos
    neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
    return [(nx, ny) for nx, ny in neighbors if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT]

def draw_grid(positions):
    """Draws active tiles with their assigned colors."""
    screen.fill(BLACK)
    for pos, color in positions.items():
        col, row = pos
        pygame.draw.rect(screen, color, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

def adjust_grid(positions, velocities):
    """Adjusts the grid based on Conway's Game of Life rules and applies momentum."""
    new_positions = {}
    new_velocities = {}
    all_neighbors = {n for pos in positions for n in get_neighbors(pos)}
    
    for pos, vel in velocities.items():
        new_pos = (pos[0] + int(vel[0]), pos[1] + int(vel[1]))
        if 0 <= new_pos[0] < GRID_WIDTH and 0 <= new_pos[1] < GRID_HEIGHT:
            new_positions[new_pos] = positions[pos]
            new_velocities[new_pos] = vel
    
    for pos in all_neighbors:
        if pos not in new_positions and sum(1 for n in get_neighbors(pos) if n in positions) == 3:
            new_positions[pos] = generate_procedural_rgb_triplet()
            new_velocities[pos] = (random.uniform(-1, 1), random.uniform(-1, 1))
    
    return new_positions, new_velocities

def apply_rotational_forces(positions, velocities, center):
    """Applies rotational forces to particles based on their distance from the center."""
    for pos in positions:
        dx, dy = pos[0] - center[0], pos[1] - center[1]
        radius = max(math.sqrt(dx**2 + dy**2), 1)
        angle = math.atan2(dy, dx)
        
        rotation_speed = 0.02 * radius  # Increased rotational influence
        angle += rotation_speed
        
        velocities[pos] = (math.cos(angle) * rotation_speed, math.sin(angle) * rotation_speed)

def main():
    """Main loop of the Game of Life."""
    running = True
    playing = False
    positions = {}
    velocities = {}
    center = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    
    while running:
        clock.tick(FPS)
        
        if playing:
            positions, velocities = adjust_grid(positions, velocities)
            apply_rotational_forces(positions, velocities, center)
        
        pygame.display.set_caption("Playing" if playing else "Paused")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // TILE_SIZE, y // TILE_SIZE
                pos = (col, row)
                if pos not in positions:
                    positions[pos] = find_valid_fibonacci_rgb_triplet()
                    velocities[pos] = (random.uniform(-1, 1), random.uniform(-1, 1))
                else:
                    positions.pop(pos)
                    velocities.pop(pos, None)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                if event.key == pygame.K_c:
                    positions.clear()
                    velocities.clear()
                    playing = False
        
        draw_grid(positions)
        pygame.display.update()
    
    pygame.quit()

if __name__ == "__main__":
    main()
