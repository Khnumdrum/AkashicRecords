import pygame
import random

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
GREY = (128, 128, 128)

# Screen dimensions and grid setup
WIDTH, HEIGHT = 800, 800
TILE_SIZE = 20
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

def precompute_fibonacci_triplets():
    """Precomputes all valid RGB triplets with consecutive Fibonacci values."""
    triplets = []
    for i in range(len(FIBONACCI) - 2):
        r, g, b = FIBONACCI[i], FIBONACCI[i + 1], FIBONACCI[i + 2]
        if r <= 255 and g <= 255 and b <= 255:
            triplets.append((r, g, b))
    return triplets

# Precompute valid Fibonacci triplets
VALID_FIBONACCI_TRIPLETS = precompute_fibonacci_triplets()

def find_valid_fibonacci_rgb_triplet():
    """Returns a random valid Fibonacci RGB triplet from precomputed values."""
    if VALID_FIBONACCI_TRIPLETS:
        return random.choice(VALID_FIBONACCI_TRIPLETS)
    else:
        print("Error: No valid Fibonacci triplets precomputed.")
        return generate_random_rgb_triplet()

def generate_random_rgb_triplet():
    """Generates a random RGB triplet."""
    return [random.randint(0, 255) for _ in range(3)]

def gen(num):
    """Generates a set of random grid positions."""
    return set([(random.randrange(0, GRID_HEIGHT), random.randrange(0, GRID_WIDTH)) for _ in range(num)])

def draw_grid(positions):
    """Draws the grid with random or Fibonacci colors."""
    try:
        for position in positions:
            col, row = position
            top_left = (col * TILE_SIZE, row * TILE_SIZE)
            color = find_valid_fibonacci_rgb_triplet()  # Use precomputed Fibonacci triplets
            pygame.draw.rect(screen, color, (*top_left, TILE_SIZE, TILE_SIZE))
    except Exception as e:
        print(f"Error: {e}")
        pass

    # Draw grid lines
    for row in range(GRID_HEIGHT):
        pygame.draw.line(screen, BLACK, (0, row * TILE_SIZE), (WIDTH, row * TILE_SIZE))
    for col in range(GRID_WIDTH):
        pygame.draw.line(screen, BLACK, (col * TILE_SIZE, 0), (col * TILE_SIZE, HEIGHT))

def adjust_grid(positions):
    """Adjusts the grid based on Game of Life rules."""
    all_neighbors = set()
    new_positions = set()
    try:
        for position in positions:
            neighbors = get_neighbors(position)
            all_neighbors.update(neighbors)

            neighbors = list(filter(lambda x: x in positions, neighbors))

            if len(neighbors) in [2, 3]:
                new_positions.add(position)
    except Exception as e:
        print(f"Error: {e}")
        pass

    for position in all_neighbors:
        neighbors = get_neighbors(position)
        neighbors = list(filter(lambda x: x in positions, neighbors))

        if len(neighbors) == 3:
            new_positions.add(position)

    return new_positions

def get_neighbors(pos):
    """Gets all neighbors for a given position."""
    x, y = pos
    neighbors = []
    for dx in [-1, 0, 1]:
        if x + dx < 0 or x + dx >= GRID_WIDTH:
            continue
        for dy in [-1, 0, 1]:
            if y + dy < 0 or y + dy >= GRID_HEIGHT:
                continue
            if dx == 0 and dy == 0:
                continue
            neighbors.append((x + dx, y + dy))
    return neighbors

def main():
    """Main loop of the Game of Life."""
    running = True
    playing = False
    count = 0
    update_freq = 120

    positions = set()

    while running:
        clock.tick(FPS)

        if playing:
            count += 1

        if count >= update_freq:
            count = 0
            positions = adjust_grid(positions)

        pygame.display.set_caption("Playing" if playing else "Paused")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col = x // TILE_SIZE
                row = y // TILE_SIZE
                pos = (col, row)

                if pos in positions:
                    positions.remove(pos)
                else:
                    positions.add(pos)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                if event.key == pygame.K_c:
                    positions = set()
                    playing = False
                if event.key == pygame.K_g:
                    positions = gen(random.randrange(5, 7) * GRID_WIDTH)

        screen.fill(GREY)
        draw_grid(positions)
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
