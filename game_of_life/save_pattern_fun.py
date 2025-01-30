# Additional State Variables
learning_mode = False  # Tracks whether we are in symbol learning mode

def enter_learning_mode():
    """
    Activate the symbol learning mode. Allows the user to draw a symbol
    on the grid while paused, and save it to memory with a label.
    """
    global learning_mode
    learning_mode = True
    print("Learning mode activated. Draw your symbol and press 'S' to save.")

def exit_learning_mode():
    """
    Deactivate the symbol learning mode.
    """
    global learning_mode
    learning_mode = False
    print("Learning mode exited.")

def recall_menu():
    """
    Displays a menu of learned symbols, allowing users to delete or edit entries.
    """
    if not os.path.exists(MEMORY_FILE):
        print("No learned symbols found.")
        return

    with open(MEMORY_FILE, "r") as file:
        memory_data = json.load(file)

    print("Learned Symbols:")
    for i, pattern in enumerate(memory_data):
        print(f"{i + 1}: {pattern['label']}")

    choice = input("Enter the number to delete/edit, or 'q' to quit: ")
    if choice.isdigit():
        choice = int(choice) - 1
        if 0 <= choice < len(memory_data):
            action = input("Enter 'd' to delete or 'e' to edit: ").lower()
            if action == "d":
                del memory_data[choice]
                print("Symbol deleted.")
            elif action == "e":
                label = input("Enter a new label for this symbol: ")
                memory_data[choice]["label"] = label
                print("Symbol updated.")
            with open(MEMORY_FILE, "w") as file:
                json.dump(memory_data, file, indent=4)
        else:
            print("Invalid choice.")
    elif choice.lower() != "q":
        print("Invalid input.")

# Main Loop Updates
def main():
    # Inside the event handling loop
    global paused, learning_mode

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            # Unique Key Handling
            if paused:  # Learning mode can only be activated when paused
                if event.key == pygame.K_l and not learning_mode:
                    enter_learning_mode()
                elif event.key == pygame.K_q and learning_mode:
                    exit_learning_mode()
                elif event.key == pygame.K_r:
                    recall_menu()

            if event.key == pygame.K_s and learning_mode:
                # Save drawn symbol
                label = input("Enter a label for your symbol: ")
                save_pattern_to_memory(grid, color_map, label)

    # Update grid logic and other UI elements
    if not paused and not learning_mode:
        grid, color_map = (grid, color_map)

    # Render updated screen
    draw_grid(screen, grid, color_map)
    if learning_mode:
        font = pygame.font.Font(None, 24)
        label = font.render("Learning Mode Active", True, (255, 255, 0))
        screen.blit(label, (10, SCREEN_HEIGHT - 30))
    else:
        draw_legend(screen)