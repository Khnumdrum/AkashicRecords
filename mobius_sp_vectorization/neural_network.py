import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pathlib
# THIS_DIR = pathlib.Path(__file__).parent
MEMORY_FILE = "learned_patterns.json"

# MemoryManager class for tracking and pruning state memory
class MemoryManager:
    def __init__(self, threshold=0.01):
        self.memory = []  # A list to store past vectors
        self.threshold = threshold  # Minimum difference to keep a state
    
    def store_vector(self, new_vector):
        """
        Store a new vector, but first check if it's redundant (i.e., similar to an existing vector).
        If it's too similar, delete the oldest one.
        """
        # Check if the new vector is significantly different from the stored ones
        if len(self.memory) > 0:
            differences = [np.linalg.norm(new_vector - v) for v in self.memory]
            if min(differences) < self.threshold:  # If the new vector is too similar, ignore it
                return
        
        # If we are here, the new vector is not redundant
        if len(self.memory) >= 10:  # Set a memory limit
            self.memory.pop(0)  # Remove the oldest one
        
        self.memory.append(new_vector)  # Add the new vector to memory
    
    def get_memory(self):
        """
        Returns the current memory as a list of stored vectors.
        """
        return self.memory

# Neural Network to make decisions about retention/deletion
class ColorMemoryCNN(nn.Module):
    def __init__(self):
        super(ColorMemoryCNN, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # Input is a 3D RGB vector
        self.fc2 = nn.Linear(128, 64)  # Intermediate layer
        self.fc3 = nn.Linear(64, 1)  # Output (e.g., decision to retain or delete)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Output between 0 and 1, decide retention or deletion

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