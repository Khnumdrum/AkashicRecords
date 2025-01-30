import numpy as np
def mobius_transform(rgb_vector):
    # Normalize the vector to a range [0, 1] to avoid large numbers
    norm_vector = rgb_vector / np.linalg.norm(rgb_vector)
    
    # Simple Möbius-like transformation
    x, y, z = norm_vector
    transformed = np.array([x + 2 * y * np.sin(np.pi * z),
                            y + 2 * z * np.cos(np.pi * x),
                            z + 2 * x * np.sin(np.pi * y)])
    
    # Normalize the transformed vector
    return transformed / np.linalg.norm(transformed)