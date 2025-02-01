import numpy as np
import cupy as cp
import time
import pandas as pd

# Generate synthetic test signals
def generate_test_data(size=10000):
    np.random.seed(42)
    base_signal = np.sin(np.linspace(0, 10, size))  # Smooth sinusoidal data
    noise = np.random.normal(0, 0.1, size)  # Random noise
    spikes = np.zeros(size)
    spikes[np.random.randint(0, size, size // 100)] = np.random.uniform(-1, 1, size // 100)  # Sparse spikes
    return base_signal + noise + spikes

def benchmark_function(func, data, iterations=10):
    start_time = time.time()
    processed = func(cp.array(data), iterations)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, processed

# Dummy traditional method for comparison
def traditional_method(data, iterations=10):
    for _ in range(iterations):
        data = np.log1p(np.abs(data))  # Basic transformation
    return data

# Import Harmonia functions
from harmonic_data_processing import process_data

# Benchmark setup
sizes = [1000, 5000, 10000, 50000]
results = []

for size in sizes:
    data = generate_test_data(size)
    
    # Benchmark Harmonia
    harmonia_time, harmonia_output = benchmark_function(process_data, data)
    
    # Benchmark Traditional
    start_time = time.time()
    traditional_output = traditional_method(data)
    traditional_time = time.time() - start_time
    
    # Calculate adaptability score (RMSE + adjustment time factor)
    rmse = np.sqrt(np.mean((harmonia_output["Final Processed Data"] - traditional_output)**2))
    adaptability_score = 100 - (rmse * 100) / np.max(np.abs(traditional_output))
    
    results.append({
        "Dataset Size": size,
        "Harmonia Time (s)": harmonia_time,
        "Traditional Time (s)": traditional_time,
        "Speed Improvement (%)": ((traditional_time - harmonia_time) / traditional_time) * 100,
        "Adaptability Score (%)": adaptability_score
    })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("harmonia_benchmark_results.csv", index=False)

# Display results
print(df)
