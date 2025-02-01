import numpy as np
import cupy as cp
from fastapi import FastAPI

app = FastAPI()

def logarithmic_inversion(data):
    return cp.log(cp.abs(data) + 1e-10)  # Handling negatives by taking abs and adding epsilon

def exponential_decay_inversion(data):
    return cp.exp(-data)

def thresholding_inversion(data, threshold=1):
    return cp.where(data > threshold, 1, data)

def power_law_inversion(data, exponent=0.5):
    return cp.sign(data) * cp.power(cp.abs(data), exponent)  # Preserve sign for negative values

def feedback_loop(data, iterations=10, initial_weight=0.9):
    results = [data]
    weight = initial_weight
    for _ in range(iterations):
        weight *= 0.95  # Gradient-based adjustment
        data = weight * data + (1 - weight) * cp.mean(data)
        results.append(data)
    return cp.array(results)

def hyperbolic_iteration(data, iterations=10, factor=1.1):
    results = [data]
    for i in range(1, iterations + 1):
        dynamic_factor = factor / (1 + i**0.5)  # Hybrid hyperbolic adjustment
        data = data * dynamic_factor
        results.append(data)
    return cp.array(results)

def imperfect_sphere_inversion(data, threshold=10):
    magnitude = cp.abs(data)
    mask = magnitude > threshold
    data[mask] = threshold**2 / data[mask]  # Sphere inversion for large values
    return data

def process_data(data, iterations=10):
    log_inv = logarithmic_inversion(data)
    exp_inv = exponential_decay_inversion(data)
    thresh_inv = thresholding_inversion(data)
    power_inv = power_law_inversion(data)
    
    feedback_results = feedback_loop(data, iterations)
    hyperbolic_results = hyperbolic_iteration(data, iterations)
    
    sphere_inverted = imperfect_sphere_inversion(hyperbolic_results[-1])
    final_output = sphere_inverted
    
    return {
        "Original Data": data.get(),
        "Logarithmic Inversion": log_inv.get(),
        "Exponential Decay Inversion": exp_inv.get(),
        "Thresholding Inversion": thresh_inv.get(),
        "Power Law Inversion": power_inv.get(),
        "Feedback Loop": feedback_results.get(),
        "Hyperbolic Iteration": hyperbolic_results.get(),
        "Imperfect Sphere Inversion": sphere_inverted.get(),
        "Final Processed Data": final_output.get()
    }

@app.post("/process")
def process_endpoint(data: list, iterations: int = 10):
    data_array = cp.array(data)
    results = process_data(data_array, iterations)
    return results

# Example test data
data = cp.array([9, -6, 3, -6, 9, 100, -50])
iterations = 10
results = process_data(data, iterations)

# Display results
for key, value in results.items():
    print(f"{key}: {value}")
