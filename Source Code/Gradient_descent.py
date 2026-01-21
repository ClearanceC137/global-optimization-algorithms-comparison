# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 19:22:20 2025

@author: tshepisom
"""
import numpy as np
import time
# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(func, bounds, lr=0.01, max_iter=500, epsilon=1e-6, verbose=True, print_every=50):
    # Randomly initialize starting point x within the given bounds
    x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    
    history = []                     # Store function values at each iteration
    best_x = np.copy(x)              # Keep track of the best solution found
    best_val = func(x)               # Evaluate initial function value
    start_time = time.time()         # Record start time for runtime

    # -----------------------------
    # Numerical Gradient Calculation
    # -----------------------------
    def numerical_grad(f, x, h=1e-5):
        grad = np.zeros_like(x)      # Gradient vector initialization
        for i in range(len(x)):
            x1, x2 = np.copy(x), np.copy(x)
            x1[i] += h               # Perturb variable i positively
            x2[i] -= h               # Perturb variable i negatively
            # Central difference approximation of partial derivative
            grad[i] = (f(x1) - f(x2)) / (2 * h)
        return grad

    # -----------------------------
    # Main Gradient Descent Loop
    # -----------------------------
    for iteration in range(max_iter):
        grad = numerical_grad(func, x)        # Compute gradient at current x
        x_new = x - lr * grad                 # Update rule: move opposite to gradient
        step_size = np.linalg.norm(x_new - x) # Measure change in x (for convergence check)
        x = x_new                             
        current_val = func(x)                  # Evaluate function at new x
        history.append(current_val)            # Store function value

        # -----------------------------
        # Update Best Solution
        # -----------------------------
        if current_val < best_val:
            best_val = current_val
            best_x = np.copy(x)

        # Print progress periodically
        if verbose and (iteration % print_every == 0 or iteration == max_iter - 1):
            print(f"Iteration {iteration:4d} | f(x) = {current_val:.6f} | Step = {step_size:.6e}")

        # Convergence check: stop if step size is smaller than epsilon
        if step_size < epsilon:
            print(f"Converged after {iteration} iterations.")
            break

    # -----------------------------
    # Print total runtime and return results
    # -----------------------------
    total_time = time.time() - start_time
    print(f"GD runtime: {total_time:.3f} s\n")
    
    return best_x, best_val, history   # Return best solution, its value, and function history