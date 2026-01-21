# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 19:22:25 2025

@author: tshepisom
"""

import numpy as np
import time

# -----------------------------
# Differential Evolution (Custom)
# -----------------------------
def differential_evolution_custom(func, bounds, pop_size=8, generations=200, F=0.8, CR=0.7):
    dim = len(bounds)  # Number of variables/dimensions

    # -----------------------------
    # Initialize Population
    # -----------------------------
    # Randomly generate initial population within bounds
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    
    # Evaluate initial fitness of each individual
    fitness = np.array([func(ind) for ind in pop])
    
    start_time = time.time()  # Record start time for runtime measurement

    # -----------------------------
    # Main Evolution Loop
    # -----------------------------
    for gen in range(generations):
        for i in range(pop_size):
            # -----------------------------
            # Mutation
            # -----------------------------
            # Choose 3 distinct individuals different from current i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            # Generate mutant vector using differential mutation formula
            # v_i = a + F * (b - c)
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
            # Clip to stay within variable bounds

            # -----------------------------
            # Crossover
            # -----------------------------
            # Decide which dimensions to take from mutant vector
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                # Ensure at least one dimension comes from mutant
                cross_points[np.random.randint(0, dim)] = True

            # Create trial vector by mixing mutant and current individual
            trial = np.where(cross_points, mutant, pop[i])

            # -----------------------------
            # Selection
            # -----------------------------
            # Evaluate trial vector fitness
            f_trial = func(trial)
            # Replace individual if trial is better
            if f_trial < fitness[i]:
                pop[i], fitness[i] = trial, f_trial

    # -----------------------------
    # Return Best Solution
    # -----------------------------
    best_idx = np.argmin(fitness)  # Index of best individual
    total_time = time.time() - start_time
    print(f"DE runtime: {total_time:.3f} s")
    print(f"Best DE solution: {pop[best_idx]}, f={fitness[best_idx]:.6f}")
    
    # Return best individual, its fitness, and fitness array of population
    return pop[best_idx], fitness[best_idx], fitness