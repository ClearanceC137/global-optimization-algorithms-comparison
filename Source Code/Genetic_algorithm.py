# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 19:22:22 2025

@author: tshepisom
"""

import numpy as np
import time

# -----------------------------
# Genetic Algorithm
# -----------------------------
def genetic_algorithm(func, bounds, pop_size=6, generations=100, mutation_rate=0.3):
    dim = len(bounds)  # Number of variables/dimensions

    # -----------------------------
    # Initialize Population
    # -----------------------------
    # Each individual is randomly generated within the bounds
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    
    history = []             # Store best fitness of each generation
    start_time = time.time() # Record start time for runtime measurement

    # -----------------------------
    # Main Evolution Loop
    # -----------------------------
    for gen in range(generations):
        # Evaluate fitness of all individuals
        fitness = np.array([func(ind) for ind in pop])
        
        # Find the best individual of this generation
        best_idx = np.argmin(fitness)
        history.append(fitness[best_idx])  # Track best fitness
        
        # -----------------------------
        # Selection
        # -----------------------------
        # Select top 50% of individuals as parents for crossover
        parents = pop[np.argsort(fitness)[:pop_size//2]]

        # -----------------------------
        # Crossover (Recombination)
        # -----------------------------
        offspring = []
        for i in range(pop_size//2):
            # Randomly select two parents
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            if dim == 1:
                # For 1D, child is simple average of parents
                child = np.array([(p1[0] + p2[0]) / 2])
            else:
                # For multi-dimensional, pick a crossover point
                cp = np.random.randint(1, dim)
                # Combine first part from p1, second part from p2
                child = np.concatenate((p1[:cp], p2[cp:]))
            offspring.append(child)

        # -----------------------------
        # Mutation
        # -----------------------------
        offspring = np.array(offspring)
        for i, child in enumerate(offspring):
            # With mutation probability, randomly perturb one gene
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(dim)
                child[idx] += np.random.uniform(-1, 1)  # Small random change

        # -----------------------------
        # Form Next Generation
        # -----------------------------
        # Combine parents and offspring to form new population
        pop = np.vstack((parents, offspring))

    # -----------------------------
    # Return Best Solution
    # -----------------------------
    final_fitness = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(final_fitness)
    best_ind = pop[best_idx]

    # Print runtime and best solution
    print(f"\nGA finished in {time.time() - start_time:.3f} s")
    print(f"Best GA solution: {best_ind}, f={func(best_ind):.6f}")
    
    return best_ind, func(best_ind), history  # Return best individual, its value, and fitness history
