import time

import numpy as np


# Update the leader
def update_leader(population, fitness):
    best_idx = np.argmin(fitness)
    return population[best_idx, :].copy()


# Update team members
def update_members(leader, members):
    alpha = 0.5  # Influence of the leader
    beta = 0.5  # Influence of the team mean
    return alpha * leader + beta * np.mean(members, axis=0)


# Teamwork Optimization Algorithm
def TOA(population, VRmin, VRmax, objective_function, Max_iter):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    best_solution = None
    best_fitness = float('inf')
    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    # TOA loop
    for gen in range(Max_iter):
        # Evaluate fitness
        fitness = np.apply_along_axis(objective_function, 1, population)

        # Update leader
        leader = update_leader(population, fitness)

        # Divide population into teams
        members = np.delete(population, np.argmin(fitness), axis=0)

        # Update team members
        new_members = update_members(leader, members)

        # Combine leader and updated members
        new_population = np.vstack((leader, new_members))

        # Select the best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]

        population = new_population
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_solution, ct
