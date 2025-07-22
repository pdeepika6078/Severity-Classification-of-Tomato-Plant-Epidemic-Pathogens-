import time
import numpy as np


def random_select(v1, v2, alpha, betha):
    return alpha * v1 + betha * v2


def exploration(vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
    return vulture_X + p1 * F * (random_vulture_X - vulture_X)


def exploitation(vulture_X, best_vulture1_X, best_vulture2_X, random_vulture_X, F, p2, p3, variables_no, upper_bound,
                 lower_bound):
    return vulture_X + p2 * F * (best_vulture1_X - best_vulture2_X) + p3 * F * (random_vulture_X - vulture_X)


def boundary_check(X, lower_bound, upper_bound):
    return np.clip(X, lower_bound, upper_bound)


# African Vultures Optimization (AVO)
def PROPOSED(X, fobj, lower_bound, upper_bound, max_iter):
    pop_size, variables_no = X.shape[0], X.shape[1]

    # Initialize Best_vulture1 and Best_vulture2
    Best_vulture1_X = np.zeros(variables_no)
    Best_vulture1_F = float('inf')
    Best_vulture2_X = np.zeros(variables_no)
    Best_vulture2_F = float('inf')

    # Initialize the first random population of vulture
    # Controlling parameters
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    alpha = 0.8
    betha = 0.2
    gamma = 2.5

    # Main loop
    current_iter = 0  # Loop counter
    # List to store convergence curve
    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    while current_iter < max_iter:
        for i in range(X.shape[0]):
            # Calculate the fitness of the population
            current_vulture_X = X[i, :]
            current_vulture_F = fobj(current_vulture_X)

            # Update the best vultures
            if current_vulture_F < Best_vulture1_F:
                Best_vulture1_F = current_vulture_F
                Best_vulture1_X = current_vulture_X.copy()
            elif current_vulture_F > Best_vulture1_F and current_vulture_F < Best_vulture2_F:
                Best_vulture2_F = current_vulture_F
                Best_vulture2_X = current_vulture_X.copy()

        a = np.random.uniform(-2, 2) * ((np.sin((np.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (np.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a

        # Update the location of vultures
        for i in range(X.shape[0]):
            current_vulture_X = X[i, :]
            F = P1 * (2 * np.random.rand() - 1)

            random_vulture_X = random_select(Best_vulture1_X, Best_vulture2_X, alpha, betha)

            if abs(F) >= 1:  # Exploration
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)
            elif abs(F) < 1:  # Exploitation
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X,
                                                 F, p2, p3, variables_no, upper_bound, lower_bound)

            X[i, :] = current_vulture_X
        Convergence_curve[t] = Best_vulture1_F
        t += 1
        X = boundary_check(X, lower_bound, upper_bound)

    return Best_vulture1_X, Convergence_curve, Best_vulture1_F, ct
