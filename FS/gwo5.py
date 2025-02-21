# https://blog.csdn.net/sfejojno/article/details/135511015
# Multi-Objective Grey Wolf optimizer
# https://www.sciencedirect.com/science/article/pii/S0957417415007435
import numpy as np
from numpy.random import rand

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def is_dominated(p, q):
    """Check if solution p is dominated by solution q."""
    return all(p <= q) and any(p < q)

def non_dominated_sort(population, fitness):
    """Perform non-dominated sorting and return the Pareto front."""
    pareto_front = []
    for i in range(len(population)):
        dominated = False
        for j in range(len(population)):
            if i != j and is_dominated(fitness[i], fitness[j]):
                dominated = True
                break
        if not dominated:
            pareto_front.append(population[i])
    return pareto_front

def crowding_distance(pareto_front, fitness):
    """Calculate the crowding distance for each solution in the Pareto front."""
    distances = np.zeros(len(pareto_front))
    for i in range(len(pareto_front)):
        distances[i] = np.mean(fitness[i])
    return distances

def select_leaders(pareto_front, fitness, num_leaders):
    """Select the least crowded solutions as leaders."""
    distances = crowding_distance(pareto_front, fitness)
    indices = np.argsort(distances)
    leaders = [pareto_front[i] for i in indices[:num_leaders]]
    return leaders

def jfs_mo(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5

    N = opts['N']
    max_iter = opts['T']
    archive_size = 100  # Maximum size of the archive

    # Dimension
    dim = xtrain.shape[1]
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)

    # Initialize archive
    archive = []
    for i in range(N):
        archive.append((X[i, :], fit[i, 0]))

    # Non-dominated sort to initialize the Pareto front
    pareto_front = non_dominated_sort([x for x, _ in archive], [f for _, f in archive])

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = fit.min()
    t += 1

    while t < max_iter:
        # Update parameters
        a = 2 - t * (2 / max_iter)

        # Select leaders from the archive
        leaders = select_leaders(pareto_front, [f for _, f in pareto_front], 3)
        Xalpha, Xbeta, Xdelta = leaders

        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()

                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[d] - X[i, d])
                Dbeta = abs(C2 * Xbeta[d] - X[i, d])
                Ddelta = abs(C3 * Xdelta[d] - X[i, d])

                # Parameter A (3.3)
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a

                # Compute X1, X2 & X3 (3.6)
                X1 = Xalpha[d] - A1 * Dalpha
                X2 = Xbeta[d] - A2 * Dbeta
                X3 = Xdelta[d] - A3 * Ddelta

                # Update wolf (3.7)
                X[i, d] = (X1 + X2 + X3) / 3

                # Boundary
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)

        # Update archive
        for i in range(N):
            archive.append((X[i, :], fit[i, 0]))
            if len(archive) > archive_size:
                # Remove the most crowded solution
                distances = crowding_distance([x for x, _ in archive], [f for _, f in archive])
                max_distance_index = np.argmax(distances)
                del archive[max_distance_index]

        # Non-dominated sort to update the Pareto front
        pareto_front = non_dominated_sort([x for x, _ in archive], [f for _, f in archive])

        curve[0, t] = fit.min()
        t += 1

    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.arange(dim)
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return gwo_data