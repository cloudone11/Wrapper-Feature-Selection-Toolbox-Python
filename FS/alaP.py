import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
from scipy.special import gamma

def initialization(N, dim, ub, lb):
    X = np.zeros((N, dim), dtype='float')
    for i in range(N):
        X[i, :] = lb + np.random.rand(dim) * (ub - lb)
    return X

def binary_conversion(X, thres, N, dim):
    return (X > thres).astype(int)

def boundary_check(X, lb, ub):
    return np.clip(X, lb, ub)

def Levy(dim):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta)/2) * beta * 2 ** ((beta - 1)/2))) ** (1/beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / np.abs(v) ** (1/beta)

def update_subpopulation_ala(X, fitness, Position, Score, Iter, Max_iter, N, dim, thres, opts, ub, lb, xtrain, ytrain, theta):
    Xnew = np.zeros((N, dim))
    RB = np.random.randn(N, dim)
    F = (-1) ** np.random.randint(2)
    
    for i in range(N):
        E_rand = 2 * np.log(1 / np.random.rand()) * theta
        if E_rand > 1:
            if np.random.rand() < 0.3:
                r1 = 2 * np.random.rand(dim) - 1
                other = np.random.randint(N)
                term1 = r1 * (Position[0] - X[i])
                term2 = (1 - r1) * (X[i] - X[other])
                Xnew[i] = Position[0] + F * RB[i] * (term1 + term2)
            else:
                r2 = np.random.rand() * (1 + np.sin(0.5 * Iter))
                other = np.random.randint(N)
                Xnew[i] = X[i] + F * r2 * (Position[0] - X[other])
        else:
            if np.random.rand() < 0.5:
                radius = np.linalg.norm(Position[0] - X[i])
                r3 = np.random.rand()
                spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
                Xnew[i] = Position[0] + F * X[i] * spiral * np.random.rand()
            else:
                G = 2 * (np.sign(np.random.rand() - 0.5)) * (1 - Iter / Max_iter)
                levy = Levy(dim)
                Xnew[i] = Position[0] + F * G * levy * (Position[0] - X[i])
    
    Xnew = boundary_check(Xnew, lb, ub)
    
    for i in range(N):
        Xbin = (Xnew[i] > thres).astype(int)
        new_fit = Fun(xtrain, ytrain, Xbin, opts)
        if new_fit < fitness[i]:
            X[i] = Xnew[i]
            fitness[i] = new_fit
            if new_fit < Score:
                Position = X[i,:]
                Score = new_fit
    
    return X, fitness, Position, Score

def jfs(xtrain, ytrain, opts):
    ub, lb = 1, 0
    thres = 0.5
    N = opts['N']
    Max_iter = opts['T']
    dim = xtrain.shape[1]
    N = N // 2
    # Initialize two subpopulations
    X1 = initialization(N, dim, ub, lb)
    X2 = initialization(N, dim, ub, lb)
    
    Position1, Score1 = np.zeros(dim), float('inf')
    fitness1 = np.full(N, float('inf'))
    Position2, Score2 = np.zeros(dim), float('inf')
    fitness2 = np.full(N, float('inf'))

    # Initial evaluation
    for i in range(N):
        Xbin = (X1[i] > thres).astype(int)
        fit = Fun(xtrain, ytrain, Xbin, opts)
        fitness1[i] = fit
        if fit < Score1:
            Position1 = X1[i]
            Score1 = fit

        Xbin = (X2[i] > thres).astype(int)
        fit = Fun(xtrain, ytrain, Xbin, opts)
        fitness2[i] = fit
        if fit < Score2:
            Position2 = X2[i]
            Score2 = fit

    Convergence_curve = np.zeros((1, Max_iter), dtype='float')
    
    for Iter in range(Max_iter):
        theta = 2 * np.arctan(1 - (Iter + 1) / Max_iter)
        
        # Update subpopulations
        X1, fitness1, Position1, Score1 = update_subpopulation_ala(
            X1, fitness1, Position1, Score1, Iter+1, Max_iter, N, dim, 
            thres, opts, ub, lb, xtrain, ytrain, theta)
        
        X2, fitness2, Position2, Score2 = update_subpopulation_ala(
            X2, fitness2, Position2, Score2, Iter+1, Max_iter, N, dim, 
            thres, opts, ub, lb, xtrain, ytrain, theta)
        
        # Migration every 5 generations
        if (Iter + 1) % 5 == 0 and Iter != 0:
            k = max(1, int(0.2 * N))
            
            # Subpopulation 1
            sorted_idx1 = np.argsort(fitness1)
            top1 = sorted_idx1[:k]
            worst1 = sorted_idx1[-k:]
            
            # Subpopulation 2
            sorted_idx2 = np.argsort(fitness2)
            top2 = sorted_idx2[:k]
            worst2 = sorted_idx2[-k:]
            
            # Exchange
            X2[worst2] = X1[top1].copy()
            fitness2[worst2] = fitness1[top1].copy()
            
            X1[worst1] = X2[top2].copy()
            fitness1[worst1] = fitness2[top2].copy()
            
            # Update best for each subpopulation
            idx1 = np.argmin(fitness1)
            if fitness1[idx1] < Score1:
                Position1 = X1[idx1]
                Score1 = fitness1[idx1]
            
            idx2 = np.argmin(fitness2)
            if fitness2[idx2] < Score2:
                Position2 = X2[idx2]
                Score2 = fitness2[idx2]
        
        # Track global best
        Convergence_curve[0,Iter] = min(Score1, Score2)
    
    # Final best solution
    best_Position = Position1 if Score1 < Score2 else Position2
    Gbin = (best_Position > thres).astype(int)
    sel_index = np.where(Gbin == 1)[0]
    
    return {'sf': sel_index, 'c': Convergence_curve, 'nf': len(sel_index)}