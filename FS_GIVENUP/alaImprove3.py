# 增加动态多狼机制3-1

import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
# 适应度函数（伪代码示例）
from scipy.special import gamma  

# def Fun(xtrain, ytrain, Xbin, opts):
#     # 计算特征选择的分类准确度，假设分类器函数为 staticClassifier
#     # 例如：准确度 = staticClassifier(xtrain[:, Xbin], ytrain)
#     # 返回负的准确度作为适应度（因为 GWO 是最小化问题）
#     # return -1 * 准确度
#     # 假设这里直接返回随机值作为示例
#     return np.random.rand()

def initialization(N, dim, ub, lb):
    X = np.zeros((N, dim), dtype='float')
    for i in range(N):
        X[i, :] = lb + np.random.rand(dim) * (ub - lb)
    return X
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin
def boundary_check(X, lb, ub):
    X = np.clip(X, lb, ub)
    return X
def update_position_high_energy(X, Xnew, Xalpha, F, RB, Iter, dim, N, i):
    if np.random.rand() < 0.3:
        r1 = 2 * np.random.rand(dim) - 1
        Xnew[i, :] += Xalpha[0, :] + F * RB[i, :] * (r1 * (Xalpha[0, :] - X[i, :]) +
                                                    (1 - r1) * (X[i, :] - X[np.random.randint(N), :]))
    else:
        r2 = np.random.rand() * (1 + np.sin(0.5 * Iter))
        Xnew[i, :] += X[i, :] + F * r2 * (Xalpha[0, :] - X[np.random.randint(N), :])
    return Xnew[i, :]
def update_position_low_energy(X, Xnew, Xalpha, F, Iter, Max_iter, dim, Levy, i):
    if np.random.rand() < 0.5:
        radius = np.linalg.norm(Xalpha[0, :] - X[i, :])
        r3 = np.random.rand()
        spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
        Xnew[i, :] += Xalpha[0, :] + F * X[i, :] * spiral * np.random.rand()
    else:
        G = 2 * (np.sign(np.random.rand() - 0.5)) * (1 - Iter / Max_iter)
        Xnew[i, :] += Xalpha[0, :] + F * G * Levy(dim) * (Xalpha[0, :] - X[i, :])
    return Xnew[i, :]
def Levy(dim):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta)/2) * beta * 2 ** ((beta - 1)/2))) ** (1/beta)
    u = np.random.randn(1, dim) * sigma
    v = np.random.randn(1, dim)
    step = u / np.abs(v) ** (1/beta)
    return step.flatten()

def jfs(xtrain, ytrain, opts):
    # Parameters
    ub = opts['ub']  if ('runcec' in opts and opts['runcec'] == True) else 1
    lb = opts['lb']  if ('runcec' in opts and opts['runcec'] == True) else 0
    thres = 0.5

    N = opts['N']
    Max_iter = opts['T']
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)

    # Initialize position
    X = initialization(N, dim, ub, lb)
    Xnew = np.zeros((N, dim), dtype='float')
    Xalpha = np.zeros((1, dim), dtype='float')
    Falpha = float('inf')
    fitness = np.zeros((N, 1), dtype='float')
    Convergence_curve = np.zeros((1, Max_iter), dtype='float')

    # init for gwo
    Xbeta  = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    # Initial fitness evaluation
    for i in range(N):
        Xbin = np.where(X[i, :] > thres, 1, 0)
        fitness[i, 0] = Fun(xtrain, ytrain, Xbin, opts)
        if fitness[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fitness[i, 0]
            
        if fitness[i,0] < Fbeta and fitness[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fitness[i,0]
            
        if fitness[i,0] < Fdelta and fitness[i,0] > Fbeta and fitness[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fitness[i,0]
    Iter = 1
    while Iter <= Max_iter:
        # Coefficient decreases linearly from 2 to 0 
        a = 2 - Iter * (2 / Max_iter) 
        # Vectorized implementation
        # Generate random matrices
        C = 2 * rand(N, dim, 3)  # C1,C2,C3
        A = 2 * a * rand(N, dim, 3) - a  # A1,A2,A3
        
        # Compute distances
        D = np.abs(C * np.stack([Xalpha, Xbeta, Xdelta], axis=2) - X[:,:,None])
        
        # Compute new positions
        gwoX_new = (np.stack([Xalpha, Xbeta, Xdelta], axis=2) - A * D)
        gwoX = np.mean(gwoX_new, axis=2)
        
        # Apply boundary constraints
        gwoX = np.clip(X, lb, ub)
        
        RB = np.random.randn(N, dim)
        F = (-1) ** (np.random.randint(2))
        theta = 2 * np.arctan(1 - Iter / Max_iter)
        # E_rand = 2 * np.log(1 / np.random.rand(N, dim)) * theta
        if Iter < Max_iter//2:
            for i in range(N):
                E_rand = 2*np.log(1/rand())*theta
                if E_rand > 1:
                    Xnew[i, :] = update_position_high_energy(X, Xnew, Xalpha, F, RB, Iter, dim, N, i)
                else:
                    Xnew[i, :] = update_position_low_energy(X, Xnew, Xalpha, F, Iter, Max_iter, dim, Levy, i)
                if E_rand > 1:
                    Xnew[i, :] = update_position_high_energy(X, Xnew, Xbeta, F, RB, Iter, dim, N, i)
                else:
                    Xnew[i,:]  = update_position_low_energy(X, Xnew, Xbeta, F, Iter, Max_iter, dim, Levy, i)
                if E_rand > 1:
                    Xnew[i, :] = update_position_high_energy(X, Xnew, Xdelta, F, RB, Iter, dim, N, i)
                else:
                    Xnew[i,:]  = update_position_low_energy(X, Xnew, Xdelta, F, Iter, Max_iter, dim, Levy, i)
            Xnew = Xnew / 3
        else:
            for i in range(N):
                E_rand = 2*np.log(1/rand())*theta
                if E_rand > 1:
                    Xnew[i, :] = update_position_high_energy(X, Xnew, Xalpha, F, RB, Iter, dim, N, i)
                else:
                    Xnew[i, :] = update_position_low_energy(X, Xnew, Xalpha, F, Iter, Max_iter, dim, Levy, i)
       
        # Boundary check and evaluation
        Xnew = boundary_check(Xnew, lb, ub)
        # greedy stragedy
        for i in range(N):
            Xbin = np.where(Xnew[i, :] > thres, 1, 0)
            newPopfit = Fun(xtrain, ytrain, Xbin, opts)
            if newPopfit < fitness[i, 0]:
                X[i, :] = Xnew[i, :]
                fitness[i, 0] = newPopfit
            if fitness[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fitness[i, 0]
            if fitness[i,0] < Fbeta and fitness[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta       = fitness[i,0]
                
            if fitness[i,0] < Fdelta and fitness[i,0] > Fbeta and fitness[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fitness[i,0]
        # Record convergence
        Convergence_curve[0, Iter - 1] = Falpha
        Iter += 1

    # Best feature subset
    Gbin       = binary_conversion(Xalpha, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    # XalphaBin = np.where(Xalpha > thres, 1, 0)
    # sel_index = np.where(XalphaBin == 1).flatten()
    num_feat = len(sel_index)
    gwo_data = {'sf': sel_index, 'c': Convergence_curve, 'nf': num_feat}

    return gwo_data
