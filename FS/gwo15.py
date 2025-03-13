# https://blog.csdn.net/Logic_9527/article/details/142392985
# https://www.sciencedirect.com/science/article/pii/S0950705123000473
# 基于记忆、进化算子和局部搜索的改进灰狼优化算法
import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
# N or namely Pint should be 100
# Pmin should be 10
def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()
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

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def jfs(xtrain, ytrain, opts):
    # Parameters
    ub = opts['ub']  if ('runcec' in opts and opts['runcec'] == True) else 1
    lb = opts['lb']  if ('runcec' in opts and opts['runcec'] == True) else 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    
    # Initialize positions
    X = init_position(lb, ub, N, dim)
    Xbin = binary_conversion(X, thres, N, dim)
    
    # Initialize memory swarm
    memory_swarm = X.copy()
    memory_fitness = np.zeros([N, 1], dtype='float')
    
    # Fitness evaluation
    fit = np.zeros([N, 1], dtype='float')
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
        memory_fitness[i,0] = fit[i,0]
    
    # Initialize alpha, beta, and delta
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')
    
    for i in range(N):
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha = fit[i,0]
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:] = X[i,:]
            Fbeta = fit[i,0]
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    curve[0,t] = Falpha.copy()
    t += 1
    # set the parameter Pint to 100 and Pmin to 10
    Pint = N
    Pmin = 4
    while t < max_iter:
        # Update a
        a = 2 - t * (2 / max_iter)
        
        # Update positions
        for i in range(N):
            for d in range(dim):
                # Parameter C
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta, Ddelta
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d])
                Dbeta = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2, X3
                X1 = Xalpha[0,d] - A1 * Dalpha
                X2 = Xbeta[0,d] - A2 * Dbeta
                X3 = Xdelta[0,d] - A3 * Ddelta
                # Update wolf position
                X[i,d] = (X1 + X2 + X3) / 3
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness evaluation
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
        
        # Update alpha, beta, and delta
        for i in range(N):
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha = fit[i,0]
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:] = X[i,:]
                Fbeta = fit[i,0]
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta = fit[i,0]
        
        # Evolutionary operators
        F = 0 + (2-0)*((max_iter - t +1 ) / max_iter)
        for i in range(N):
            # randomly choose a j different from i
            j = np.random.randint(0, N)
            while j == i:
                j = np.random.randint(0, N)
            Xj = X[j,:]
            # Mutation
            V  = Xj + F * (Xalpha[0,:] - X[i,:])
            V = V.reshape(1,-1)
            # Crossover
            U = np.zeros([1, dim], dtype='float')
            for d in range(dim):
                if rand() < 0.6:
                    U[0,d] = V[0,d]
                else:
                    U[0,d] = X[i,d]
                U[0,d] = boundary(U[0,d], lb[0,d], ub[0,d])
            Ubin = binary_conversion(U, thres, 1, dim)
            fit_U = Fun(xtrain, ytrain, Ubin[0,:], opts,np.clip(U,lb,ub))
            if fit_U < fit[i,0]:
                X[i,:] = U[0,:]
                fit[i,0] = fit_U
        
        # Update memory swarm
        for i in range(N):
            if fit[i,0] < memory_fitness[i,0]:
                memory_swarm[i,:] = X[i,:]
                memory_fitness[i,0] = fit[i,0]
        
        # Local search
        # randomly choose N//2 wolves in the memory swarm to perform local search
        choosen_wolves = np.random.choice(N, N // 2, replace=False)
        for i in range(N // 2):
            # initialize the temporary wolves
            temp_wolves = np.zeros([1, dim], dtype='float')
            idx = choosen_wolves[i]
            # First, find ith wolf’s nearest neighbor among all wolves in the memory wolf, 
            # in terms of Euclidean distance of their positions within search space;
            nearst_neighbor_index = 0
            min_distance = float('inf')
            for j in range(N):
                if j == idx:
                    continue
                distance = np.linalg.norm(memory_swarm[idx,:] - memory_swarm[j,:])
                if distance < min_distance:
                    min_distance = distance
                    nearst_neighbor = j
            # if the cost function of the nearest neighbor is less than or equal to that of ith wolf
            if memory_fitness[nearst_neighbor,0] <= fit[idx,0]:
                temp_wolves[0,:] = memory_swarm[idx,:] + 2 * rand() * (memory_swarm[nearst_neighbor,:] - memory_swarm[idx,:])
            else:
                temp_wolves[0,:] = memory_swarm[idx,:] + 2 * rand() * (memory_swarm[nearst_neighbor,:] - memory_swarm[idx,:]) * -1 
            for d in range(dim):                               
                temp_wolves[0,d] = boundary(temp_wolves[0,d], lb[0,d], ub[0,d])
            temp_wolves_bin = binary_conversion(temp_wolves, thres, 1, dim)
            fit_local = Fun(xtrain, ytrain, temp_wolves_bin[0,:], opts,np.clip(temp_wolves[i,:],lb,ub))
            if fit_local < memory_fitness[idx,0]:
                memory_swarm[idx,:] = temp_wolves
                memory_fitness[idx,0] = fit_local
        
        # Linear population size reduction (LPSR)
        N = round((Pmin - Pint)/max_iter) * t + Pint
        # At the end of each iteration, the wolves
        # in the explorer swarm are sorted and the worst ranked wolves
        # are deleted. This procedure is repeated for the memory swarm
        # as well. 
        # sort the explorer swarm by fitness
        sorted_index = np.argsort(fit[:,0])
        X = X[sorted_index,:]
        fit = fit[sorted_index,:]
        # delete the worst ranked wolves
        X = X[:N,:]
        fit = fit[:N,:]
        # sort the memory swarm by fitness
        sorted_index = np.argsort(memory_fitness[:,0])
        memory_swarm = memory_swarm[sorted_index,:]
        memory_fitness = memory_fitness[sorted_index,:]
        # delete the worst ranked wolves    
        memory_swarm = memory_swarm[:N,:]
        memory_fitness = memory_fitness[:N,:]                
        # Update curve
        curve[0,t] = Falpha.copy()
        t += 1
    
    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return gwo_data