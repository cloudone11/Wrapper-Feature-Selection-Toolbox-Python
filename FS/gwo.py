#[2014]-"Grey wolf optimizer"

import numpy as np
from numpy.random import rand
from FS.functionHO import Fun


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
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X      = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin   = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit    = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta  = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    # fit    = opts['alpha']*staticClassifier.muti_classifier([Xbin]).reshape(-1,1) + (1-opts['alpha'])*np.sum(Xbin,axis=1).reshape(-1,1)/dim
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha      = fit[i,0]
            
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fit[i,0]
            
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = Falpha.copy()
    # print("Iteration:", t + 1)
    # print("Best (GWO):", curve[0,t])
    t += 1
    
    while t < max_iter:  
      	# Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter) 
        
        # Vectorized implementation
        # Generate random matrices
        C = 2 * rand(N, dim, 3)  # C1,C2,C3
        A = 2 * a * rand(N, dim, 3) - a  # A1,A2,A3
        
        # Compute distances
        D = np.abs(C * np.stack([Xalpha, Xbeta, Xdelta], axis=2) - X[:,:,None])
        
        # Compute new positions
        X_new = (np.stack([Xalpha, Xbeta, Xdelta], axis=2) - A * D)
        X = np.mean(X_new, axis=2)
        
        # Apply boundary constraints
        X = np.clip(X, lb, ub)
        
        # Binary conversion
        Xbin  = binary_conversion(X, thres, N, dim)
        
        # Fitness
        # 注意：此处有优化空间，可增加并行计算
        
        # fit    = opts['alpha']*staticClassifier.muti_classifier([Xbin]).reshape(-1,1) + (1-opts['alpha'])*np.sum(Xbin,axis=1).reshape(-1,1)/dim
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha      = fit[i,0]
                
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta       = fit[i,0]
                
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fit[i,0]
        
        curve[0,t] = Falpha.copy()
        # print("Iteration:", t + 1)
        # print("Best (GWO):", curve[0,t])
        t += 1
    
                
    # Best feature subset
    Gbin       = binary_conversion(Xalpha, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return gwo_data
