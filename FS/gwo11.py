import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
# https://blog.csdn.net/u011835903/article/details/125511418
# https://qikan.cqvip.com/Qikan/Article/Detail?id=671506068

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
    ub = 1
    lb = 0
    thres = 0.5
    
    N = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position and velocity
    X = init_position(lb, ub, N, dim)
    V = np.zeros([N, dim], dtype='float')
    
    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')
    
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
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
    # add history best storage for X and fit
    X_hist = np.zeros([N, dim], dtype='float')
    fit_hist = np.zeros([N, 1], dtype='float')
    # set the parameters of the algorithm
    w = 0.8
    while t < max_iter:
        # store the location of current X
        X_old = X.copy()
        # Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter) 
        # Calculate average fitness
        favg = np.mean(fit)
        
        # Update positions and velocities
        for i in range(N):
            # Adaptive adjustment strategy
            if fit[i,0] > favg:
                # Original strategy
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                Dalpha = abs(C1 * Xalpha[0,:] - X[i,:])
                Dbeta = abs(C2 * Xbeta[0,:] - X[i,:])
                Ddelta = abs(C3 * Xdelta[0,:] - X[i,:])
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                X1 = Xalpha[0,:] - A1 * Dalpha
                X2 = Xbeta[0,:] - A2 * Dbeta
                X3 = Xdelta[0,:] - A3 * Ddelta
                X[i,:] = (X1 + X2 + X3) / 3
            else:
                # Original strategy
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                Dalpha = abs(C1 * Xalpha[0,:] - X[i,:])
                Dbeta = abs(C2 * Xbeta[0,:] - X[i,:])
                Ddelta = abs(C3 * Xdelta[0,:] - X[i,:])
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                X1 = Xalpha[0,:] - A1 * Dalpha
                X2 = Xbeta[0,:] - A2 * Dbeta
                X3 = Xdelta[0,:] - A3 * Ddelta
                # Adaptive adjustment using fitness proportion
                f_inv = 1 / (Falpha + Fbeta + Fdelta)
                X1 = (1 / Falpha) * X1 + (1 / Fbeta) * X2 + (1 / Fdelta) * X3
                X[i,:] = X1 / f_inv
            
            # Boundary
            for d in range(dim):
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
            
            # Escape local optima strategy
            r = rand() +1  # Random number in [-2, 2]
            # 以50%的概率取负值
            if rand() < 0.5:
                r = -r
            X_prime = Xalpha[0,:] + np.abs(Xalpha[0,:] - X[i,:]) * r
            X_prime.reshape(-1, dim)
            
            X_prime = binary_conversion(X_prime, thres, 1, dim)
            fit_prime = Fun(xtrain, ytrain, X_prime[0,:], opts)
            if fit_prime < fit[i,0]:
                X[i,:] = X_prime
                fit[i,0] = fit_prime
            
            # Optimal learning search equation
            V[i,:] = w * V[i,:] + (rand()*2 -1) * (X_hist[i,:] - X_old[i,:]) + rand() * 1.5 * (Xalpha[0,:] - X[i,:])
            X[i,:] = X[i,:] + V[i,:]
            for d in range(dim):
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha = fit[i,0]
                
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:] = X[i,:]
                Fbeta = fit[i,0]
                
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta = fit[i,0]
        # modify the history best fitness
        for i in range(N):
            if fit[i,0] < fit_hist[i,0]:
                X_hist[i,:] = X[i,:]
                fit_hist[i,0] = fit[i,0]
        curve[0,t] = Falpha.copy()
        t += 1
    
    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim) 
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))    
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return gwo_data