# https://d.wanfangdata.com.cn/thesis/ChhUaGVzaXNOZXdTMjAyNDA5MjAxNTE3MjUSCUQwMjM4MDEyORoINzhhcWtvNG4%3D
# 很可能错！
# DE-GWO
import numpy as np  
from numpy.random import rand  
from FS.functionHO import Fun  
  
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
  
def jfs(xtrain, ytrain, opts):  
    # Parameters  
    ub = opts['ub']  if ('runcec' in opts and opts['runcec'] == True) else 1  
    lb = opts['lb']  if ('runcec' in opts and opts['runcec'] == True) else 0  
    thres = 0.5  
  
    N = opts['N']  
    max_iter = opts['T']  
  
    # Dimension  
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)  
    if np.size(lb) == 1:  
        ub = ub * np.ones([1, dim], dtype='float')  
        lb = lb * np.ones([1, dim], dtype='float')  
  
    # Initialize position  
    X = init_position(lb, ub, N, dim)  
  
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
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts,np.clip(X[i,:],lb,ub))  
        if fit[i, 0] < Falpha:  
            Xalpha[0, :] = X[i, :]  
            Falpha = fit[i, 0]  
        elif fit[i, 0] < Fbeta and fit[i, 0] > Falpha:  
            Xbeta[0, :] = X[i, :]  
            Fbeta = fit[i, 0]  
        elif fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:  
            Xdelta[0, :] = X[i, :]  
            Fdelta = fit[i, 0]  
  
    # Pre  
    curve = np.zeros([1, max_iter], dtype='float')  
    t = 0  
  
    curve[0, t] = Falpha.copy()  
    t += 1  
  
    while t < max_iter:  
        # Update parameter a  
        a = 2 - t * (2 / max_iter)  
  
        for i in range(N):  
            X1 = np.zeros([1, dim], dtype='float')
            X2 = np.zeros([1, dim], dtype='float')
            X3 = np.zeros([1, dim], dtype='float')
            for d in range(dim):
                # Parameter C (3.4)
                C1     = 2 * rand()
                C2     = 2 * rand()
                C3     = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A (3.3)
                A1     = 2 * a * rand() - a
                A2     = 2 * a * rand() - a
                A3     = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6) 
                X1[0,d]     = Xalpha[0,d] - A1 * Dalpha
                X2[0,d]     = Xbeta[0,d] - A2 * Dbeta
                X3[0,d]     = Xdelta[0,d] - A3 * Ddelta
  
            # Compute mutation vector  
            F = 0.5  
            mutation_vector = X3 + F * (X1 - X2)  
  
            # Update position based on |A|  
            A = 2 * a * rand() - a  
            if abs(A) >= 1:  
                X[i, :] = mutation_vector  
            else:  
                X[i, :] = X1 + F * (X2 - X3)  
  
            # Apply boundary constraints  
            for d in range(dim):  
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  
  
        # Binary conversion  
        Xbin = binary_conversion(X, thres, N, dim)  
  
        # Fitness  
        for i in range(N):  
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts,np.clip(X[i,:],lb,ub))  
            if fit[i, 0] < Falpha:  
                Xalpha[0, :] = X[i, :]  
                Falpha = fit[i, 0]  
            elif fit[i, 0] < Fbeta and fit[i, 0] > Falpha:  
                Xbeta[0, :] = X[i, :]  
                Fbeta = fit[i, 0]  
            elif fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:  
                Xdelta[0, :] = X[i, :]  
                Fdelta = fit[i, 0]  
  
        curve[0, t] = Falpha.copy()  
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