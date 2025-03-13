# https://blog.csdn.net/weixin_43821559/article/details/116668114
# https://d.wanfangdata.com.cn/periodical/wdzxyjsj201905018
# 一种改进非线性收敛方式的灰狼优化算法
import numpy as np
from numpy.random import rand
from FS.functionHO import Fun

def logistic_mapping(N, dim, lb, ub, mu=4.0):
    X = np.zeros([N, dim], dtype='float')
    for d in range(dim):
        x = rand()
        for i in range(N):
            x = mu * x * (1 - x)
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * x
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
def cauchy_mutation_alpha(Xalpha,a,lb,ub):
    dim = Xalpha.shape[1]
    # randomly select a dimension to mutate
    rand_dim = np.random.randint(0, dim)
    Xalpha[0,rand_dim] = Xalpha[0,rand_dim] + np.random.standard_cauchy()
    # Boundary
    if Xalpha[0,rand_dim] < lb[0,rand_dim]:
        Xalpha[0,rand_dim] = lb[0,rand_dim]
    if Xalpha[0,rand_dim] > ub[0,rand_dim]:
        Xalpha[0,rand_dim] = ub[0,rand_dim]
    return Xalpha

def jfs(xtrain, ytrain, opts):
    ub = opts['ub']  if ('runcec' in opts and opts['runcec'] == True) else 1
    lb = opts['lb']  if ('runcec' in opts and opts['runcec'] == True) else 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']
    k = 2  # Nonlinear adjustment coefficient
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position using logistic mapping
    X = logistic_mapping(N, dim, lb, ub)
    
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
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha = fit[i,0]
        elif fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:] = X[i,:]
            Fbeta = fit[i,0]
        elif fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t = 0
    curve[0,t] = Falpha.copy()
    t += 1
    
    while t < max_iter:
        # Coefficient decreases nonlinearly
        a = 2 - 2 * ((np.exp(t / max_iter) - 1) / (np.exp(1) - 1)) ** k
        
        for i in range(N):
            for d in range(dim):
                # Parameter C
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d])
                Dbeta = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3
                X1 = Xalpha[0,d] - A1 * Dalpha
                X2 = Xbeta[0,d] - A2 * Dbeta
                X3 = Xdelta[0,d] - A3 * Ddelta
                # Update wolf
                X[i,d] = (X1 + X2 + X3) / 3
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        id4alpha = 0
        # Fitness evaluation
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha = fit[i,0]
                id4alpha = i
            elif fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:] = X[i,:]
                Fbeta = fit[i,0]
            elif fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta = fit[i,0]
        
        # Apply Cauchy mutation for alpha wolf
        Xalpha2 = cauchy_mutation_alpha(Xalpha,a,lb,ub)
        Xalpha2bin = binary_conversion(Xalpha2, thres, 1, dim)
        Xalpha2bin = Xalpha2bin.flatten()
        # print(Xalpha2bin)
        Xalpha2Fit = Fun(xtrain, ytrain, Xalpha2bin, opts,np.clip(Xalpha2,lb,ub))
        if Xalpha2Fit < Falpha:
            Xalpha = Xalpha2.copy()
            Falpha = Xalpha2Fit.copy()
            X[id4alpha,:] = Xalpha2.copy()
            fit[id4alpha,0] = Xalpha2Fit.copy()
                    
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