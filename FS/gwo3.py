#[2014]-"Grey wolf optimizer"
#[Modified with I-GWO]
# https://blog.csdn.net/yuchunyu12/article/details/137788275
# https://www.sciencedirect.com/science/article/pii/S0957417420307107
# An improved grey wolf optimizer for solving engineering problems
import numpy as np
from numpy.random import rand
from FS.functionHO import Fun


def init_position(lb, ub, N, dim):
    X = np.zeros((N, dim), dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()        
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros((N, dim), dtype='int')
    for i in range(N):
        for d in range(dim):
            Xbin[i, d] = 1 if X[i, d] > thres else 0
    return Xbin


def boundary(x, lb, ub):
    return np.clip(x, lb, ub)
    

def jfs(xtrain, ytrain, opts):
    # Parameters
    ub = opts['ub']  if ('runcec' in opts and opts['runcec'] == True) else 1
    lb = opts['lb']  if ('runcec' in opts and opts['runcec'] == True) else 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones((1, dim), dtype='float')
        lb = lb * np.ones((1, dim), dtype='float')
        
    # Initialize position 
    X = init_position(lb, ub, N, dim)
    
    # Binary conversion and initial fitness
    Xbin = binary_conversion(X, thres, N, dim)
    fit = np.zeros((N, 1), dtype='float')
    Xalpha = np.zeros((1, dim), dtype='float')
    Xbeta  = np.zeros((1, dim), dtype='float')
    Xdelta = np.zeros((1, dim), dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
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
    
    curve = np.zeros((1, max_iter), dtype='float')
    t = 0
    curve[0, t] = Falpha
    t += 1
    
    while t < max_iter:
        a = 2 - t * (2 / max_iter)
        X_old = X.copy()
        fit_old = fit.copy()
        Xalpha_old = Xalpha.copy()
        Xbeta_old = Xbeta.copy()
        Xdelta_old = Xdelta.copy()
        
        X_new = np.zeros_like(X)
        fit_new = np.zeros_like(fit)
        
        for i in range(N):
            # Generate GWO candidate
            temp_X_gwo = np.zeros(dim)
            for d in range(dim):
                C1, C2, C3 = 2 * rand(), 2 * rand(), 2 * rand()
                Dalpha = abs(C1 * Xalpha_old[0, d] - X_old[i, d])
                Dbeta = abs(C2 * Xbeta_old[0, d] - X_old[i, d])
                Ddelta = abs(C3 * Xdelta_old[0, d] - X_old[i, d])
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                X1 = Xalpha_old[0, d] - A1 * Dalpha
                X2 = Xbeta_old[0, d] - A2 * Dbeta
                X3 = Xdelta_old[0, d] - A3 * Ddelta
                temp_X_gwo[d] = boundary((X1 + X2 + X3) / 3, lb[0, d], ub[0, d])
            
            # Calculate radius R_i
            R_i = np.linalg.norm(X_old[i] - temp_X_gwo)
            
            # Build neighborhood N_i
            N_i = []
            for j in range(N):
                if np.linalg.norm(X_old[j] - X_old[i]) <= R_i:
                    N_i.append(X_old[j])
            if not N_i:
                N_i.append(X_old[i])  # Handle empty neighborhood
            
            # Random select X_n and X_r
            X_n = N_i[np.random.randint(0, len(N_i))]
            X_r = X_old[np.random.randint(0, N)]
            
            # Generate DLH candidate
            temp_X_dlh = np.zeros(dim)
            for d in range(dim):
                delta = X_n[d] - X_r[d]
                temp_X_dlh[d] = X_old[i, d] + rand() * delta
                temp_X_dlh[d] = boundary(temp_X_dlh[d], lb[0, d], ub[0, d])
            
            # Evaluate candidates
            Xbin_gwo = binary_conversion(temp_X_gwo.reshape(1, -1), thres, 1, dim)
            fit_gwo = Fun(xtrain, ytrain, Xbin_gwo[0], opts,np.clip(temp_X_gwo,lb,ub))
            Xbin_dlh = binary_conversion(temp_X_dlh.reshape(1, -1), thres, 1, dim)
            fit_dlh = Fun(xtrain, ytrain, Xbin_dlh[0], opts,np.clip(temp_X_dlh,lb,ub))
            
            # Select better candidate and compare with current
            if fit_gwo < fit_dlh:
                X_candidate, fit_candidate = temp_X_gwo, fit_gwo
            else:
                X_candidate, fit_candidate = temp_X_dlh, fit_dlh
            
            # Update position if candidate is better
            if fit_candidate < fit_old[i, 0]:
                X_new[i] = X_candidate
                fit_new[i, 0] = fit_candidate
            else:
                X_new[i] = X_old[i]
                fit_new[i, 0] = fit_old[i, 0]
        
        # Update population
        X = X_new.copy()
        fit = fit_new.copy()
        
        # Update alpha, beta, delta
        Falpha = Fbeta = Fdelta = float('inf')
        Xalpha.fill(0); Xbeta.fill(0); Xdelta.fill(0)
        
        for i in range(N):
            if fit[i, 0] < Falpha:
                Xdelta = Xbeta.copy()
                Fdelta = Fbeta
                Xbeta = Xalpha.copy()
                Fbeta = Falpha
                Xalpha[0] = X[i]
                Falpha = fit[i, 0]
            elif fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
                Xdelta = Xbeta.copy()
                Fdelta = Fbeta
                Xbeta[0] = X[i]
                Fbeta = fit[i, 0]
            elif fit[i, 0] < Fdelta and fit[i, 0] > Fbeta:
                Xdelta[0] = X[i]
                Fdelta = fit[i, 0]
        
        curve[0, t] = Falpha
        t += 1
    
    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim).flatten()
    sel_index = np.where(Gbin == 1)[0]
    num_feat = len(sel_index)
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    return gwo_data