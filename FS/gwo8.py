# https://d.wanfangdata.com.cn/thesis/ChhUaGVzaXNOZXdTMjAyNDA5MjAxNTE3MjUSCUQwMjM4MDEyORoINzhhcWtvNG4%3D
# 灰狼优化算法的改进研究
# AL-GWO
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

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Xgamma = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')
    Fgamma = float('inf')

    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]
        elif fit[i, 0] < Fbeta:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]
        elif fit[i, 0] < Fdelta:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]
        elif fit[i, 0] < Fgamma:
            Xgamma[0, :] = X[i, :]
            Fgamma = fit[i, 0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    t += 1

    while t < max_iter:
        # Update parameter a
        a = 2 - t * (2 / max_iter)

        for i in range(N):
            for d in range(dim):
                # Parameter C
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                C4 = 2 * rand()

                # Compute Dalpha, Dbeta, Ddelta, Dgamma
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                Dgamma = abs(C4 * Xgamma[0, d] - X[i, d])

                # Parameter A
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                A4 = 2 * a * rand() - a

                # Compute X1, X2, X3, X4
                X1 = Xalpha[0, d] - A1 * Dalpha
                X2 = Xbeta[0, d] - A2 * Dbeta
                X3 = Xdelta[0, d] - A3 * Ddelta
                X4 = Xgamma[0, d] - A4 * Dgamma

                # Update position based on |A|
                if abs(a) >= 1:
                    X[i, d] = (X1 + X2 + X3 + X4) / 4
                else:
                    X[i, d] = (X1 + X2) / 2

                # Boundary
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]
            elif fit[i, 0] < Fbeta:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]
            elif fit[i, 0] < Fdelta:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]
            elif fit[i, 0] < Fgamma:
                Xgamma[0, :] = X[i, :]
                Fgamma = fit[i, 0]

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