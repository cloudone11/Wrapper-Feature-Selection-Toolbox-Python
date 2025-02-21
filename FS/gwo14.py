# https://blog.csdn.net/u011835903/article/details/125530107
# 采用动态权重和概率扰动策略改进的灰狼优化算法
# http://qikan.cqvip.com/Qikan/Article/Detail?id=673903251
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
    ub = 1  
    lb = 0  
    thres = 0.5  
    N = opts['N']  
    max_iter = opts['T']  
    dim = np.size(xtrain, 1)  
    if np.size(lb) == 1:  
        ub = ub * np.ones([1, dim], dtype='float')  
        lb = lb * np.ones([1, dim], dtype='float')  
    X = init_position(lb, ub, N, dim)  
    Xbin = binary_conversion(X, thres, N, dim)  
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
    curve = np.zeros([1, max_iter], dtype='float')  
    t = 0  
    curve[0,t] = Falpha.copy()  
    t += 1  
    while t < max_iter:  
        a = 2 - t * (2 / max_iter)  
        for i in range(N):  
            for d in range(dim):  
                A = 2 * a * rand() - a  
                C = 2 * rand()  
                Dalpha = abs(C * Xalpha[0,d] - X[i,d])  
                Dbeta = abs(C * Xbeta[0,d] - X[i,d])  
                Ddelta = abs(C * Xdelta[0,d] - X[i,d])  
                X1 = Xalpha[0,d] - A * Dalpha  
                X2 = Xbeta[0,d] - A * Dbeta  
                X3 = Xdelta[0,d] - A * Ddelta  
                AC = abs(A * C)  
                S = 18 * AC**2 + 7 * AC  
                w1 = 1/3 
                w2 = 3 * AC / (1+6*AC) 
                w3 = (18 * AC**2 + 3 * AC) / (18*AC**2 + 18*AC +1)  
                X[i,d] = (w1 * X1 + w2 * X2 + w3 * X3 )/3 
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])  
        Xbin = binary_conversion(X, thres, N, dim)  
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
        # random interruption
        G = dim  
        P = ((G - 1) * np.exp((t - 1) / max_iter)) / (4 * G)  
        for i in range(N):  
            if rand() < P:  
                M = lb + rand() * (ub - lb)  
                M_fit = Fun(xtrain, ytrain, binary_conversion(M, thres, 1, dim)[0,:], opts)  
                if M_fit < fit[i,0]:  
                    X[i,:] = M  
                    fit[i,0] = M_fit  
        curve[0,t] = Falpha.copy()  
        t += 1  
    Gbin = binary_conversion(Xalpha, thres, 1, dim)  
    Gbin = Gbin.reshape(dim)  
    pos = np.asarray(range(0, dim))  
    sel_index = pos[Gbin == 1]  
    num_feat = len(sel_index)  
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}  
    return gwo_data  