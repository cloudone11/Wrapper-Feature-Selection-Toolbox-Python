import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
# https://blog.csdn.net/weixin_43821559/article/details/118394571
# https://jns.usst.edu.cn/shlgdxxbzk/article/abstract/20210110
# 求解全局优化问题的改进灰狼算法----刊登在小报
# eps = 1e-16  # Small constant to avoid division by zero

# 实验目的：验证提出的3种改进策略的有效性。
# 实验方法：选取了国际上通用的10种基准测试函数进行仿真实验。
# 实验环境：
# CPU：Intel(R) CPU 3550M，主频2.30 GHz
# 内存：4 GB
# 操作系统：Windows 10 64位
# 编程语言：Python 3.7
# 实验设计：
# 进行了3组对比实验。
# 所有算法的种群数 N 均设定为30。
# 总迭代次数设定为500。
# 性能评估：
# 为了排除随机性的影响，所有实验独立运行30次。
# 取30次的平均值和标准差作为算法性能的度量标准。

# 发现Xalpha取值过低，改变变异策略为*3
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
    eps = 0  # Small constant to avoid division by zero

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
        if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]
        if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    t += 1

    while t < max_iter:
        # Exponential convergence factor
        a = 2 * np.exp(-t / max_iter)

        # Sort the population based on fitness
        fit_sorted_indices = np.argsort(fit.flatten())
        X_sorted = X[fit_sorted_indices]
        Xalpha = X_sorted[0, :].reshape(1,-1)
        Xbeta = X_sorted[1, :].reshape(1,-1)
        Xdelta = X_sorted[2, :].reshape(1,-1)

        

        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1     = 2 * rand()
                C2     = 2 * rand()
                C3     = 2 * rand()
                
                # Parameter A (3.3)
                A1     = 2 * a * rand() - a
                A2     = 2 * a * rand() - a
                A3     = 2 * a * rand() - a
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Compute X1, X2 & X3 (3.6) 
                X1     = Xalpha[0,d] - A1 * Dalpha
                X2     = Xbeta[0,d] - A2 * Dbeta
                X3     = Xdelta[0,d] - A3 * Ddelta
                # Calculate dynamic weights
                # dist_alpha = abs(Xalpha[0,d] - X[i,d])
                
                # dist_beta =  abs(Xbeta[0,d] - X[i,d])
                # dist_delta = abs(Xdelta[0,d] - X[i,d])
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                dist_alpha = np.linalg.norm(X1)
                dist_beta  = np.linalg.norm(X2)
                dist_delta = np.linalg.norm(X3)
                W1 = dist_alpha / (dist_alpha + dist_beta + dist_delta + eps)
                W2 = dist_beta / (dist_alpha + dist_beta + dist_delta + eps)
                W3 = dist_delta / (dist_alpha + dist_beta + dist_delta + eps)
                # Update wolf (3.7)
                # New position
                X_new = (W1 * X1 + W2 * X2 + W3 * X3)*(1 - t / max_iter)  + Xalpha[0, d] * (t / max_iter)         
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts,np.clip(X[i,:],lb,ub))
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]
            if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]
            if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]

        curve[0, t] = Falpha.copy()
        t += 1
    # print('xalpha: ', Xalpha)
    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return gwo_data