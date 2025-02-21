import numpy as np
from numpy.random import rand, randn
from FS.functionHO import Fun
# https://blog.csdn.net/u011835903/article/details/125324187
# http://qikan.cqvip.com/Qikan/Article/Detail?id=7001148503
# 一种改进的灰狼优化算法

# 函数分类：
# f1​ 到 f8​ 是单峰函数
# f9​ 到 f18​ 是多峰函数
# 函数维度：
# 30维
# 100维
# 500维
# 1000维
# EGWO算法参数：
# 种群规模 N=30
# 最大迭代次数 tmax​=500
# 距离控制参数 a 从2线性减少到0
# 个体记忆系数 b1​=0.1
# 群体交流系数 b2​=0.9

# 没有的信息：西塔的取值。
# 这些信息涵盖了原文的主要内容。
def init_position_chaotic(lb, ub, N, dim, k = 5):
    X = np.zeros([N, dim], dtype='float')
    xMin = np.ones([N, dim], dtype='float') * float('inf')
    xMax = np.ones([N, dim], dtype='float') * -float('inf')
    for i in range(N):
        for d in range(dim):
            phi = rand()  # Initial random value in [0, 1] for phi
            xk  = rand()  # Initial random value in [0, 1] for xk
            for j in range(k):                
                if 0 < xk < phi:
                    xk = xk / phi
                elif phi <= xk < 1:
                    xk = (1 - xk) / (1 - phi)
                if  xk < xMin[i,d]:
                    xMin[i,d] = xk
                if  xk > xMax[i,d]:
                    xMax[i,d] = xk    
            X[i, d] = xMin[i,d] + (xMax[i,d] - xMin[i,d]) * xk
    return X
def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
def update_position_improved(X, Xalpha, Xbeta, Xdelta, P_ibest, Xj, a, b1, b2, r3, r4):
    X1 = (Xalpha + Xbeta + Xdelta) / 3
    X2 = X1 + b1 * r3 * (P_ibest - X)
    X3 = X2 + b2 * r4 * (Xj - X)
    return X3
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin
def jfs(xtrain, ytrain, opts):
    # parmeters
    b1 = 0.1
    b2 = 0.9
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

    # Initialize position using chaotic map
    X = init_position_chaotic(lb, ub, N, dim)
    # initialize history best position
    XhistoryBest = X.copy()
    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    # initialize history best fitness
    fitHistoryBest = np.ones([N, 1], dtype='float') * float('inf')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')

    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]
        if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]
        if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]
    # copy the best fitness
    fitHistoryBest = fit.copy()
    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    t += 1

    while t < max_iter:
        # Random adjustment strategy for control parameter a
        a_initial = 2
        a_final = 0
        a = a_initial - (a_initial - a_final) * rand() + 0.1 * randn()

        # Update positions
        for i in range(N):
            for d in range(dim):
                # Calculate parameters A and C
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()

                # Calculate Dalpha, Dbeta, Ddelta
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])

                # Calculate X1, X2, X3
                X1 = Xalpha[0, d] - A1 * Dalpha
                X2 = Xbeta[0, d] - A2 * Dbeta
                X3 = Xdelta[0, d] - A3 * Ddelta

                # Modified position update equation
                r3 = rand()
                r4 = rand()
                Xj = X[np.random.randint(0, N), d]  # Randomly select another wolf
                P_ibest = XhistoryBest[i,d]  # Best position of the wolf，没有实现记忆功能，0.1，b1,b2是常数。
                X[i, d] = update_position_improved(X[i, d], Xalpha[0, d], Xbeta[0, d], Xdelta[0, d], P_ibest, Xj, a, b1, b2, r3, r4)

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
            if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]
            if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]
        # update the best history
        for i in range(N):
            if fit[i, 0] < fitHistoryBest[i, 0]:
                fitHistoryBest[i, 0] = fit[i, 0]
                XhistoryBest[i, :] = X[i, :]
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