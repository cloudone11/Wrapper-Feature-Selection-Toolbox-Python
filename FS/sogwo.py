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
            Xbin[i, d] = 1 if X[i, d] > thres else 0
    return Xbin

def boundary(x, lb, ub):
    return max(lb, min(x, ub))
def corOppose(Positions, fitness, ub, lb, upper, lower, threshold):
    n = fitness.size  # 获取 fitness 的大小
    
    for i in range(3, n):  # MATLAB 从 4 开始（因 1-based 索引），Python 需调整为 0-based
        sum_d_squared = 0
        greater = []
        less = []
        x = 0  # 初始化计数器
        y = 0  # 初始化 greater 的计数器
        z = 0  # 初始化 less 的计数器
        
        for j in range(Positions.shape[1]):
            d = abs(Positions[0, j] - Positions[i, j])  # 计算差异
            if d < threshold:
                greater.append(j)  # 记录小于阈值的维度
                y += 1
            else:
                less.append(j)  # 记录大于等于阈值的维度
                z += 1
            sum_d_squared += d ** 2  # 累加平方和
        
        # 计算 src
        numerator = 6 * sum_d_squared
        denominator = n * (n ** 2 - 1)
        src = 1 - (numerator / denominator)
        
        # 判断条件
        if src <= 0:
            greater_size = len(greater)
            less_size = len(less)
            
            # 比较 greater 和 less 的大小
            if greater_size < less_size:
                # 可能的处理逻辑
                pass  # 根据需求补充
            else:
                for j in greater:
                    # 更新第 i 个个体的第 j 维
                    Positions[i, j] = upper[0, j] + lower[0, j] - Positions[i, j]
    
    return Positions
def jfs(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']
    dim = np.size(xtrain, 1)
    
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    
    # Initialize positions
    X = init_position(lb, ub, N, dim)
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')
    
    # Dynamic boundaries
    upper = ub.copy()
    lower = lb.copy()
    
    # Pre-allocate
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    
    # Initial fitness calculation
    Xbin = binary_conversion(X, thres, N, dim)
    fit = np.zeros([N, 1], dtype='float')
    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
        if fit[i, 0] < Falpha:
            Falpha, Xalpha = fit[i, 0], X[i, :].reshape(1,-1)
        if fit[i, 0] > Falpha and fit[i, 0] < Fbeta:
            Fbeta, Xbeta = fit[i, 0], X[i, :].reshape(1,-1)
        if fit[i, 0] > Fbeta and fit[i, 0] < Fdelta:
            Fdelta, Xdelta = fit[i, 0], X[i, :].reshape(1,-1)
    
    # Update dynamic boundaries
    upper = np.maximum(upper, np.max(X, axis=0))
    lower = np.minimum(lower, np.min(X, axis=0))
    
    curve[0, t] = Falpha
    t += 1
    
    while t < max_iter:
        # Coefficient linearly decreasing from 2 to 0
        a = 2 - t * (2 / max_iter)
        
        # Opposition-based learning
        threshold = a  # Threshold for opposition
        
        # Generate opposite solutions for the least fit individuals
        # 在这里反向
        # 加一个排序的操作
        
        # Get the indices that would sort fit in ascending order
        sorted_indices = np.argsort(fit, axis=0).flatten()

        # Sort both arrays using the same indices
        fit = fit[sorted_indices]
        X = X[sorted_indices]
        X = corOppose(X,fit,ub,lb,upper,lower,threshold)
        
        # Update positions using the best solutions
        for i in range(N):
            for d in range(dim):
                # Parameters
                r1, r2 = rand(), rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                X1 = Xalpha[0, d] - A1 * Dalpha
                
                r1, r2 = rand(), rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                X2 = Xbeta[0, d] - A2 * Dbeta
                
                r1, r2 = rand(), rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                X3 = Xdelta[0, d] - A3 * Ddelta
                
                X[i, d] = (X1 + X2 + X3) / 3
                # Boundary handling
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
        
        # Update binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Update fitness values
        for i in range(N):
            curr_fit = Fun(xtrain, ytrain, Xbin[i, :], opts)
            fit[i, 0] = curr_fit
            # Update alpha, beta, delta
            if curr_fit < Falpha:
                Falpha = curr_fit
                Xalpha = X[i, :].reshape(1,-1)
            elif curr_fit < Fbeta:
                Fbeta = curr_fit
                Xbeta = X[i, :].reshape(1,-1)
            elif curr_fit < Fdelta:
                Fdelta = curr_fit
                Xdelta = X[i, :].reshape(1,-1)
        
        # Update dynamic boundaries
        upper = np.maximum(upper, X.max(axis=0))
        lower = np.minimum(lower, X.min(axis=0))
        
        curve[0, t] = Falpha
        t += 1
    
    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim).reshape(dim)
    sel_index = np.where(Gbin == 1)[0]
    num_feat = len(sel_index)
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return gwo_data