import numpy as np  
from numpy.random import rand  
from FS.functionHO import Fun
# https://blog.csdn.net/weixin_43821559/article/details/116561969
# http://kzyjc.alljournals.cn/kzyjc/article/abstract/20171003?st=search
# 协调探索和开发能力的改进灰狼优化算法

# 以下是从提供的图像中提取的信息：
# 参数敏感性分析：
# IGWO算法对参数 ainitial​ 和 afinal​ 不太敏感。当 ainitial​=1 和 afinal​=0 时，算法的总体寻优性能最好，因此这是合理的参数选择。
# IGWO算法对参数 ε 不太敏感。当 ε=5 时，算法的总体寻优性能最佳，因此 ε=5 是最佳的参数选择。
# 参数 b1​ 和 b2​ 的影响：
# 群体交流参数 b1​ 和个体记忆系数 b2​ 对算法性能影响较大。通过调节 b1​ 和 b2​ 的值，可以改变算法的探索和开发能力。
# 除了函数 f2​ 和 f4​，IGWO算法对其他8个函数的测试结果对 b1​ 和 b2​ 的值不太敏感。
# 从5组不同的取值结果来看，当 b1​=0.7 和 b2​=0.3 时，IGWO算法得到的结果相对较好。因此，b1​=0.7,b2​=0.3 是合理的参数取值。
# 这些信息表明，IGWO算法在参数选择上具有一定的灵活性，但某些参数（如 b1​ 和 b2​）对算法性能有显著影响。通过适当的参数调整，可以优化算法的性能。
def init_position_good_set(lb, ub, N, dim):  
    # Generate good point set  
    # This is a placeholder for the actual good point set generation  
    # which should be implemented based on the specific method  
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
  
    # Initialize position using good point set  
    X = init_position_good_set(lb, ub, N, dim)  
  
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
  
    # Pre  
    curve = np.zeros([1, max_iter], dtype='float')  
    t = 0  
  
    curve[0, t] = Falpha.copy()  
    t += 1  
  
    # Non-linear control parameter  
    a_initial = 1  
    a_final = 0  
    epsilon = 5 
    b1      = 0.7 # Group interaction coefficient 
    b2      = 0.3 # Individual interaction coefficient  
    T = max_iter  
  
    # Individual memory  
    P_best = np.copy(X)  
    P_best_fit = np.copy(fit)  
  
    while t < max_iter:  
        # Update non-linear control parameter  
        a = a_initial - (a_initial - a_final) * np.tan((1 / epsilon) * (t / T) * np.pi)  
        a = max(0, a)  # Ensure a does not go below 0  
  
        for i in range(N):  
            for d in range(dim):  
                # Parameter C  
                C1 = 2 * rand()  
                C2 = 2 * rand()  
                C3 = 2 * rand()  
  
                # Compute Dalpha, Dbeta, Ddelta  
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])  
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])  
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])  
  
                # Parameter A  
                A1 = 2 * a * rand() - a  
                A2 = 2 * a * rand() - a  
                A3 = 2 * a * rand() - a  
                # parameter w1, w2, w3
                w1 = Falpha / (Falpha + Fbeta + Fdelta)
                w2 = Fbeta / (Falpha + Fbeta + Fdelta)
                w3 = Fdelta / (Falpha + Fbeta + Fdelta)
                # Compute X1, X2, X3  
                X1 = Xalpha[0, d] - A1 * Dalpha  
                X2 = Xbeta[0, d] - A2 * Dbeta  
                X3 = Xdelta[0, d] - A3 * Ddelta  
  
                # Position update with individual memory  
                X[i, d] = w1*X1 + w2*X2 + w3*X3 + b2 * rand() * (P_best[i, d] - X[i, d])  
  
                # Boundary  
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  
  
        # Binary conversion  
        Xbin = binary_conversion(X, thres, N, dim)  
  
        # Fitness  
        for i in range(N):  
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)  
            if fit[i, 0] < P_best_fit[i, 0]:  
                P_best[i, :] = X[i, :]  
                P_best_fit[i, 0] = fit[i, 0]  
  
            # Update alpha, beta, delta  
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
  
    # Best feature subset  
    Gbin = binary_conversion(Xalpha, thres, 1, dim)  
    Gbin = Gbin.reshape(dim)  
    pos = np.asarray(range(0, dim))  
    sel_index = pos[Gbin == 1]  
    num_feat = len(sel_index)  
  
    # Create dictionary  
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}  
  
    return gwo_data  