import numpy as np
from numpy.random import rand
from numpy.random import randint
from FS.functionHO import Fun
from scipy.special import gamma  
# 定义适应度函数（需要根据实际问题替换）
def Fun(xtrain, ytrain, Xbin, opts):
    # 示例：假设分类器的准确度
    # 返回负的准确度作为适应度（因为 SBOA 是最小化问题）
    return np.random.rand()
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
        return lb
    elif x > ub:
        return ub
    else:
        return x

def Levin(dim):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta)/2) * beta * 2 ** ((beta - 1)/2))) ** (1/beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1/beta)
    return step

def jfs(xtrain, ytrain, opts):
    # 参数
    ub = 1
    lb = 0
    thres = 0.5
    
    N = opts['N']  # 种群大小
    Max_iter = opts['T']  # 最大迭代次数
    dim = np.size(xtrain, 1)  # 输入特征维度
    
    # 如果输入的边界是标量，将其扩展为向量
    if np.size(lb) == 1:
        lb = lb * np.ones(dim, dtype=float)
    if np.size(ub) == 1:
        ub = ub * np.ones(dim, dtype=float)
    
    # 初始化种群
    X = np.zeros((N, dim), dtype=float)
    for i in range(dim):
        X[:, i] = lb[i] + rand(N, 1).flatten() * (ub[i] - lb[i])
    
    # 计算二值化特征选择
    Xbin = np.where(X > thres, 1, 0)
    
    # 初始适应度评估
    fit = np.zeros((N, 1), dtype=float)
    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
    
    # 初始化最优解
    Bast_P = np.zeros((1, dim), dtype=float)
    fbest = float('inf')
    
    # 主循环
    best_so_far = np.zeros((1, Max_iter), dtype=float)
    for t in range(Max_iter):
        CF = (1 - t/Max_iter) ** (2 * t/Max_iter)
        # 更新全局最优
        current_best_index = np.argmin(fit)
        current_best = fit[current_best_index]
        
        if current_best < fbest:
            fbest = current_best
            Bast_P = X[current_best_index, :].copy()
        
        # 秘书鸟捕猎策略
        for i in range(N):
            if t < Max_iter / 3:  # 捕猎阶段 1
                X_random_1 = randint(0, N)
                X_random_2 = randint(0, N)
                R1 = rand(dim)
                X1 = X[i, :] + (X[X_random_1, :] - X[X_random_2, :]) * R1
            elif Max_iter / 3 <= t < 2 * Max_iter / 3:  # 捕猎阶段 2
                RB = np.random.randn(dim)
                X1 = Bast_P + np.exp((t/Max_iter)**4) * (RB - 0.5) * (Bast_P - X[i, :])
            else:  # 捕猎阶段 3
                RL = 0.5 * Levin(dim)
                X1 = Bast_P + CF * X[i, :] * RL
            # 边界检查
            for d in range(dim):
                X1[d] = boundary(X1[d], lb[d], ub[d])
            # 更新位置
            X[i, :] = X1.copy()
        
        # 秘书鸟逃离策略
        r = rand()
        X_random = X[randint(N), :]
        for i in range(N):
            if r < 0.5:  # 隐藏策略
                RB = rand(dim)
                X2 = Bast_P + (1 - t/Max_iter) ** 2 * (2 * RB - 1) * X[i, :]
            else:  # 逃离策略
                K = np.round(1 + rand())
                R2 = rand(dim)
                X2 = X[i, :] + R2 * (X_random - K * X[i, :])
            # 边界检查
            for d in range(dim):
                X2[d] = boundary(X2[d], lb[d], ub[d])
            # 更新位置
            X[i, :] = X2.copy()
        
        # 计算适应度
        Xbin = np.where(X > thres, 1, 0)
        for i in range(N):
            current_fit = Fun(xtrain, ytrain, Xbin[i, :], opts)
            if current_fit < fit[i]:
                fit[i] = current_fit
            if current_fit < fbest:
                fbest = current_fit
                Bast_P = X[i, :].copy()
        
        # 记录最优值
        best_so_far[0, t] = fbest
    
    # 最优特征子集
    # Best feature subset
    Gbin       = binary_conversion(Bast_P.reshape(1,-1), thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    
    # 输出结果
    sboa_data = {
        'sf': sel_index,  # 选择的特征索引
        'c': best_so_far,  # 收敛曲线
        'nf': num_feat  # 特征数量
    }
    
    return sboa_data