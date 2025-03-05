# https://blog.csdn.net/weixin_43821559/article/details/125255471
# https://qikan.cqvip.com/Qikan/Article/Detail?id=7110581628
# 基于动态权重和游走策略的自适应灰狼优化算法
# 具有自适应搜索策略的灰狼优化算法
# 仿真实验设置：
# 最大迭代次数 tmax​=1000
# 种群大小 N=30
# PSOGWO算法中学习因子 c1​=c2​=2
# IGWO1算法、IGWO3算法和IGWO中调节因子为 n=600
# IGWO2算法调节因子为 n=800
# 算法性能：
# 在解决单峰和多峰函数时，IGWO算法具有更好的效果。
# 在解决固定多峰函数时，IGWO1和IGWO2的改进策略具有更好的效果。
# 算法比较：
# IGWO算法和IGWO3算法的收敛速度有大幅度提高，其中IGWO算法的收敛速度更快，说明游走策略对收敛速度有一定积极影响，但影响不大。
# IGWO算法相比IGWO3算法的收敛曲线多出许多折线，说明游走策略在α狼陷入局部最优时，可帮助算法逃离局部最优。
# 对比IGWO1算法和IGWO2算法的收敛曲线，会发现两种算法的收敛速度、收敛精度以及稳定性都不一样，说明调节因子 n 的改变会影响算法的性能。
# 结论：
# 根据实际问题，设置调节因子 n 和调节参数 α 的衰减方式的方法是有效的。

# 实验设置：
# 在相同实验环境下，建立20×20和50×50的普通栅格地图，再建立同等大小的特征栅格地图。
# 使用的算法包括：GA算法、PSO算法、GWO算法和IGWO算法。
# 路径规划参数：
# 初始种群大小 N=30
# 最大迭代次数 tmax​=100
# GA算法中交叉概率为0.8，变异概率为0.1
# PSO算法中学习因子 c1​=c2​=2，惯性权值 ωini​=0.9，ωend​=0.4
# IGWO算法中调节因子 n=600
# 20×20的特征栅格地图中，距离限制 D=5，可视步长 L=5
# 50×50的特征栅格地图中，距离限制 D=15，可视步长 L=10
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
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    
    N = opts['N']
    max_iter = opts['T']
    # set the parameters of the algorithm
    n = 600
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
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t = 0
    
    curve[0,t] = Falpha.copy()
    t += 1
    
    while t < max_iter:  
        # Coefficient decreases linearly from 2 to 0 
        a = 0
        if t < 2*n**2/max_iter:
            a = 2 - (t /n)**2
        elif t > 2*n**2/max_iter:
            a = 2*(t - max_iter)**2/(max_iter**2 -2*n**2)
        # Calculate dynamic weights
        fitness = [Falpha, Fbeta, Fdelta]
        w1 = max(fitness) / (abs(fitness[0]) + abs(fitness[1]) + abs(fitness[2]))
        w2 = (sum(fitness) - max(fitness) - min(fitness)) / (abs(fitness[0]) + abs(fitness[1]) + abs(fitness[2]))
        w3 = min(fitness) / (abs(fitness[0]) + abs(fitness[1]) + abs(fitness[2]))
        
        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                Dbeta = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A (3.3)
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6) 
                X1 = Xalpha[0,d] - A1 * Dalpha
                X2 = Xbeta[0,d] - A2 * Dbeta
                X3 = Xdelta[0,d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i,d] = (w1 * X1 + w2 * X2 + w3 * X3)
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
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
        
        # Wandering strategy
        num_wander = round(N / 4)
        new_population = np.zeros((N + num_wander * 2, dim))
        new_fitness = np.zeros((N + num_wander * 2, 1))

        # 复制当前种群
        new_population[:N, :] = X
        new_fitness[:N, :] = fit

        for i in range(num_wander):
            idx1 = np.random.randint(0, N)
            idx2 = np.random.randint(0, N)
            while idx2 == idx1:
                idx2 = np.random.randint(0, N)
            for d in range(dim):
                new_population[N + 2 * i, d] = X[idx1, d] + rand() * (X[idx2, d] - X[idx1, d])
                new_population[N + 2 * i, d] = boundary(new_population[N + 2 * i, d], lb[0, d], ub[0, d])
            for d in range(dim):
                new_population[N + 2 * i + 1, d] = X[idx1, d] + rand() * (Xalpha[0, d] - X[idx1, d])
                new_population[N + 2 * i + 1, d] = boundary(new_population[N + 2 * i + 1, d], lb[0, d], ub[0, d])

        # Binary conversion after wandering
        Xbin = binary_conversion(new_population, thres, N + num_wander * 2, dim)

        # Fitness after wandering
        for i in range(N + num_wander * 2):
            new_fitness[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)

        # 选择适应度前 N 的个体，作为新的种群
        sorted_indices = np.argsort(new_fitness.flatten())
        X = new_population[sorted_indices[:N], :]
        fit = new_fitness[sorted_indices[:N], :]

        # 更新 Alpha, Beta, Delta
        for i in range(N):
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]
            elif fit[i, 0] < Fbeta:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]
            elif fit[i, 0] < Fdelta:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]
        
        curve[0,t] = Falpha.copy()
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