#[2014]-"Grey wolf optimizer"

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
    ub = opts['ub']  if ('runcec' in opts and opts['runcec'] == True) else 1
    lb = opts['lb']  if ('runcec' in opts and opts['runcec'] == True) else 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X      = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin   = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit    = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta  = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha      = fit[i,0]
            
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fit[i,0]
            
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = Falpha.copy()
    # print("Iteration:", t + 1)
    # print("Best (GWO):", curve[0,t])
    t += 1
    
    while t < max_iter:  
        # ========== 反向学习策略 ==========
        # 计算当前种群方差
        var_current = np.sum(np.var(X, axis=0))
        var_standerd = var_current 
        
        # 计算反向个体数量
        reverse_num = round(min(var_standerd, 0.3) * N)
        reversed_num = 0
        # 生成反向个体
        if reverse_num > 0:
            selected_indices = np.random.choice(N, size=reverse_num, replace=False)
            
            # 计算当前排名
            sorted_indices = np.argsort(fit.flatten())
            ranks = np.zeros(N, dtype=int)
            for rank, idx in enumerate(sorted_indices, 1):
                ranks[idx] = rank
            
            for idx in selected_indices:
                original = X[idx].copy()
                # reverse_individual = lb + ub - original  # 生成反向解
                # 生成部分反向解。在特征空间中随机选择一部分特征进行反向
                reverse_indices = np.random.choice(dim, size=int(0.1 * dim), replace=False)
                reverse_individual = X[idx].copy()
                # for ri in reverse_indices:
                #     reverse_individual[ri] = lb[0][0] + ub[0][0] - reverse_individual[ri]
                reverse_individual[reverse_indices] = lb[0, reverse_indices] + ub[0, reverse_indices] - reverse_individual[reverse_indices]
                # 将reverse_individual化为1xdim的形式
                reverse_individual = reverse_individual.reshape(1, dim)
                f_reverse = Fun(xtrain, ytrain, binary_conversion(reverse_individual, thres, 1, dim)[0], opts,np.clip(reverse_individual,lb,ub))
                f_original = fit[idx]
                
                # 计算适应度比值
                if f_original == 0:
                    k = np.inf
                else:
                    k = f_reverse / f_original
                
                # 计算Logistic函数值
                logistic_k = 1 / (1 + np.exp(-k - 0.10))
                
                # 计算替换概率
                a = (logistic_k * ranks[idx]) / N
                
                # 决定是否替换
                if a > np.random.uniform():
                    # print(f"Reverse individual at index {idx}")
                    X[idx] = reverse_individual
                    fit[idx] = f_reverse
                    reversed_num += 1
        # if reversed_num > 0:
        #     print(f"Reversed number: {reversed_num}")
      	# Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter) 
        
        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1     = 2 * rand()
                C2     = 2 * rand()
                C3     = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A (3.3)
                A1     = 2 * a * rand() - a
                A2     = 2 * a * rand() - a
                A3     = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6) 
                X1     = Xalpha[0,d] - A1 * Dalpha
                X2     = Xbeta[0,d] - A2 * Dbeta
                X3     = Xdelta[0,d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i,d] = (X1 + X2 + X3) / 3                
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin  = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha      = fit[i,0]
                
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta       = fit[i,0]
                
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fit[i,0]
        
        curve[0,t] = Falpha.copy()
        # print("Iteration:", t + 1)
        # print("Best (GWO):", curve[0,t])
        t += 1
    
                
    # Best feature subset
    Gbin       = binary_conversion(Xalpha, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return gwo_data 
        
                
                
                
    
