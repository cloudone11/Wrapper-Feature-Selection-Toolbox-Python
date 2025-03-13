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
        
    # Initialize position and velocity
    X      = init_position(lb, ub, N, dim)
    velocity = 0.3 * np.random.randn(N, dim)  # 速度初始化
    w      = 0.5 + np.random.rand() / 2      # 固定惯性权重
    
    # Binary conversion
    Xbin   = binary_conversion(X, thres, N, dim)
    
    # Fitness initialization
    fit    = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta  = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    
    # Initial fitness evaluation
    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha = fit[i,0]
        elif fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:] = X[i,:]
            Fbeta = fit[i,0]
        elif fit[i,0] < Fdelta and fit[i,0] > Fbeta:
            Xdelta[0,:] = X[i,:]
            Fdelta = fit[i,0]
    
    # Convergence curve setup
    curve = np.zeros([1, max_iter], dtype='float') 
    t = 0
    curve[0,t] = Falpha.copy()
    t += 1
    
    while t < max_iter:  
        a = 2 - t * (2 / max_iter)  # 线性递减系数
        
        # 参数固定值设置
        C1 = C2 = C3 = 0.5
        
        # 更新每个粒子的位置
        for i in range(N):
            for d in range(dim):
                # 生成新的随机数
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                
                # 计算动态系数A
                A1 = 2 * a * r1 - a
                A2 = 2 * a * r2 - a
                A3 = 2 * a * r3 - a
                
                # 计算三个引导方向
                D_alpha = abs(C1 * Xalpha[0,d] - w * X[i,d])
                X1 = Xalpha[0,d] - A1 * D_alpha
                
                D_beta  = abs(C2 * Xbeta[0,d] - w * X[i,d])
                X2 = Xbeta[0,d] - A2 * D_beta
                
                D_delta = abs(C3 * Xdelta[0,d] - w * X[i,d])
                X3 = Xdelta[0,d] - A3 * D_delta
                
                # 更新速度(PSO机制)
                velocity[i,d] = w * (velocity[i,d] + 
                                   C1*r1*(X1 - X[i,d]) + 
                                   C2*r2*(X2 - X[i,d]) + 
                                   C3*r3*(X3 - X[i,d]))
                
                # 更新位置
                X[i,d] += velocity[i,d]
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # 二进制转换并评估适应度
        Xbin = binary_conversion(X, thres, N, dim)
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
            
            # 更新Alpha/Beta/Delta
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha = fit[i,0]
            elif fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:] = X[i,:]
                Fbeta = fit[i,0]
            elif fit[i,0] < Fdelta and fit[i,0] > Fbeta:
                Xdelta[0,:] = X[i,:]
                Fdelta = fit[i,0]
        
        curve[0,t] = Falpha.copy()
        t += 1
    
    # 最佳特征子集提取
    Gbin = binary_conversion(Xalpha, thres, 1, dim).reshape(dim)
    sel_index = np.where(Gbin == 1)[0]
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': len(sel_index)}
    
    return gwo_data