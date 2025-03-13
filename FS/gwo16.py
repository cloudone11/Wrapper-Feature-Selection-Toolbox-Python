# https://blog.csdn.net/u011835903/article/details/125381450
# https://xueshu.baidu.com/usercenter/paper/show?paperid=104b0xs07v140x60cr320a90qq274108
# 基于差分进化与优胜劣汰策略的灰狼优化算法

# 灰狼群规模：30
# 缩放因子：
# 最大值 fmax​ 为 1.5
# 最小值 fmin​ 为 0.25
# 交叉概率因子 S：0.7
# 淘汰更新比例因子 η：0.618
# 最大迭代次数 tmax​：500
# ABC算法参数（参考文献[17]）：
# 蜂群总数 n=30
# 跟随蜂数目 n/2=15
# 引领蜂数目 n/2=15
# 原论文中计算m的公式应该有误，m必须小于你，但n/yita大于n，因此修改公式。
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
    N = opts['N']
    max_iter = opts['T']
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
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
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
    # initialize the parameters of the DE
    fmax = 1.5
    fmin = 0.25
    S    = 0.7
    yita = 0.618
    while t < max_iter:
        # Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter)
        
        # Update positions
        for i in range(N):
            for d in range(dim):
                # Parameter C
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d])
                Dbeta = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3
                X1 = Xalpha[0,d] - A1 * Dalpha
                X2 = Xbeta[0,d] - A2 * Dbeta
                X3 = Xdelta[0,d] - A3 * Ddelta
                # Update wolf
                X[i,d] = (X1 + X2 + X3) / 3
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts,np.clip(X[i,:],lb,ub))
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha = fit[i,0]
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:] = X[i,:]
                Fbeta = fit[i,0]
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta = fit[i,0]
        
        # Evolutionary operators (DE)
        for i in range(N):
            W = (fmax -fmin)*(max_iter - t +1)/max_iter + fmin
            V = Xalpha + W * (Xbeta - Xdelta)
            U = np.zeros([1, dim], dtype='float')
            # 在0-dim-1中随机选取一个整数
            random_index = np.random.randint(0, dim - 1)
            for d in range(dim):
                if S < rand() and d != random_index:
                    U[0,d] = V[0,d]
                else:
                    U[0,d] = X[i,d]
                U[0,d] = boundary(U[0,d], lb[0,d], ub[0,d])
            Ubin = binary_conversion(U, thres, 1, dim)
            fit_U = Fun(xtrain, ytrain, Ubin[0,:], opts,np.clip(U,lb,ub))
            if fit_U < fit[i,0]:
                X[i,:] = U[0,:]
                fit[i,0] = fit_U
        # 优胜劣汰策略，淘汰种群X中fit值最高的m个个体，再随机生成m个个体
        m = np.random.randint(int(N*yita),int(N*yita/(0.618))+1)
        # 依照适应度值对种群进行排序
        idx = np.argsort(fit[:, 0])
        # 选择适应度值最高的m个个体
        X = X[idx[:-m], :]
        fit = fit[idx[:-m], :]

        # 随机生成m个新个体
        new_X = np.random.uniform(lb, ub, (m, dim))
        new_fit = np.zeros((m, 1))
        for i in range(m):
            new_fit[i, 0] = Fun(xtrain, ytrain, new_X[i, :], opts,np.clip(new_X[i,:],lb,ub))

        # 将新个体加入种群
        X = np.vstack((X, new_X))
        fit = np.vstack((fit, new_fit))        
        
        # Update curve
        curve[0,t] = Falpha.copy()
        t += 1
    
    # Best feature subset
    Gbin = binary_conversion(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return gwo_data