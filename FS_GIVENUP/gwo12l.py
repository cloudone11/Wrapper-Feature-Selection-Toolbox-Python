# https://image.hanspub.org/Html/8-1540774_21093.htm#txtF5
# 小生境灰狼优化(NGWO)算法
    # 算法与实验设置：
    #     使用NGWO（新型灰狼优化）算法对5个基准测试函数进行数值实验。
    #     参数设置：种群规模 N=30，最大迭代次数 t=500，每个函数的维数 D=30，小生境半径 σshare​=0.5。
    #     PSO（粒子群优化）算法参数：种群规模 N=30，最大迭代次数 t=500，c1​=c2​=2，惯性权重 ω∈[0.2,0.9]，Vmax​=6。
    # 实验结果：
    #     对于只有一个全局最优解的单峰函数，测试结果表明算法的寻优能力。
    #     对于有多个极值点的多峰函数，测试结果表明算法跳出局部最优解的能力。
    #     对于Sphere函数、Schwefel函数、Rastrigin函数、Griewank函数，NGWO算法均能较好地找到全局最优解，且得到较优的平均值、标准差。
    #     对于Griewank函数，算法能找到理论最优值0，而对Sphere函数其寻优精度达到了 10−30。
    #     Step函数由于属于阶跃函数，由很多平滑的高地和陡脊组成，且不连续，所以在寻求最优值上有一定的难度，PSO算法却能获得较好的全局最优解。
    # 图示分析：
    #     图2-6给出了5个基准函数在三种算法下的进化收敛曲线。
    #     除Step函数外其余4个函数均能较快的收敛到全局最优解。
    # 算法性能：
    #     改进的灰狼优化算法在处理函数优化问题时表现出优越的稳定性及鲁棒性，具有更好的优化性能。
    # 没有指出算法的罚函数是如何定义的。
import numpy as np
from numpy.random import rand
from FS.functionHO import Fun

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X
from pyDOE import lhs
def lhs_initialization(LB,UB,N, Dim):
    X = lhs(Dim, samples=N)
    X = LB + X * (UB - LB)
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
    sigma_share = 0.5  # Niching radius

    N = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X = lhs_initialization(lb, ub, N, dim)
    
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
    # set the parameters of the algorithm
    penalty = 0.02
    while t < max_iter:
        # Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter)
        
        # Calculate distances between individuals
        # initialise the array of fitness to add for penalty
        fitness_2add = np.zeros([N, 1], dtype='float')
        for i in range(N):
            for j in range(N):
                if i != j:
                    d_ij = np.linalg.norm(X[i,:] - X[j,:])
                    if d_ij < sigma_share:
                        # Apply penalty function to the individual with worse fitness                        
                        if fit[i,0] < fit[j,0]:
                            fitness_2add[j,0] += penalty
        # Update fitness
        fit += fitness_2add
        # Update positions
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
                X[i,d] = (X1 + X2 + X3) / 3                
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