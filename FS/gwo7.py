# https://blog.csdn.net/weixin_43821559/article/details/113983507
#  An Efficient Modified Grey Wolf Optimizer with Lévy Flight for Optimization Tasks
# https://www.sciencedirect.com/science/article/pii/S1568494617303873?via%3Dihub
import numpy as np  
from numpy.random import rand  
from scipy.special import gamma  
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
  
    N = opts['N']  
    max_iter = opts['T']  
  
    # Dimension  
    dim = 100        if ('runcec' in opts and opts['runcec'] == True) else np.size(xtrain, 1)  
    if np.size(lb) == 1:  
        ub = ub * np.ones([1, dim], dtype='float')  
        lb = lb * np.ones([1, dim], dtype='float')  
  
    # Initialize positions  
    Positions = init_position(lb, ub, N, dim)  
    old_Positions = Positions.copy()
    # Initialize alpha, beta, and delta  
    Alpha_pos = np.zeros([1, dim], dtype='float')  
    Alpha_score = float('inf')  
    Beta_pos = np.zeros([1, dim], dtype='float')  
    Beta_score = float('inf')  
    Delta_pos = np.zeros([1, dim], dtype='float')  
    Delta_score = float('inf')  
    # Pre  
    Convergence_curve = np.zeros([1, max_iter], dtype='float')  
    l = 0  
    
    # Main loop  
    while l < max_iter:  
        for i in range(N):  
            # Boundary handling  
            for d in range(dim):
                Positions[i, d] = boundary(Positions[i, d], lb[0, d], ub[0, d])
        # Binary conversion
        Pbin = binary_conversion(Positions, thres, N, dim)        
        for i in range(N):            
            # Calculate fitness  
            fitness = Fun(xtrain, ytrain, Pbin[i], opts,np.clip(Positions[i,:],lb,ub))  
            # Update alpha, beta, and delta  
            if fitness < Alpha_score:  
                Alpha_score = fitness  
                Alpha_pos = Positions[i, :]
            elif fitness < Beta_score:  
                Beta_score = fitness  
                Beta_pos = Positions[i, :]  
            elif fitness < Delta_score:  
                Delta_score = fitness  
                Delta_pos = Positions[i, :]  
        a = 2 - l * (2 / max_iter)  # Linearly decreasing coefficient  
        new_Position = np.zeros_like(Positions)
        for i in range(N):
              
            for j in range(dim): 
                r1 = rand()  
                r2 = rand()  
                A1 = 2 * a * r1 - a  
                C1 = 2 * r2 
                D_alpha = abs(C1 * Alpha_pos[ j] - Positions[i, j])  
                X1 = Alpha_pos[ j] - A1 * D_alpha  
  
                r1 = rand()  
                r2 = rand()  
                A2 = 2 * a * r1 - a  
                C2 = 2 * r2  
                D_beta = abs(C2 * Beta_pos[ j] - Positions[i, j])  
                X2 = Beta_pos[ j] - A2 * D_beta  
  
                # Lévy flight  
                beta = 1.5  # Lévy exponent  
                sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /  
                            (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)  
                u = np.random.normal(0, sigma_u)  
                v = np.random.normal(0, 1)  
                alpha_levi = 0.01 * u / (abs(v) ** beta)  
                A = 2 * a * rand() - a
                if abs(A) > 1:  
                    new_Position[i, j] = 0.5 * (X1 + X2) + alpha_levi * (Positions[i, j] - Alpha_pos[ j])  
                else:  
                    new_Position[i, j] = 0.5 * (X1 + X2)
                # Boundary handling  
                new_Position[i, j] = boundary(new_Position[i, j], lb[0, j], ub[0, j])  
            # binary_conversion(old_Positions[i, :],thres,1,dim)
            # Greedy selection  
            rnew = rand()  
            p = rand()  
            f1 = Fun(xtrain, ytrain, Pbin[i,:], opts,np.clip(Positions[i,:],lb,ub))
            f2 = Fun(xtrain, ytrain, binary_conversion(new_Position[i, :].reshape(1,-1),thres,1,dim).flatten(), opts,np.clip(new_Position[i,:],lb,ub))
            if f1 > f2 and rnew < p:  
                Positions[i, :] = new_Position[i, :]  # Keep the old position  
  
        Convergence_curve[0, l] = Alpha_score  
        l += 1  
  
    # Best feature subset  
    Alpha_pos = Alpha_pos.reshape(1, dim)
    Gbin = binary_conversion(Alpha_pos, thres, 1, dim)  
    Gbin = Gbin.reshape(dim)  
    pos = np.asarray(range(0, dim))  
    sel_index = pos[Gbin == 1]  
    num_feat = len(sel_index)  
  
    # Create dictionary  
    gwo_data = {'sf': sel_index, 'c': Convergence_curve, 'nf': num_feat}  
  
    return gwo_data  