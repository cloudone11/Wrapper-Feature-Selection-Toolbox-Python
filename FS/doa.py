import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
# 不好复现，暂不复现。
def initialization(pop, D, ub, lb):
    return lb + (ub - lb) * np.random.rand(pop, D)

def binary_conversion(X, thres=0.5):
    return (X > thres).astype(int)

def boundary_handle(x, lb, ub, D, pop_size, current_idx):
    if D > 15:
        select = np.delete(np.arange(pop_size), current_idx)
        print(select[np.random.randint(pop_size-1)])
        print(x.shape)
        return x[select[np.random.randint(pop_size-1)]]
    else:
        return lb + (ub - lb) * rand()

def jfs(xtrain, ytrain, opts):
    # 参数解析
    N = opts['N']
    max_iter = opts['T']
    dim = xtrain.shape[1]
    lb = np.zeros((1, dim))
    ub = np.ones((1, dim))
    thres = 0.5
    
    # 初始化
    X = initialization(N, dim, ub, lb)
    Xbin = binary_conversion(X, thres)
    
    # 记录变量
    fbest = np.inf
    sbest = np.zeros(dim)
    fbest_history = np.zeros(max_iter)
    group_num = 5
    
    # 分组初始化
    group_size = N // group_num
    sbestd = np.tile(lb, (group_num, 1))
    fbestd = np.full(group_num, np.inf)
    
    for t in range(max_iter):
        # 探索阶段 (前90%迭代)
        if t < 0.9 * max_iter:
            for m in range(group_num):
                # 更新组最优
                group_start = m * group_size
                group_end = (m+1) * group_size
                for j in range(group_start, group_end):
                    current_fit = Fun(xtrain, ytrain, binary_conversion(X[j:j+1])[0], opts)
                    if current_fit < fbestd[m]:
                        sbestd[m] = X[j].copy()
                        fbestd[m] = current_fit
                
                # 组内更新
                for j in range(group_start, group_end):
                    X[j] = sbestd[m].copy()
                    k = np.random.randint(np.ceil(dim/(8*(m+1))), np.ceil(dim/(3*(m+1))) + 1)
                    indices = np.random.choice(dim, k, replace=False)
                    
                    if rand() < 0.9:
                        for h in indices:
                            X[j,h] += (rand()*(ub[0,h]-lb[0,h]) + lb[0,h]) * (np.cos((t + max_iter/10)*np.pi/max_iter) + 1)/2
                            if X[j,h] > ub[0,h] or X[j,h] < lb[0,h]:
                                X[j,h] = boundary_handle(X, lb[0,h], ub[0,h], dim, N, j)
                    else:
                        for h in indices:
                            X[j,h] = X[np.random.randint(N), h]
                
                # 更新全局最优
                if fbestd[m] < fbest:
                    fbest = fbestd[m]
                    sbest = sbestd[m].copy()
        
        # 开发阶段 (后10%迭代)
        else:
            # 更新全局最优
            for j in range(N):
                current_fit = Fun(xtrain, ytrain, binary_conversion(X[j:j+1])[0], opts)
                if current_fit < fbest:
                    fbest = current_fit
                    sbest = X[j].copy()
            
            # 开发阶段更新
            km = max(2, np.ceil(dim/3).astype(int))
            for j in range(N):
                X[j] = sbest.copy()
                k = np.random.randint(2, km+1)
                indices = np.random.choice(dim, k, replace=False)
                for h in indices:
                    X[j,h] += (rand()*(ub[0,h]-lb[0,h]) + lb[0,h]) * (np.cos(t*np.pi/max_iter) + 1)/2
                    if X[j,h] > ub[0,h] or X[j,h] < lb[0,h]:
                        X[j,h] = boundary_handle(X, lb[0,h], ub[0,h], dim, N, j)
        
        # 记录收敛曲线
        fbest_history[t] = fbest
    
    # 最终特征选择
    Gbin = binary_conversion(sbest.reshape(1,-1), thres)[0]
    sel_index = np.where(Gbin == 1)[0]
    
    return {
        'sf': sel_index,
        'c': fbest_history.reshape(1,-1),
        'nf': len(sel_index)
    }