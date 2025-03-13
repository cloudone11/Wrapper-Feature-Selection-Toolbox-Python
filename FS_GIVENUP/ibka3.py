import numpy as np
from numpy.random import rand
from tqdm import tqdm
from FS.functionHO import Fun

# 初始化函数
def initialization(N, Dim, UB, LB):
    if not isinstance(UB[0], list):
        X = (np.random.rand(N, Dim) * (UB[0] - LB[0])) + LB[0]
    else:
        UB = UB[0]
        LB = LB[0]
        X = np.zeros((N, Dim))
        for i in range(Dim):
            Ub_i = UB[i]
            Lb_i = LB[i]
            base_i = np.random.rand(N, 1) * (Ub_i - Lb_i)
            base_lb = np.ones((N, 1)) * Lb_i
            X[:, i] = (base_i + base_lb).squeeze()
    return X
def halton_initialization(N, Dim, UB, LB):
    sampler = Halton(d=Dim, scramble=True)
    X = sampler.random(n=N)
    X = LB + X * (UB - LB)
    return X
# 二进制转换
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    return Xbin

# 边界处理
def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def golden_sine_mutation( X, XLeader_Pos):
        # 黄金正弦变异
        phi = (1 + np.sqrt(5)) / 2  # 黄金比例
        r1 = rand()
        r2 = rand()
        XNew = X + phi * np.sin(2 * np.pi * r1) * (X - XLeader_Pos) * r2
        return XNew
# 黑翅鸢优化算法
class BKA:
    def __init__(self, pop, T, lb, ub, dim, fobj):
        self.pop = pop
        self.T = T
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.fobj = fobj
        self.XPos = initialization(pop, dim, self.ub, self.lb)
        self.XFit = np.array([fobj(X) for X in self.XPos])
        self.Best_Fitness_BKA = float('inf')
        self.Best_Pos_BKA = None
        self.p = 0.9
        self.Convergence_curve = []
        self.initial_diversity = np.mean([np.linalg.norm(self.XPos[i] - np.mean(self.XPos, axis=0)) for i in range(pop)])

    def optimize(self):
        for t in tqdm(range(0, self.T)):
            # 计算当前多样性
            diversity = np.mean([np.linalg.norm(self.XPos[i] - np.mean(self.XPos, axis=0)) for i in range(self.pop)])
            # 动态调整变异概率
            p_mutation = 0.1 + 0.4 * (1 - diversity / self.initial_diversity)

            sorted_indexes = np.argsort(self.XFit)
            XLeader_Pos = self.XPos[sorted_indexes[0]]
            XLeader_Fit = self.XFit[sorted_indexes[0]]

            for i in range(self.pop):
                # 自适应变异
                if rand() < p_mutation:
                    self.XPos[i] = self.golden_sine_mutation(self.XPos[i], XLeader_Pos)

                n = 0.05 * np.exp(-2 * (t / self.T) ** 2)
                r = rand()
                if self.p < r:
                    XPosNew = self.XPos[i] + n * (1 + np.sin(r)) * self.XPos[i]
                else:
                    XPosNew = self.XPos[i] * (n * (2 * r - 1) + 1)
                XPosNew = np.clip(XPosNew, self.lb, self.ub)
                XFit_New = self.fobj(XPosNew)
                if XFit_New < self.XFit[i]:
                    self.XPos[i] = XPosNew
                    self.XFit[i] = XFit_New

            for i in range(self.pop):
                r = rand()
                m = 2 * np.sin(r + np.pi / 2)
                s = np.random.randint(0, self.pop)
                r_XFitness = self.XFit[s]
                ori_value = rand(self.dim)
                cauchy_value = np.tan((ori_value - 0.5) * np.pi)
                if self.XFit[i] < r_XFitness:
                    XPosNew = self.XPos[i] + cauchy_value * (self.XPos[i] - XLeader_Pos)
                else:
                    XPosNew = self.XPos[i] + cauchy_value * (XLeader_Pos - m * self.XPos[i])
                XPosNew = np.clip(XPosNew, self.lb, self.ub)
                XFit_New = self.fobj(XPosNew)
                if XFit_New < self.XFit[i]:
                    self.XPos[i] = XPosNew
                    self.XFit[i] = XFit_New

            for i in range(self.pop):
                if self.XFit[i] < XLeader_Fit:
                    self.Best_Fitness_BKA = min(self.Best_Fitness_BKA, self.XFit[i])
                    self.Best_Pos_BKA = self.XPos[i, :]
                else:
                    self.Best_Fitness_BKA = min(self.Best_Fitness_BKA, XLeader_Fit)
                    self.Best_Pos_BKA = XLeader_Pos
            self.Convergence_curve.append(self.Best_Fitness_BKA)
        return self.Best_Fitness_BKA, self.Best_Pos_BKA, self.Convergence_curve

    def golden_sine_mutation(self, X, XLeader_Pos):
        # 黄金正弦变异
        phi = (1 + np.sqrt(5)) / 2  # 黄金比例
        r1 = rand()
        r2 = rand()
        XNew = X + phi * np.sin(2 * np.pi * r1) * (X - XLeader_Pos) * r2
        return np.clip(XNew, self.lb, self.ub)
    
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
    X = initialization(N, dim, ub, lb)
    fit = np.zeros([N, 1], dtype='float')
    Xrb = np.zeros([1, dim], dtype='float')
    fitR = float('inf')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    XPos = binary_conversion(X, thres, N, dim)
    initial_diversity = np.mean([np.linalg.norm(XPos[i] - np.mean(XPos, axis=0)) for i in range(N)])
    while t < max_iter:
        Xbin = binary_conversion(X, thres, N, dim)
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
            if fit[i, 0] < fitR:
                Xrb[0, :] = X[i, :]
                fitR = fit[i, 0]
        curve[0, t] = fitR.copy()
        print("Iteration:", t + 1)
        print("Best (BKA):", curve[0, t])
        t += 1
        sorted_indexes = np.argsort(fit[:, 0])
        XLeader_Pos = X[sorted_indexes[0]]
        XLeader_Fit = fit[sorted_indexes[0]]
        # 计算当前多样性
        diversity = np.mean([np.linalg.norm(X[i] - np.mean(X, axis=0)) for i in range(N)])
        # 动态调整变异概率
        p_mutation = 0.1 + 0.4 * (1 - diversity / initial_diversity)
        for i in range(N):
            # 自适应变异
            if rand() < p_mutation:
                X[i] = golden_sine_mutation(X[i], XLeader_Pos)

            n = 0.05 * np.exp(-2 * (t / max_iter) ** 2)
            r = rand()
            
            if 0.9 < r:
                XPosNew = X[i] + n * (1 + np.sin(r)) * X[i]
            else:
                XPosNew = X[i] * (n * (2 * r - 1) + 1)
            XPosNew = np.clip(XPosNew, lb, ub)
            XFit_New = Fun(xtrain, ytrain, binary_conversion(XPosNew.reshape(1, -1), thres, 1, dim)[0], opts)
            if XFit_New < fit[i, 0]:
                X[i] = XPosNew
                fit[i,0] = XFit_New
                
        for i in range(N):
            n = 0.05 * np.exp(-2 * (t / max_iter) ** 2)
            r = np.random.rand()
            if 0.9 < r:
                XPosNew = X[i] + n * (1 + np.sin(r)) * X[i]
            else:
                XPosNew = X[i] * (n * (2 * r - 1) + 1)
            XPosNew = np.clip(XPosNew, lb, ub)
            XFit_New = Fun(xtrain, ytrain, binary_conversion(XPosNew.reshape(1, -1), thres, 1, dim)[0], opts)
            if XFit_New < fit[i, 0]:
                X[i] = XPosNew
                fit[i, 0] = XFit_New
        for i in range(N):
            r = np.random.rand()
            m = 2 * np.sin(r + np.pi / 2)
            s = np.random.randint(0, N)
            r_XFitness = fit[s, 0]
            ori_value = np.random.rand(dim)
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)
            if fit[i, 0] < r_XFitness:
                XPosNew = X[i] + cauchy_value * (X[i] - XLeader_Pos)
            else:
                XPosNew = X[i] + cauchy_value * (XLeader_Pos - m * X[i])
            XPosNew = np.clip(XPosNew, lb, ub)
            XFit_New = Fun(xtrain, ytrain, binary_conversion(XPosNew.reshape(1, -1), thres, 1, dim)[0], opts)
            if XFit_New < fit[i, 0]:
                X[i] = XPosNew
                fit[i, 0] = XFit_New
    Gbin = binary_conversion(Xrb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    bka_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    return bka_data

# 测试代码
if __name__ == "__main__":
    def example_obj(inx):
        return np.sum(inx ** 2)

    pop = 20
    T = 3000
    lb = [-100]
    ub = [100]
    dim = 30
    bka_optimizer = BKA(pop, T, lb, ub, dim, example_obj)
    Best_Fit_BKA, Best_Pos_BKA, Convergence_curve = bka_optimizer.optimize()
    print("Best Fitness:", Best_Fit_BKA)