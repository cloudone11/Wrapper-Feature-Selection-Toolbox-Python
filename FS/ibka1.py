# aim1: 还原原来敲的bka优化器
# aim2: 进行两到三个优化
# aim3: 进行实验
import numpy as np
from numpy.random import rand
from tqdm import tqdm
from FS.functionHO import Fun

from pyDOE import lhs

def lhs_initialization(N, Dim, UB, LB):
    X = lhs(Dim, samples=N)
    X = LB + X * (UB - LB)
    print(X.shape)
    return X

from scipy.stats.qmc import Sobol

def sobol_initialization(N, Dim, UB, LB):
    sampler = Sobol(d=Dim, scramble=True)
    m = int(np.log2(N))
    X = sampler.random_base2(m=m)
    
    # 如果生成的样本数少于N，用随机样本填充剩余部分
    if X.shape[0] < N:
        additional_samples = N - X.shape[0]
        random_samples = np.random.rand(additional_samples, Dim)
        random_samples = LB + random_samples * (UB - LB)
        X = np.vstack((X, random_samples))
    
    X = LB + X * (UB - LB)
    print(X.shape)
    return X

from scipy.stats.qmc import Halton

def halton_initialization(N, Dim, UB, LB):
    sampler = Halton(d=Dim, scramble=True)
    X = sampler.random(n=N)
    X = LB + X * (UB - LB)
    return X
import numpy as np

def initialization_new(SearchAgents_no, dim, ub, lb, fun):
    Boundary_no = len(ub)  # number of boundaries
    BackPositions = np.zeros((SearchAgents_no, dim))
    
    # If each variable has a different lb and ub
    if Boundary_no > 1:
        PositionsF = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            PositionsF[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
            # Calculate the opposite population
            BackPositions[:, i] = (ub_i + lb_i) - PositionsF[:, i]
    
    # Get elite population
    index = np.zeros(SearchAgents_no, dtype=bool)
    for i in range(SearchAgents_no):
        if fun(PositionsF[i, :]) < fun(BackPositions[i, :]):  # Original solution is better
            index[i] = True
        else:  # Opposite solution is better
            PositionsF[i, :] = BackPositions[i, :]
    
    XJY = PositionsF[index, :]
    return XJY

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

class BKA:
    def __init__(self, pop, T, lb, ub, dim, fobj):
        self.e = 0.6
        self.w = 3
        self.pop = pop
        self.T = T
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.fobj = fobj
        self.p = 0.9
        self.XPos = sobol_initialization(pop, dim, self.ub, self.lb)
        self.XFit = np.array([fobj(X) for X in self.XPos])
        self.Best_Fitness_BKA = float('inf')
        self.Best_Pos_BKA = None
        self.Convergence_curve = []

    def optimize(self):
        for t in tqdm(range(0, self.T)):
            sorted_indexes = np.argsort(self.XFit)
            XLeader_Pos = self.XPos[sorted_indexes[0]]
            XLeader_Fit = self.XFit[sorted_indexes[0]]
            for i in range(self.pop):
                n = 0.05 * np.exp(-2 * (t / self.T) ** 2)
                r = np.random.rand()
                if self.p < r:
                    XPosNew = self.XPos[i] + n * (1 + np.sin(r)) * self.XPos[i]
                else:
                    XPosNew = self.XPos[i] * (n * (2 * r - 1) + 1)
                # 在此处应用eq8的改进
                k = 2 * r
                XPosNew = (self.ub + self.lb)*(0.5 + 1/2*k) - XPosNew/k
                
                XPosNew = np.clip(XPosNew, self.lb, self.ub)
                XFit_New = self.fobj(XPosNew)
                if XFit_New < self.XFit[i]:
                    self.XPos[i] = XPosNew
                    self.XFit[i] = XFit_New
            for i in range(self.pop):
                r = np.random.rand()
                m = 2 * np.sin(r + np.pi / 2)
                s = np.random.randint(0, self.pop)
                r_XFitness = self.XFit[s]
                ori_value = np.random.rand(self.dim)
                cauchy_value = np.tan((ori_value - 0.5) * np.pi)
                if self.XFit[i] < r_XFitness:
                    XPosNew = self.XPos[i] + cauchy_value * (self.XPos[i] - XLeader_Pos)
                else:
                    XPosNew = self.XPos[i] + cauchy_value * (XLeader_Pos - m * self.XPos[i])
                    
                # new :

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

def jfs(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    X = initialization(N, dim, ub, lb)
    fit = np.zeros([N, 1], dtype='float')
    Xrb = np.zeros([1, dim], dtype='float')
    fitR = float('inf')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
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
        for i in range(N):
            n = 0.05 * np.exp(-2 * (t / max_iter) ** 2)
            r = np.random.rand()
            if 0.9 < r:
                XPosNew = X[i] + n * (1 + np.sin(r)) * X[i]
            else:
                XPosNew = X[i] * (n * (2 * r - 1) + 1)
            # 在此处应用eq8的改进
            k = 2 * r
            XPosNew = (ub + lb)*(0.5 + 1/2*k) - XPosNew/k
            
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