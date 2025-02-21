#[2014]-"Grey wolf optimizer" Modified for Multi-Objective Feature Selection
import numpy as np
from numpy.random import rand
from FS.functionHO import Fun  # 需修改为多目标评估函数即返回向量。
def select_leader(archive):
    # 基于网格概率选择
    grid_density = calculate_grid_density(archive)
    probabilities = 1 / (grid_density + 1e-6)
    return np.random.choice(archive, p=probabilities/probabilities.sum())

# 收敛参数a的线性递减
a = 2 - iter * (2 / max_iter)
# 自适应网格参数
alpha = 0.1 * (iter / max_iter)


class Solution:
    def __init__(self):
        self.Position = []
        self.Cost = []
        self.Dominated = False
        self.GridIndex = []
        self.GridSubIndex = []

def init_position(lb, ub, N, dim):
    return np.random.randint(low=lb, high=ub+1, size=(N, dim))

def binary_conversion(X, thres=0.5):
    return (X > thres).astype(int)

def dominates(solution1, solution2):
    # 判断solution1是否支配solution2
    return (all(s1 <= s2 for s1, s2 in zip(solution1.Cost, solution2.Cost))) and (any(s1 < s2 for s1, s2 in zip(solution1.Cost, solution2.Cost)))

def determine_domination(population):
    for sol in population:
        sol.Dominated = False
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j and dominates(population[i], population[j]):
                population[j].Dominated = True
    return population

def get_non_dominated(population):
    return [sol for sol in population if not sol.Dominated]

def create_hypercube(costs, n_grid=10, inflation=0.1):
    # 创建多目标网格
    mins = np.min(costs, axis=0)
    maxs = np.max(costs, axis=0)
    ranges = maxs - mins
    return [np.linspace(mins[i]-inflation*ranges[i], 
                       maxs[i]+inflation*ranges[i], 
                       n_grid+1) for i in range(2)]

def grid_index(solution, hypercube):
    # 计算解的网格索引
    return [np.searchsorted(hypercube[i], solution.Cost[i], side='right') 
           for i in range(2)]

def mojfs(xtrain, ytrain, opts):
    # 多目标参数
    n_pop = opts['N']          # 种群数量
    max_iter = opts['T']       # 最大迭代
    archive_size = 100         # 存档大小
    n_grid = 10                # 网格数量
    alpha = 0.1                # 网格膨胀系数
    beta = 1                   # 选择压力
    
    dim = xtrain.shape[1]      # 特征维度
    lb, ub = 0, 1              # 特征选择范围
    
    # 初始化种群
    population = [Solution() for _ in range(n_pop)]
    for sol in population:
        sol.Position = rand(dim)
        sol.Cost = [0.0, 0.0]  # 双目标：[分类误差, 特征数量]
    
    # 初始化存档
    archive = []
    
    # 主循环
    for iter in range(max_iter):
        # 评估目标函数（需实现多目标Fun函数）
        for sol in population:
            Xbin = binary_conversion(sol.Position)
            error_rate = Fun(xtrain, ytrain, Xbin, opts)  # 目标1：分类误差
            n_features = np.sum(Xbin)                     # 目标2：特征数量
            sol.Cost = [error_rate, n_features]
        
        # 合并种群和存档并去重
        combined = population + archive
        combined = list({tuple(sol.Position): sol for sol in combined}.values())
        
        # 非支配排序
        determine_domination(combined)
        non_dominated = get_non_dominated(combined)
        
        # 更新存档
        archive = non_dominated.copy()
        if len(archive) > archive_size:
            # 网格选择机制
            hypercube = create_hypercube([sol.Cost for sol in archive], n_grid, alpha)
            for sol in archive:
                sol.GridIndex = grid_index(sol, hypercube)
            # 根据网格密度裁剪存档
            while len(archive) > archive_size:
                # 移除最密集网格中的解
                pass
        
        # 灰狼更新机制
        a = 2 - iter * (2 / max_iter)
        for sol in population:
            # 从存档中选择领导者
            leaders = np.random.choice(archive, size=3, replace=False)
            # 位置更新
            new_position = np.mean([leaders[i].Position * 
                                  (a * (2*rand() - 1)) for i in range(3)], axis=0)
            sol.Position = np.clip(new_position, lb, ub)
    
    # 提取帕累托前沿
    pareto_front = sorted(archive, key=lambda x: x.Cost[0])
    
    # 返回结果
    return {
        'pareto_front': [{'features': np.where(sol.Position > 0.5)[0],
                         'error': sol.Cost[0],
                         'n_features': sol.Cost[1]} 
                        for sol in pareto_front],
        'convergence': []
    }