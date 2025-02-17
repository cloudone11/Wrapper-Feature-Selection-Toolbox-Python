import numpy as np
import matplotlib.pyplot as plt

# 适应度函数（Sphere函数）
def fitness_func(x):
    return np.sum(x**2) + 3

# 原始灰狼优化算法
def original_gwo(pop_size, max_iter, dim, lb, ub):
    # 初始化种群
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([fitness_func(x) for x in pop])
    
    # 记录最优解
    best_scores = []
    
    for iter in range(max_iter):
        # ========== 更新头狼位置 ==========
        sorted_indices = np.argsort(fitness)
        alpha_pos = pop[sorted_indices[0]].copy()
        beta_pos = pop[sorted_indices[1]].copy()
        delta_pos = pop[sorted_indices[2]].copy()
        best_scores.append(fitness[sorted_indices[0]])
        # ========== 位置更新 ==========
        a_parameter = 2 - iter * (2 / max_iter)  # 线性递减
        
        for i in range(pop_size):
            for j in range(dim):
                # 更新Alpha引导
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a_parameter * r1 - a_parameter
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - pop[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # 更新Beta引导
                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a_parameter * r1 - a_parameter
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - pop[i][j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # 更新Delta引导
                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a_parameter * r1 - a_parameter
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - pop[i][j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # 综合三个位置
                pop[i][j] = (X1 + X2 + X3) / 3
        
        # 边界处理
        pop = np.clip(pop, lb, ub)
        
        # 更新适应度
        fitness = np.array([fitness_func(x) for x in pop])
    
    return best_scores

# 改进的灰狼优化算法
def improved_gwo(pop_size, max_iter, dim, lb, ub, max_reverse_ratio=0.3):
    # 初始化种群
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([fitness_func(x) for x in pop])
    
    # 初始化历史最大方差
    var_total = np.sum(np.var(pop, axis=0))
    max_var = var_total
    
    # 记录最优解
    best_scores = []
    
    for iter in range(max_iter):
        # ========== 反向学习策略 ==========
        # 计算当前种群方差
        var_current = np.sum(np.var(pop, axis=0))
        max_var = max(max_var, var_current)
        var_norm = var_current / max_var
        
        # 计算反向个体数量
        reverse_num = int(var_norm * max_reverse_ratio * pop_size)
        reverse_num = min(reverse_num, int(max_reverse_ratio * pop_size))
        
        # 生成反向个体
        if reverse_num > 0:
            selected_indices = np.random.choice(pop_size, size=reverse_num, replace=False)
            
            # 计算当前排名
            sorted_indices = np.argsort(fitness)
            ranks = np.zeros(pop_size, dtype=int)
            for rank, idx in enumerate(sorted_indices, 1):
                ranks[idx] = rank
            
            for idx in selected_indices:
                original = pop[idx].copy()
                # reverse_individual = lb + ub - original  # 生成反向解
                # 生成部分反向解。在特征空间中随机选择一部分特征进行反向
                reverse_indices = np.random.choice(dim, size=int(0.5 * dim), replace=False)
                reverse_individual = pop[idx].copy()
                reverse_individual[reverse_indices] = lb + ub - reverse_individual[reverse_indices]
                
                f_reverse = fitness_func(reverse_individual)
                f_original = fitness[idx]
                
                # 计算适应度比值
                if f_original == 0:
                    k = 0
                else:
                    k = f_reverse / f_original
                
                # 计算Logistic函数值
                logistic_k = 1 / (1 + np.exp(-k))
                
                # 计算替换概率
                a = (logistic_k * ranks[idx]) / pop_size
                
                # 决定是否替换
                if a > np.random.uniform():
                    print(f"Reverse individual at index {idx}")
                    pop[idx] = reverse_individual
                    fitness[idx] = f_reverse
        
        # ========== 更新头狼位置 ==========
        sorted_indices = np.argsort(fitness)
        alpha_pos = pop[sorted_indices[0]].copy()
        beta_pos = pop[sorted_indices[1]].copy()
        delta_pos = pop[sorted_indices[2]].copy()
        best_scores.append(fitness[sorted_indices[0]])
        
        # ========== 位置更新 ==========
        a_parameter = 2 - iter * (2 / max_iter)  # 线性递减
        
        for i in range(pop_size):
            for j in range(dim):
                # 更新Alpha引导
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a_parameter * r1 - a_parameter
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - pop[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # 更新Beta引导
                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a_parameter * r1 - a_parameter
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - pop[i][j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # 更新Delta引导
                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a_parameter * r1 - a_parameter
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - pop[i][j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # 综合三个位置
                pop[i][j] = (X1 + X2 + X3) / 3
        
        # 边界处理
        pop = np.clip(pop, lb, ub)
        
        # 更新适应度
        fitness = np.array([fitness_func(x) for x in pop])
    
    return best_scores

# 参数设置
pop_size = 30
max_iter = 100
dim = 10
lb = -10
ub = 10

# 运行原始GWO
original_scores = original_gwo(pop_size, max_iter, dim, lb, ub)

# 运行改进的GWO
improved_scores = improved_gwo(pop_size, max_iter, dim, lb, ub)
print("Original GWO best score:", original_scores[-1])
print("Improved GWO best score:", improved_scores[-1])
# 绘制对比图
plt.plot(original_scores, label='Original GWO')
plt.plot(improved_scores, label='Improved GWO')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness Score')
plt.legend()
plt.show()