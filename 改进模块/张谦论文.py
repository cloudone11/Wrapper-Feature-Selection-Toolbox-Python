import numpy as np

# 假设的适应度函数，需根据实际问题定义
def fitness(individual):
    return np.sum(individual**2)

# 步骤3: 扩散过程
def diffuse_population(population, BP, g):
    N, dim = population.shape
    new_population = np.zeros_like(population)
    for i in range(N):
        P_i = population[i]
        sigma = np.abs((np.log(g) / g) * (P_i - BP))
        # 使用公式(18)生成新个体（可切换为公式19）
        epsilon = np.random.uniform(0, 1)
        epsilon_prime = np.random.uniform(0, 1)
        GW = np.random.normal(loc=BP, scale=sigma) + epsilon * BP - epsilon_prime * P_i
        new_population[i] = GW
    return new_population

# 步骤4: 第一次更新操作
def update_population(population):
    N, dim = population.shape
    # 根据适应度排序
    fitness_values = np.array([fitness(ind) for ind in population])
    sorted_indices = np.argsort(fitness_values)
    sorted_pop = population[sorted_indices].copy()
    
    for i in range(N):
        Pa_i = (i + 1) / N  # Rank从1开始计算
        epsilon = np.random.uniform(0, 1)
        if Pa_i < epsilon:
            # 随机选择参考个体
            r = np.random.randint(N)
            P_r = sorted_pop[r]
            # 更新所有维度
            sorted_pop[i] = P_r - epsilon * (P_r - sorted_pop[i])
    return sorted_pop

# 示例用法
N = 50  # 种群大小
dim = 10  # 个体维度
max_g = 100  # 最大评估次数
population = np.random.rand(N, dim)  # 初始化种群

for g in range(1, max_g + 1):
    # 计算当前最优个体BP
    fitness_values = np.array([fitness(ind) for ind in population])
    BP = population[np.argmin(fitness_values)]
    
    # 执行扩散过程（步骤3）
    population = diffuse_population(population, BP, g)
    
    # 执行第一次更新操作（步骤4）
    population = update_population(population)
    
    # 更新最优个体（此处可能需要重新计算适应度）
    fitness_values = np.array([fitness(ind) for ind in population])
    BP = population[np.argmin(fitness_values)]
    
    # 输出最优值
    print(f'Generation {g}: Best fitness = {fitness(BP)}')