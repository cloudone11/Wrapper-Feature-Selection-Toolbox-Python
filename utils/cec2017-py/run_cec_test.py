import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO
import cec2017.functions as functions

# 优化参数设置
dimension = 10        # 问题维度（根据CEC2017标准测试维度）
pop_size = 50         # 种群大小
epoch = 100           # 迭代次数
lb = [-100] * dimension  # 变量下界
ub = [100] * dimension   # 变量上界

# 遍历所有CEC2017测试函数
for func in functions.all_functions:
    # 使用闭包捕获当前函数对象
    def make_fit_func(f):
        def fit_func(solution):
            return f(solution)
        return fit_func
    
    # 创建优化问题
    problem = {
        "fit_func": func,
        "lb": lb,
        "ub": ub,
        "minmax": "min",    # 最小化问题
        "name": func.__name__
    }
    
    # 初始化PSO算法
    model = OriginalPSO(epoch=epoch, pop_size=pop_size)
    
    # 执行优化
    best_position, best_fitness = model.solve(problem)
    
    # 输出结果
    print(f"\nFunction: {func.__name__}")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Best solution (first 5 dim): {np.round(best_position[:5], 4)}...")
    print("-" * 60)

# 对于更高维问题（例如30维），可调整dimension参数：
# dimension = 30
# 注意：更高维度需要更大的种群和更多迭代次数