import numpy as np

# ----------------------
# 转换函数实现
# ----------------------
def transfer_function(x):
    """公式(25): 转换函数Tr(x)"""
    return np.abs(2 / np.pi * np.arctan(np.pi / 2 * x))

def binary_conversion(individual, rand_values):
    """公式(24): 连续值转二进制"""
    tx = np.zeros_like(individual)
    # 前两个维度保留为k和m（不进行转换）
    for j in range(2, len(individual)):
        tr = transfer_function(individual[j])
        if rand_values[j] < tr:
            tx[j] = 0
        else:
            tx[j] = 1
    return tx

# ----------------------
# 适应度函数实现
# ----------------------
def fitness_function(error, selected_feature_ratio, w=0.95):
    """公式(26): 适应度计算"""
    return w * error + (1 - w) * selected_feature_ratio

# ----------------------
# 示例用法
# ----------------------
if __name__ == "__main__":
    # 假设一个个体（前两维是k和m，后续为特征）
    individual = np.array([5, 1.5, 0.3, -0.7, 2.1, 0.9])
    rand_values = np.random.rand(len(individual))  # 随机数生成
    
    # 二进制转换（仅作用于第三维及之后）
    tx_individual = binary_conversion(individual, rand_values)
    print("转换后的个体:", tx_individual)
    
    # 计算适应度（假设分类错误率0.1，选2/5特征）
    error = 0.05
    selected_ratio = 51/170
    fitness = fitness_function(error, selected_ratio)
    print("适应度值:", fitness)