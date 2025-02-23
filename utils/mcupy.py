# import cupy as cp
# import numpy as np
# import time

# # 创建一个非常大的数组
# data = np.random.rand(10000, 10000)  # 使用 NumPy 创建数组

# # 使用 CPU 计算（NumPy）
# start_time = time.time()
# result_cpu = np.dot(data, data)
# print("CPU 计算时间:", time.time() - start_time)

# # 使用 GPU 计算（CuPy）
# data_gpu = cp.array(data)  # 将数据转移到 GPU
# start_time = time.time()
# result_gpu = cp.dot(data_gpu, data_gpu)
# print("GPU 计算时间:", time.time() - start_time)

import cupy as cp
from bitarray import bitarray

class MatrixMultiplier:
    def __init__(self, matrix):
        """
        初始化类，将矩阵存入 GPU。
        :param matrix: 一个二维列表或 NumPy 数组，表示初始矩阵。
        """
        # 将输入的矩阵转换为 CuPy 数组并存入 GPU
        self.gpu_matrix = cp.array(matrix)

    def multiply_with_bitarrays(self, bitarrays):
        """
        将 bitarray 列表转换为矩阵，并与初始化的矩阵进行矩阵乘法。
        :param bitarrays: 一个包含 bitarray 的列表。
        :return: 结果矩阵（CuPy 数组）。
        """
        # 将 bitarray 列表转换为二进制矩阵
        binary_matrix = [list(map(int, bitarr)) for bitarr in bitarrays]
        
        # 将二进制矩阵转换为 CuPy 数组并存入 GPU
        gpu_bitarray_matrix = cp.array(binary_matrix, dtype=cp.float32)
        
        # 进行矩阵乘法
        result_matrix = cp.dot(self.gpu_matrix, gpu_bitarray_matrix.T)  # .T 表示转置
        
        # 输出结果矩阵
        print("结果矩阵:")
        print(result_matrix)
        
        return result_matrix

# 测试代码
if __name__ == "__main__":
    # 初始化一个矩阵
    initial_matrix = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    # 创建类的实例
    multiplier = MatrixMultiplier(initial_matrix)

    # 创建一些 bitarray
    ba1 = bitarray('110')
    ba2 = bitarray('101')
    ba3 = bitarray('011')

    # 调用方法进行矩阵乘法
    result = multiplier.multiply_with_bitarrays([ba1, ba2, ba3])
    
import cupy as cp

# 示例矩阵
matrix = cp.array([
    [4, 2, 9, 1],
    [1, 5, 3, 7],
    [7, 6, 8, 2],
    [3, 4, 5, 6],
    [2, 1, 4, 3],
    [5, 7, 6, 8]
], dtype=cp.float32)

n1 = 3  # 每个块的行数
n2 = 2  # 块的数量
dim = 4  # 每个块的列数
k = 2  # 每列取 k 个最小值的行号

# 将矩阵分块
blocks = [matrix[i * n1:(i + 1) * n1, :dim] for i in range(n2)]

# 对每个块逐列求 k 个最小值的行号
result = []
for block in blocks:
    sorted_indices = cp.argsort(block, axis=0)  # 对每列排序，返回行号
    k_min_indices = sorted_indices[:k, :]       # 取前 k 个最小值的行号
    result.append(k_min_indices)

print("原始矩阵：")
print(matrix)
print("\n分块后的矩阵：")
for i, block in enumerate(blocks):
    print(f"块 {i + 1}:")
    print(block)
print("\n每块每列前 k 个最小值的行号：")
for i, indices in enumerate(result):
    print(f"块 {i + 1}:")
    print(indices)