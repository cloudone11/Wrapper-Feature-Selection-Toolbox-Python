import numpy as np

# 假设 arr 是一个形状为 (136,) 的一维数组
arr = np.arange(136)  # 示例数据

# 将一维数组转换为形状为 (1, 136) 的二维数组
arr_2d = arr.reshape(1, -1)  # -1 表示自动计算行数

print(arr_2d.shape)  # 输出: (1, 136)