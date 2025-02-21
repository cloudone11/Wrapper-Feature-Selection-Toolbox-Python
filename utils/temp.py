import numpy as np

# 假设 Alpha_pos 是一个形状为 (136,) 的一维数组
Alpha_pos = np.random.rand(136)  # 示例数据
print(Alpha_pos.shape)  # 输出: (136,)
# 假设 dim 是你希望得到的二维数组的列数
dim = 4  # 例如，如果你有4个维度

# 将一维数组转换为二维数组
Alpha_pos_2d = Alpha_pos.reshape(-1, 136)  # 注意这里 -1 表示自动计算行数

# 现在 Alpha_pos_2d 是一个形状为 (1, 4) 的二维数组
print(Alpha_pos_2d.shape)  # 输出: (1, 4)
print(Alpha_pos_2d[0,0])  # 输出: [[0. 0. 0. 0.]]