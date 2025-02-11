import pandas as pd

# 假设三个CSV文件的路径分别为file1.csv、file2.csv和file3.csv
file1_path = 'AD.csv'
file2_path = 'MCI.csv'
file3_path = 'CN.csv'

# 读取CSV文件，跳过前两行（因为前两行是索引和标题）
df1 = pd.read_csv(file1_path, skiprows=1, header=None)
df2 = pd.read_csv(file2_path, skiprows=1, header=None)
df3 = pd.read_csv(file3_path, skiprows=1, header=None)

# 为合并后的数据集添加标签列
# 假设类别1对应标签'g'，类别2对应标签'h'，类别3对应标签'i'
df1['label'] = 'AD'
df2['label'] = 'MCI'
df3['label'] = 'CN'

# 两两合并数据集
# 合并类别1和类别2
merged_df12 = pd.concat([df1, df2], ignore_index=True)
# 合并类别1和类别3
merged_df13 = pd.concat([df1, df3], ignore_index=True)
# 合并类别2和类别3
merged_df23 = pd.concat([df2, df3], ignore_index=True)

# 保存合并后的数据集到新的CSV文件
merged_df12.to_csv('merged_df12.csv', index=False)
merged_df13.to_csv('merged_df13.csv', index=False)
merged_df23.to_csv('merged_df23.csv', index=False)

print("合并完成，生成了三个新的CSV文件：merged_df12.csv, merged_df13.csv, merged_df23.csv")