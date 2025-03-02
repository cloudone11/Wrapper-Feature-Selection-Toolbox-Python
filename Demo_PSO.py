import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from FS.gwosca import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from utils.mClassifier import knn_classifier_for_static_data, np_knn_classifier_for_static_data
import pandas as pd
import numpy as np
import time
# Load data
data = pd.read_csv(r'./data/merged_df12.csv')

# 删除包含缺失值的行
data = data.dropna()

# 将数据转换为 NumPy 数组
data = data.values

# 分离特征和标签
feat = np.asarray(data[:, :-1],dtype='float')  # 特征
label = np.asarray(data[:, -1])  # 标签
# 统计标签类别并转为整数
types, counts = np.unique(label, return_counts=True)
label = np.searchsorted(types, label)

# 标签转为 one-hot 编码
label = np.eye(len(types))[label]

# 特征归一化
label = np.argmax(label, axis=1)

# 归一化特征到 [-1, 1]
min_feat = np.min(feat, axis=0)  # 每列的最小值
max_feat = np.max(feat, axis=0)  # 每列的最大值

# 检查是否存在 max_feat == min_feat 的情况
# 如果存在，将这些列的值设置为 0（或其他默认值）
range_feat = max_feat - min_feat
range_feat[range_feat == 0] = 1  # 避免除以零
# 输出为零的列

feat = -1 + 2 * (feat - min_feat) / range_feat

# 此时 feat 已经归一化到 [-1, 1]，可以继续后续处理


# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.1, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 30    # number of particles
T    = 100   # maximum number of iterations
dim  = np.size(xtrain,1) # the dim
alpha= 0.90
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'alpha':alpha,'fold':fold}
start_time = time.time()

# initial static classifier
staticClassifier = np_knn_classifier_for_static_data(xtrain,ytrain,xtest,ytest,k)
# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']
print(sf)
# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# confusion matrix
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
# convert label of y_valid and y_pred to 'b' or 'g'
selected_features = fmdl['sf']
sensitivity = recall_score(y_valid, y_pred, pos_label=0)
precision = precision_score(y_valid, y_pred, pos_label=0)
specificity = tn / (tn + fp)
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
print("selected features:", selected_features)
print("Sensitivity:", sensitivity)
print("Precision:", precision)
print("Specificity:", specificity)
print("MCC:", mcc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)
end_time = time.time()
print("Time elapsed:", end_time - start_time, "seconds")
# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSO')
ax.grid()
plt.show()

