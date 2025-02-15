import numpy as np
import faiss
# from sklearn.neighbors import KNeighborsClassifier

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.index = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # 确保标签为整数
        self.y_train = y_train.astype(np.int64)  # 关键修改点
        # 转换数据为 float32 格式
        X_train = np.ascontiguousarray(X_train.astype(np.float32))
        
        # 创建索引（使用 L2 欧式距离）
        dimension = X_train.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(X_train)

    def predict(self, X_test):
        # 转换数据为 float32 格式
        X_test = np.ascontiguousarray(X_test.astype(np.float32))
        
        # 搜索最近的 k 个邻居
        distances, indices = self.index.search(X_test, self.n_neighbors)
        
        # 获取邻居的标签
        neighbor_labels = self.y_train[indices]
        
        # 多数投票决定预测结果（处理可能的空输入）
        predictions = []
        for row in neighbor_labels:
            if len(row) == 0:
                # 若邻居为空，随机预测或返回默认值
                predictions.append(0)
            else:
                counts = np.bincount(row)
                predictions.append(np.argmax(counts))
        return np.array(predictions)

# Error rate 函数修改
def error_rate(xtrain, ytrain, x, opts):
    # 参数解析
    k = opts['k']
    fold = opts['fold']
    xt = fold['xt']
    yt = fold['yt'].astype(np.int64)  # 确保标签为整数
    xv = fold['xv']
    yv = fold['yv'].astype(np.int64)  # 确保标签为整数
    
    # 特征选择
    selected = x == 1
    if np.sum(selected) == 0:
        return 1.0  # 若无特征选中，直接返回最大错误率
    
    xtrain = xt[:, selected]
    xvalid = xv[:, selected]
    
    # 训练和预测
    try:
        mdl = KNeighborsClassifier(n_neighbors=k)
        mdl.fit(xtrain, yt)
        ypred = mdl.predict(xvalid)
        acc = np.sum(yv == ypred) / len(yv)
        return 1 - acc
    except:
        return 1.0  # 异常时返回最大错误率

# Fun 函数微调
def Fun(xtrain, ytrain, x, opts):
    alpha = 0.90
    beta = 1 - alpha
    max_feat = len(x)
    num_feat = np.sum(x == 1)
    
    if num_feat == 0:
        return 1.0  # 直接返回最大成本
    else:
        error = error_rate(xtrain, ytrain, x, opts)
        return alpha * error + beta * (num_feat / max_feat)