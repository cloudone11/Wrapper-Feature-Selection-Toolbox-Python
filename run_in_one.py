import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, matthews_corrcoef
import importlib
import json
from multiprocessing import Pool
from tqdm import tqdm
import time

# Parameters
k = 5  # k-value in KNN
N = 10  # Number of particles
T = 100  # Maximum number of iterations
opts = {'k': k, 'N': N, 'T': T}

# Define algorithms to run
# algorithms = ["ibka1h", "ibka2h", "ibka3h", "woa", "ja", "pso", "sca", "ssa", "gwo", "bka"]
algorithms = ['gwo','gwoh','gwol','gwos']
# algorithms = [ "woa", "ja", "pso", "sca", "ssa", "gwo", "bka"]
# Function to run the algorithm and collect metrics
def run_algorithm(algo, train_index, test_index, feat, label):
    # Dynamically import the module
    module_name = f"FS.{algo}"
    module = importlib.import_module(module_name)
    jfs = getattr(module, "jfs")
    
    # Split data into train and test based on current fold
    xtrain, xtest = feat[train_index], feat[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}
    opts['fold'] = fold

    # Perform feature selection
    fmdl = jfs(feat, label, opts)
    sf = fmdl['sf']

    # Model with selected features
    x_train = xtrain[:, sf]
    y_train = ytrain
    x_valid = xtest[:, sf]
    y_valid = ytest

    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(x_train, y_train)

    # Calculate metrics
    y_pred = mdl.predict(x_valid)
    Acc = accuracy_score(y_valid, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    sensitivity = recall_score(y_valid, y_pred, pos_label=0)
    precision = precision_score(y_valid, y_pred, pos_label=0)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_valid, y_pred)

    num_feat = fmdl['nf']
    curve = fmdl['c'].tolist()  # Convert numpy array to list for JSON serialization

    return {
        "Algorithm": algo,
        "Fold": fold,
        "Accuracy": Acc,
        "Sensitivity": sensitivity,
        "Precision": precision,
        "Specificity": specificity,
        "MCC": mcc,
        "Number of Features": num_feat,
        "selected_features": sf.tolist(),
        "Convergence Curve": curve
    }

def worker(i):
    print(f"worker {i} started")
    if i == 0:
        data = pd.read_csv(r'./data/merged_df12.csv')
    elif i == 1:
        data = pd.read_csv(r'./data/merged_df13.csv')
    elif i == 2:
        data = pd.read_csv(r'./data/merged_df23.csv')
    else:
        print("Invalid input")
        return

    data = data.dropna()  # 删除包含缺失值的行
    data = data.values
    feat = np.asarray(data[:, 0:-1])
    label = np.asarray(data[:, -1])
    
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
    # 归一化
    feat = -1 + 2 * (feat - min_feat) / range_feat

    # 此时 feat 已经归一化到 [-1, 1]，可以继续后续处理
    
    k = 5  # k-value in KNN
    N = 10  # Number of particles
    T = 100  # Maximum number of iterations
    opts = {'k': k, 'N': N, 'T': T}
    results = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation
    for algo in algorithms:
        print(f"Running experiment with algorithm: {algo}")
        fold_results = []
        for j in tqdm(range(10), desc=f"Algorithm: {algo}", unit="run"):
            for fold, (train_index, test_index) in enumerate(kf.split(feat)):
                result = run_algorithm(algo, train_index, test_index, feat, label)
                result['Fold'] = fold + 1  # Add fold number to result
                result['Run'] = j + 1
                fold_results.append(result)
        results.extend(fold_results)

    # Save results to JSON file
    with open(f"experiment_results_{i}.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Experiment results saved to experiment_results_{i}.json")
if __name__ == '__main__':
    # with Pool(processes=3) as pool:
    #     pool.map(worker, [0])
    for i in range(3):
        worker(i)