import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder
import importlib
import json
from multiprocessing import Pool
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import traceback

# # Define algorithms to run
algorithms1 = []
# algorithms2 = ['gwo','gwos','gwo8s','gwo8h','gwo8l','gwo12s','gwo12h','gwo12l']
# algorithms3  = ["woa", "ja", "pso", "sca", "ssa", "gwo", "bka",'ba','bka','cs','de','fa','fpa','ga']
algorithms4 = ['gwo1','gwo3','gwo4','gwo6','gwo7','gwo8','gwo9','gwo10','gwo11','gwo12','gwo13','gwo14','gwo16','gwo17','gwosca','sogwo']
# algorithms5 = ['ala']
algorithms = set()
algorithmsall = algorithms1+algorithms4
for alg in algorithmsall:
    algorithms.add(alg)
# algorithms = ['ala']
print(algorithms)
# Function to run the algorithm and collect metrics
def run_algorithm(algo, train_index, test_index, feat, label,opts):
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
    if 'runcec' in opts and opts['runcec'] ==True:
        return {
            "Algorithm": algo,
            "Accuracy": fmdl['c'].tolist()[0][-1],
            "Convergence Curve": fmdl['c'].tolist()
        }
    else:
        sf = fmdl['sf']

        # Model with selected features
        x_train = xtrain[:, sf]
        y_train = ytrain
        x_valid = xtest[:, sf]
        y_valid = ytest
        k       = opts['k']
        mdl = KNeighborsClassifier(n_neighbors=k)
        mdl.fit(x_train, y_train)
        
        # Calculate metrics
        y_pred = mdl.predict(x_valid)
        cm = confusion_matrix(y_valid, y_pred)
        num_classes = len(np.unique(y_valid))
        
        # Per-class metrics
        metrics = {}
        for i in range(num_classes):
            metrics[f"{i}_precision"] = precision_score(y_valid, y_pred, average=None)[i]
            metrics[f"{i}_recall"] = recall_score(y_valid, y_pred, average=None)[i]
            metrics[f"{i}_f1"] = f1_score(y_valid, y_pred, average=None)[i]
        
        # Overall metrics
        metrics["accuracy"] = accuracy_score(y_valid, y_pred)
        metrics["macro_precision"] = precision_score(y_valid, y_pred, average='macro')
        metrics["macro_recall"] = recall_score(y_valid, y_pred, average='macro')
        metrics["macro_f1"] = f1_score(y_valid, y_pred, average='macro')
        metrics["mcc"] = matthews_corrcoef(y_valid, y_pred)

        num_feat = fmdl['nf']
        curve = fmdl['c'].tolist()  # Convert numpy array to list for JSON serialization

        result = {
            "Algorithm": algo,
            "Fold": fold,
            "Confusion_Matrix": cm.tolist(),
            "Number of Features": num_feat,
            "selected_features": sf.tolist(),
            "Convergence Curve": curve
        }
        # Add all metrics to result
        result.update(metrics)
        
        return result

def worker(i, pre_feature_selection_algorithm='none', feature_drop_rate = 0 , w = 0.7, check_cec = False):
    if check_cec:
        opts       = {}
        opts['lb'] = -100
        opts['ub'] = 100
        opts['N']  = 30
        opts['T']  = 1000
        opts['dim']= 100
        opts['runcec'] = True
        opts['selectedFunIndex'] = 0
        totalRun   = 30
        train_index_placeH = [0,1]
        test_index_placeH  = [0,1]
        feat = np.zeros((2,opts['dim']),dtype='float')
        label= np.zeros((2,1),dtype='float')
        results = []
        for algo in algorithms:
            for j in tqdm(range(1), desc=f"Algorithm: {algo}", unit="run"):
                result = run_algorithm(algo, train_index_placeH, test_index_placeH, feat, label,opts)
                results.append(result)
        with open(f"{opts['selectedFunIndex']}_{opts['N']}_{opts['T']}_{opts['dim']}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
    else:
        print(f"worker {i} started")
        if i == 0:
            data = pd.read_csv(r'./data/merged_df12.csv')
        elif i == 1:
            data = pd.read_csv(r'./data/merged_df13.csv')
        elif i == 2:
            data = pd.read_csv(r'./data/merged_df23.csv')
        elif i == 3:
            data = pd.read_csv(r'./data/merged_df123.csv')
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
        print("Columns with zero range:", np.where(range_feat == 0)[0])
        if pre_feature_selection_algorithm == 'chi2':
            # 使用卡方检验进行特征选择
            chi2_selector = SelectKBest(score_func=chi2, k='all')
            chi2_selector.fit(feat + 1, label)
            chi2_scores = chi2_selector.scores_
            # k = (1 - feature_drop_rate) * len(chi2_scores)
            k = int((1 - feature_drop_rate) * len(chi2_scores))
            # 如果 X 是 numpy 数组，先将其转换为 DataFrame
            if isinstance(feat, np.ndarray):
                X = pd.DataFrame(feat)
            chi2_top_k_features = X.columns[chi2_selector.get_support(indices=True)[:k]]
            feat = feat[:, chi2_top_k_features]
            # 打印feature_drop_rate*len(chi2_scores)个最低特征的评分
            
        elif pre_feature_selection_algorithm == 'mi':
            # 使用信息增益进行特征选择
            mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
            mi_selector.fit(feat, label)
            mi_scores = mi_selector.scores_
            k = int((1 - feature_drop_rate) * len(mi_scores))
            mi_top_k_features = np.argsort(mi_scores)[::-1][:k]        
            feat = feat[:, mi_top_k_features]
        elif pre_feature_selection_algorithm == 'none':
            pass
        else:
            print("Invalid input")
        k = 5  # k-value in KNN
        N = 30  # Number of particless
        T = 50  # Maximum number of iterations
        opts = {'k': k, 'N': N, 'T': T ,'w':w}
        results = []
        accuracy_scores = []
        num_feat = []
        kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation
        for algo in algorithms:
            print(f"Running experiment with algorithm: {algo}")
            fold_results = []
            for j in tqdm(range(30), desc=f"Algorithm: {algo}", unit="run"):
                # set the static knn classifier
                knnClassifierList = []
                for fold, (train_index, test_index) in enumerate(kf.split(feat)):
                    pass
                for fold, (train_index, test_index) in enumerate(kf.split(feat)):
                    # if fold != 0:
                    #     continue
                    # try - catch block to handle exceptions
                    try:
                        stime   = time.time()
                        result = run_algorithm(algo, train_index, test_index, feat, label,opts)
                        etime   = time.time()
                        result['Fold'] = fold + 1  # Add fold number to result
                        result['Run'] = j + 1
                        result['pre_feature_selection_algorithm'] = pre_feature_selection_algorithm
                        result['feature_drop_rate']               = feature_drop_rate
                        result['dataPath']                        = i
                        result['time'] = etime-stime
                        fold_results.append(result)
                        num_feat.append(result['Number of Features'])
                        accuracy_scores.append(result['accuracy'])
                    except Exception as e:
                        print(f"Error in fold {fold + 1} of run {j + 1} with algorithm {algo}: {e}")
                        traceback.print_exc()  # 打印完整的错误栈跟踪信息              
            results.extend(fold_results)

        # Save results to JSON file
        with open(f"{i}_{N}_{T}_{pre_feature_selection_algorithm}_{feature_drop_rate}_{w}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Experiment results saved to experiment_results_{i}_{pre_feature_selection_algorithm}_{feature_drop_rate}.json")
        print(f"{np.mean(num_feat):.4f} features on average were selected")
        print(f"{np.mean(accuracy_scores):.4f} accuracy on average was achieved")
    
if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    # with Pool(processes=3) as pool:
    #     pool.map(worker, [0])
    # for i in range(3):
    #     worker(i)
    
    
    # worker(0, 'mi', 0.5)
    # worker(0, 'none', 0.5)
    # worker(0, 'chi2', 0.5)
    # worker(0, 'mi', 0.3)
    # # worker(0, 'none', 0.3)
    # worker(0, 'chi2', 0.3)
    # worker(0, 'mi', 0.1)
    # # worker(0, 'none', 0.1)
    # worker(0, 'chi2', 0.1)
    # File Name	Average Accuracy	Average Number of Features
    # experiment_results_0_mi_0.5.json	0.9215	20.7600
    # experiment_results_0_mi_0.3.json	0.9146	27.6700
    # experiment_results_0_mi_0.1.json	0.9187	35.2300
    # experiment_results_0_chi2_0.5.json	0.8926	19.7800
    # experiment_results_0_chi2_0.3.json	0.9054	-
    # worker(0, 'mi', 0.2)
    # worker(0, 'chi2', 0.2)
    
    # worker(0, 'none', 0.2,0.4)
    # worker(0,'none',0.2,0.3)
    worker(3,'none',0.2,0.4,True)
    # worker(1, 'none', 0.2)
    # worker(2, 'none', 0.2)
    
    # worker(0, 'mi', 0.2)
    # worker(1, 'mi', 0.2)
    # worker(2, 'mi', 0.2)
    
    # worker(0, 'chi2', 0.2)
    # worker(1, 'chi2', 0.2)
    # worker(2, 'chi2', 0.2)
    
    # Worker	File Name	Features Selected (avg)	Accuracy (avg)	Time per Run (s)
    # 0	experiment_results_0_none_0.2.json	39.9000	0.9099	23.34
    # 1	experiment_results_1_none_0.2.json	28.0200	0.9623	23.01
    # 2	experiment_results_2_none_0.2.json	36.2700	0.9404	24.30
    # 0	experiment_results_0_mi_0.2.json	31.7200	0.9185	19.49
    # 1	experiment_results_1_mi_0.2.json	23.4000	0.9610	19.00
    # 2	experiment_results_2_mi_0.2.json	28.7200	0.9473	19.99
    # 0	experiment_results_0_chi2_0.2.json	31.0100	0.9198	19.48
    # 1	experiment_results_1_chi2_0.2.json	23.8800	0.9613	19.05
    
    # worker(0, 'mi', 0.2)
