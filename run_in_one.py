import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, matthews_corrcoef
import importlib
import json
from multiprocessing import Pool
from tqdm import tqdm
# Load data
data = pd.read_csv('merged_df12.csv')
data = data.dropna()  # 删除包含缺失值的行
data = data.values
feat = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

# Parameters
k = 5  # k-value in KNN
N = 10  # Number of particles
T = 100  # Maximum number of iterations
opts = {'k': k, 'N': N, 'T': T}

# Define algorithms to run
algorithms = ["ibka1h", "ibka2h", "ibka3h", "woa", "ja", "pao", "sca", "ssa", "gwo", "bka"]
# algorithms = [ "woa", "ja", "pso", "sca", "ssa", "gwo", "bka"]
# Function to run the algorithm and collect metrics
def run_algorithm(algo, train_index, test_index):
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
    sensitivity = recall_score(y_valid, y_pred, pos_label='AD')
    precision = precision_score(y_valid, y_pred, pos_label='AD')
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

# Run the algorithms with 10-fold cross-validation and collect results
results = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation
if 1:
    for algo in algorithms:
        print(f"Running experiment with algorithm: {algo}")
        fold_results = []
        for i in range(10):
            print(f"Running time {i + 1}/10")
            for fold, (train_index, test_index) in enumerate(kf.split(feat)):
                print(f"Running fold {fold + 1}/10")
                result = run_algorithm(algo, train_index, test_index)
                result['Fold'] = fold + 1  # Add fold number to result
                result['Run'] = i + 1
                fold_results.append(result)
        results.extend(fold_results)

    # Save results to JSON file
    with open("experiment_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Experiment results saved to experiment_results.json")

if 0:
    # Function to run the algorithm and collect metrics
    def run_algorithm_with_indices(args):
        algo, train_index, test_index, fold, run = args
        result = run_algorithm(algo, train_index, test_index)
        result['Fold'] = fold + 1  # Add fold number to result
        result['Run'] = run + 1
        return result

    # Run the algorithms with 10-fold cross-validation and collect results
    results = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation

    for algo in algorithms:
        print(f"Running experiment with algorithm: {algo}")
        fold_results = []
        for i in tqdm(range(10), desc=f"Run {i + 1}/10", unit="run"):
            print(f"Running time {i + 1}/10")
            args_list = [(algo, train_index, test_index, fold, i) for fold, (train_index, test_index) in enumerate(kf.split(feat))]
            with Pool() as pool:
                fold_results.extend(pool.map(run_algorithm_with_indices, args_list))
        results.extend(fold_results)

    # Save results to JSON file
    with open("experiment_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Experiment results saved to experiment_results.json")