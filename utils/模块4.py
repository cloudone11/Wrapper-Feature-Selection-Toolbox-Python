import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 使用鸢尾花数据集作为示例
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# 使用信息增益进行特征选择
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mi_selector.fit(X, y)
mi_scores = mi_selector.scores_

# 使用卡方检验进行特征选择
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X, y)
chi2_scores = chi2_selector.scores_

# 如果 X 是 numpy 数组，先将其转换为 DataFrame
if isinstance(X, np.ndarray):
    X = pd.DataFrame(X)

# 打印特征评分
for i in range(len(X.columns)):
    print(f"Feature: {X.columns[i]}, MI Score: {mi_scores[i]}, Chi2 Score: {chi2_scores[i]}")

# 根据评分选择前k个特征（例如，选择前2个特征）
k = 2
mi_top_k_features = X.columns[mi_selector.get_support(indices=True)[:k]]
chi2_top_k_features = X.columns[chi2_selector.get_support(indices=True)[:k]]

print(f"Top {k} features based on MI: {mi_top_k_features}")
print(f"Top {k} features based on Chi2: {chi2_top_k_features}")

# 根据评分选择前k个特征（例如，选择前2个特征）进行SVM分类
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 使用 Chi2 选择的特征进行 SVM 分类
X_chi2_selected = chi2_selector.fit_transform(X, y)
X_train_chi2, X_test_chi2, y_train, y_test = train_test_split(X_chi2_selected, y, test_size=0.2, random_state=42)
svm_chi2 = SVC()
svm_chi2.fit(X_train_chi2, y_train)
y_pred_chi2 = svm_chi2.predict(X_test_chi2)
print(f"Accuracy with Chi2 selected features: {accuracy_score(y_test, y_pred_chi2)}")