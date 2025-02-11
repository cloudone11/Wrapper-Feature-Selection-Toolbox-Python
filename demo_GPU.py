import cupy as cp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 假设数据集为170维，两类数据
# 随机生成模拟数据
np.random.seed(42)
X = np.random.rand(1000, 170)  # 1000个样本，170维特征
y = np.random.randint(0, 2, 1000)  # 两类标签

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为CuPy数组以利用GPU加速
X_train_gpu = cp.array(X_train)
X_test_gpu = cp.array(X_test)
y_train_gpu = cp.array(y_train)

# 群体优化算法（简化版）进行特征选择
# 假设我们使用一个简单的遗传算法作为群体优化算法
def genetic_algorithm(X, y, population_size=50, generations=100, mutation_rate=0.01):
    num_features = X.shape[1]
    population = cp.random.randint(2, size=(population_size, num_features))  # 初始化种群
    best_fitness = cp.inf
    best_individual = None

    for generation in range(generations):
        fitness = cp.zeros(population_size)
        for i in range(population_size):
            selected_features = cp.where(population[i] == 1)[0]
            if len(selected_features) == 0:
                fitness[i] = cp.inf
            else:
                X_selected = X[:, selected_features]
                X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
                    X_selected.get(), y.get(), test_size=0.2, random_state=42)
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(X_train_selected, y_train_selected)
                y_pred = knn.predict(X_test_selected)
                fitness[i] = 1 - accuracy_score(y_test_selected, y_pred)

        # 选择最佳个体
        best_idx = cp.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_individual = population[best_idx]

        # 选择、交叉和变异
        # 简化实现，仅展示基本逻辑
        new_population = cp.copy(population)
        for i in range(population_size):
            parent1 = population[cp.random.randint(population_size)]
            parent2 = population[cp.random.randint(population_size)]
            crossover_point = cp.random.randint(num_features)
            new_population[i, :crossover_point] = parent1[:crossover_point]
            new_population[i, crossover_point:] = parent2[crossover_point:]
            for j in range(num_features):
                if cp.random.rand() < mutation_rate:
                    new_population[i, j] = 1 - new_population[i, j]

        population = new_population

    return best_individual

# 使用遗传算法选择特征
best_features = genetic_algorithm(X_train_gpu, y_train_gpu)
selected_features = cp.where(best_features == 1)[0]

# 使用选择的特征训练KNN分类器
X_train_selected = X_train_gpu[:, selected_features].get()
X_test_selected = X_test_gpu[:, selected_features].get()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_selected, y_train)

# 在测试集上评估分类器性能
y_pred = knn.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")